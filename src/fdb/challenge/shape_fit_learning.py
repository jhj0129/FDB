from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SHAPES = ("square", "circle", "triangle", "rectangle", "unknown")
KNOWN_SHAPE_COUNT = 4
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 64


@dataclass(frozen=True)
class SceneLabel:
    object_shape: int
    hole_shape: int
    fits: bool
    angle_delta_rad: float
    object_radius_px: float
    hole_radius_px: float
    object_angle_rad: float
    hole_angle_rad: float


def build_shape_fit_model():
    from flax import linen as nn
    import jax.numpy as jnp

    class ShapeEncoder(nn.Module):
        @nn.compact
        def __call__(self, image):
            value = nn.relu(nn.Conv(24, (5, 5), strides=(2, 2), padding="SAME")(image))
            value = nn.relu(nn.Conv(48, (3, 3), strides=(2, 2), padding="SAME")(value))
            value = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME")(value))
            value = value.reshape((value.shape[0], -1))
            return nn.relu(nn.Dense(128)(value))

    class ShapeFitCNN(nn.Module):
        @nn.compact
        def __call__(self, image):
            encoder = ShapeEncoder(name="shared_visual_encoder")
            object_features = encoder(image[:, :, :32, :])
            hole_features = encoder(image[:, :, 32:, :])
            object_head = nn.Dense(8, name="object_head")(object_features)
            hole_head = nn.Dense(8, name="hole_head")(hole_features)
            relation = nn.relu(nn.Dense(128)(
                jnp.concatenate((object_features, hole_features), axis=1)
            ))
            fit_logit = nn.Dense(1, name="fit_head")(relation)
            return jnp.concatenate((
                object_head[:, :5], hole_head[:, :5], fit_logit,
                object_head[:, 5:7], hole_head[:, 5:7],
                object_head[:, 7:8], hole_head[:, 7:8],
            ), axis=1)

    return ShapeFitCNN()


class ShapeFitPredictor:
    def __init__(self, model_path: Path) -> None:
        import jax
        import jax.numpy as jnp
        from flax.serialization import from_bytes

        self.model = build_shape_fit_model()
        template = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        )["params"]
        self.params = from_bytes(template, model_path.read_bytes())

    def predict(self, image: np.ndarray) -> dict[str, object]:
        import jax.numpy as jnp

        outputs = np.asarray(
            self.model.apply({"params": self.params}, jnp.asarray(image[None, ...]))
        )[0]
        return _decode_prediction(outputs)


def _decode_prediction(outputs: np.ndarray) -> dict[str, object]:
    object_shifted = outputs[:5] - outputs[:5].max()
    hole_shifted = outputs[5:10] - outputs[5:10].max()
    object_probability = np.exp(object_shifted) / np.exp(object_shifted).sum()
    hole_probability = np.exp(hole_shifted) / np.exp(hole_shifted).sum()
    object_index = int(object_probability.argmax())
    hole_index = int(hole_probability.argmax())
    raw_fit = bool(outputs[10] >= 0.0)
    safe_fit = raw_fit and object_index == hole_index and object_index < KNOWN_SHAPE_COUNT
    if safe_fit and object_index != 1:
        symmetry = (4, 1, 3, 2)[object_index]
        object_phase = math.atan2(float(outputs[11]), float(outputs[12]))
        hole_phase = math.atan2(float(outputs[13]), float(outputs[14]))
        period = 2 * math.pi / symmetry
        rotation = ((hole_phase - object_phase) / symmetry + period / 2) % period - period / 2
    else:
        rotation = 0.0
    return {
        "object_shape": SHAPES[object_index],
        "object_confidence": float(object_probability.max()),
        "hole_shape": SHAPES[hole_index],
        "hole_confidence": float(hole_probability.max()),
        "raw_neural_fit": raw_fit,
        "fits": safe_fit,
        "fit_probability": float(1.0 / (1.0 + math.exp(-float(np.clip(outputs[10], -40, 40))))),
        "rotation_rad": float(rotation),
        "estimated_object_radius_px": float(outputs[15] * 10.0),
        "estimated_hole_radius_px": float(outputs[16] * 10.0),
    }


def predict_scene(image: np.ndarray, model_path: Path) -> dict[str, object]:
    return ShapeFitPredictor(model_path).predict(image)


def _polygon(shape: int, center: tuple[int, int], radius: float, angle: float) -> np.ndarray:
    cx, cy = center
    if shape == 0:
        count, aspect = 4, 1.0
        angle += math.pi / 4
    elif shape == 2:
        count, aspect = 3, 1.0
        angle -= math.pi / 2
    elif shape == 3:
        corners = np.asarray([[-1.35, -0.72], [1.35, -0.72], [1.35, 0.72], [-1.35, 0.72]])
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        return np.rint(corners @ rotation.T * radius + [cx, cy]).astype(np.int32)
    else:
        count, aspect = 5, 1.0
    points = []
    for index in range(count):
        phase = angle + 2.0 * math.pi * index / count
        points.append((cx + radius * math.cos(phase), cy + radius * aspect * math.sin(phase)))
    return np.rint(points).astype(np.int32)


def _period(shape: int) -> float:
    return (math.pi / 2, 2 * math.pi, 2 * math.pi / 3, math.pi, 2 * math.pi)[shape]


def _wrapped_delta(object_angle: float, hole_angle: float, shape: int) -> float:
    if shape == 1 or shape >= KNOWN_SHAPE_COUNT:
        return 0.0
    period = _period(shape)
    return (hole_angle - object_angle + period / 2) % period - period / 2


def render_scene(rng: np.random.Generator) -> tuple[np.ndarray, SceneLabel]:
    object_shape = int(rng.choice(5, p=[0.22, 0.22, 0.22, 0.22, 0.12]))
    if rng.random() < 0.58 and object_shape < KNOWN_SHAPE_COUNT:
        hole_shape = object_shape
    else:
        hole_shape = int(rng.choice(5, p=[0.22, 0.22, 0.22, 0.22, 0.12]))
    object_radius = float(rng.uniform(5.0, 12.0))
    hole_radius = float(rng.uniform(6.5, 13.5))
    object_angle = float(rng.uniform(-math.pi, math.pi))
    hole_angle = float(rng.uniform(-math.pi, math.pi))
    object_center = (16 + int(rng.integers(-2, 3)), 16 + int(rng.integers(-2, 3)))
    hole_center = (48 + int(rng.integers(-2, 3)), 16 + int(rng.integers(-2, 3)))
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 0.16, dtype=np.float32)
    image += rng.normal(0.0, 0.018, image.shape).astype(np.float32)
    image[:, 31:33] = 0.32
    if object_shape == 1:
        cv2.circle(image, object_center, round(object_radius), (0.92, 0.18, 0.10), -1, lineType=cv2.LINE_AA)
    else:
        cv2.fillPoly(image, [_polygon(object_shape, object_center, object_radius, object_angle)],
                     (0.92, 0.18, 0.10), lineType=cv2.LINE_AA)
    if hole_shape == 1:
        cv2.circle(image, hole_center, round(hole_radius), (0.04, 0.04, 0.05), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, hole_center, round(hole_radius), (0.18, 0.60, 0.95), 2, lineType=cv2.LINE_AA)
    else:
        points = _polygon(hole_shape, hole_center, hole_radius, hole_angle)
        cv2.fillPoly(image, [points], (0.04, 0.04, 0.05), lineType=cv2.LINE_AA)
        cv2.polylines(image, [points], True, (0.18, 0.60, 0.95), 2, lineType=cv2.LINE_AA)
    if rng.random() < 0.25:
        x = int(rng.integers(2, 59))
        cv2.line(image, (x, 0), (x + int(rng.integers(-4, 5)), 31), (0.28, 0.28, 0.28), 1)
    fits = bool(
        object_shape == hole_shape
        and object_shape < KNOWN_SHAPE_COUNT
        and object_radius + 1.2 <= hole_radius
    )
    return np.clip(image, 0.0, 1.0), SceneLabel(
        object_shape, hole_shape, fits,
        _wrapped_delta(object_angle, hole_angle, object_shape) if fits else 0.0,
        object_radius, hole_radius, object_angle, hole_angle,
    )


def make_dataset(count: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    images, labels = zip(*(render_scene(rng) for _ in range(count)))
    return np.asarray(images, dtype=np.float32), {
        "object_shape": np.asarray([label.object_shape for label in labels], dtype=np.int32),
        "hole_shape": np.asarray([label.hole_shape for label in labels], dtype=np.int32),
        "fits": np.asarray([label.fits for label in labels], dtype=np.float32),
        "angle": np.asarray([label.angle_delta_rad for label in labels], dtype=np.float32),
        "object_orientation": np.asarray([
            (math.sin((4, 1, 3, 2, 1)[label.object_shape] * label.object_angle_rad),
             math.cos((4, 1, 3, 2, 1)[label.object_shape] * label.object_angle_rad))
            for label in labels
        ], dtype=np.float32),
        "hole_orientation": np.asarray([
            (math.sin((4, 1, 3, 2, 1)[label.hole_shape] * label.hole_angle_rad),
             math.cos((4, 1, 3, 2, 1)[label.hole_shape] * label.hole_angle_rad))
            for label in labels
        ], dtype=np.float32),
        "object_radius": np.asarray([label.object_radius_px / 10.0 for label in labels], dtype=np.float32),
        "hole_radius": np.asarray([label.hole_radius_px / 10.0 for label in labels], dtype=np.float32),
    }


def train(output: Path, metrics_path: Path, seed: int = 42) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from flax.serialization import to_bytes
    from flax.training import train_state
    import optax

    train_images, train_labels = make_dataset(6000, seed)
    test_images, test_labels = make_dataset(1500, seed + 1)
    model = build_shape_fit_model()
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(train_images[:1]))["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(learning_rate=2.5e-3, weight_decay=1e-5),
    )

    @jax.jit
    def step(current, images, object_shape, hole_shape, fits, object_orientation, hole_orientation,
             object_radius, hole_radius):
        def loss_fn(parameters):
            output_values = current.apply_fn({"params": parameters}, images)
            object_loss = optax.softmax_cross_entropy_with_integer_labels(output_values[:, :5], object_shape)
            hole_loss = optax.softmax_cross_entropy_with_integer_labels(output_values[:, 5:10], hole_shape)
            fit_loss = optax.sigmoid_binary_cross_entropy(output_values[:, 10], fits)
            object_mask = ((object_shape < 4) & (object_shape != 1)).astype(jnp.float32)
            hole_mask = ((hole_shape < 4) & (hole_shape != 1)).astype(jnp.float32)
            object_angle_loss = object_mask * jnp.sum((output_values[:, 11:13] - object_orientation) ** 2, axis=1)
            hole_angle_loss = hole_mask * jnp.sum((output_values[:, 13:15] - hole_orientation) ** 2, axis=1)
            size_loss = (output_values[:, 15] - object_radius) ** 2 + (output_values[:, 16] - hole_radius) ** 2
            return jnp.mean(
                object_loss + hole_loss + fit_loss
                + 2.5 * object_angle_loss + 2.5 * hole_angle_loss + 0.8 * size_loss
            )
        loss, gradients = jax.value_and_grad(loss_fn)(current.params)
        return current.apply_gradients(grads=gradients), loss

    rng = np.random.default_rng(seed)
    batch_size = 128
    last_loss = math.inf
    for _ in range(60):
        for start in range(0, len(train_images), batch_size):
            indices = rng.integers(0, len(train_images), size=batch_size)
            state, loss = step(
                state,
                jnp.asarray(train_images[indices]),
                jnp.asarray(train_labels["object_shape"][indices]),
                jnp.asarray(train_labels["hole_shape"][indices]),
                jnp.asarray(train_labels["fits"][indices]),
                jnp.asarray(train_labels["object_orientation"][indices]),
                jnp.asarray(train_labels["hole_orientation"][indices]),
                jnp.asarray(train_labels["object_radius"][indices]),
                jnp.asarray(train_labels["hole_radius"][indices]),
            )
            last_loss = float(loss)
    outputs = np.asarray(model.apply({"params": state.params}, jnp.asarray(test_images)))
    object_prediction = outputs[:, :5].argmax(axis=1)
    hole_prediction = outputs[:, 5:10].argmax(axis=1)
    fit_prediction = outputs[:, 10] >= 0.0
    fit_mask = test_labels["fits"] > 0.5
    fit_shapes = test_labels["object_shape"][fit_mask]
    symmetries = np.asarray([4, 1, 3, 2], dtype=float)[fit_shapes]
    object_phase = np.arctan2(outputs[fit_mask, 11], outputs[fit_mask, 12])
    hole_phase = np.arctan2(outputs[fit_mask, 13], outputs[fit_mask, 14])
    predicted_angle = (hole_phase - object_phase) / symmetries
    periods = 2 * np.pi / symmetries
    predicted_angle = (predicted_angle + periods / 2) % periods - periods / 2
    predicted_angle[fit_shapes == 1] = 0.0
    angle_error = np.abs(
        (predicted_angle - test_labels["angle"][fit_mask] + periods / 2) % periods
        - periods / 2
    )
    metrics = {
        "model": "three-layer visual CNN with object, receptacle, fit, and alignment heads",
        "input": "32x64 RGB camera-like synthetic scenes",
        "classes": list(SHAPES),
        "train_samples": len(train_images),
        "test_samples": len(test_images),
        "object_shape_accuracy": float(np.mean(object_prediction == test_labels["object_shape"])),
        "hole_shape_accuracy": float(np.mean(hole_prediction == test_labels["hole_shape"])),
        "fit_decision_accuracy": float(np.mean(fit_prediction == fit_mask)),
        "fit_positive_count": int(np.sum(fit_mask)),
        "mean_alignment_error_deg": float(np.degrees(np.mean(angle_error))),
        "p95_alignment_error_deg": float(np.degrees(np.quantile(angle_error, 0.95))),
        "last_training_loss": last_loss,
        "scope_limit": "합성 장면에서의 시각·어포던스 파일럿이며 실제 카메라 일반화는 아직 미검증",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(to_bytes(state.params))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview = np.hstack([np.asarray(test_images[index] * 255, dtype=np.uint8) for index in range(6)])
    cv2.imwrite(str(metrics_path.with_suffix(".png")), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="영상 기반 물체-구멍 적합성 신경망 학습")
    parser.add_argument("--model", type=Path, default=Path("models/v6/shape_fit_cnn.msgpack"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/0029_shape_fit_cnn.metrics.json"))
    args = parser.parse_args()
    print(json.dumps(train(args.model, args.metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
