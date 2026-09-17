from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .safety_data import ManipulationSafetyDataset, SAFETY_FEATURE_NAMES


class TableSafetyMLP(torch.nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 32) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def _decision_metrics(predicted_safe: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    safe = target.astype(bool)
    true_safe = int(np.sum(predicted_safe & safe))
    false_safe = int(np.sum(predicted_safe & ~safe))
    false_unsafe = int(np.sum(~predicted_safe & safe))
    true_unsafe = int(np.sum(~predicted_safe & ~safe))
    return {
        "count": int(len(target)),
        "accuracy": float(np.mean(predicted_safe == safe)),
        "safe_precision": float(true_safe / max(1, true_safe + false_safe)),
        "safe_recall": float(true_safe / max(1, true_safe + false_unsafe)),
        "false_safe_count": false_safe,
        "false_safe_rate": float(false_safe / max(1, false_safe + true_unsafe)),
        "false_unsafe_count": false_unsafe,
        "confusion": {
            "true_safe": true_safe,
            "false_safe": false_safe,
            "false_unsafe": false_unsafe,
            "true_unsafe": true_unsafe,
        },
    }


def _binary_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, Any]:
    return _decision_metrics(probability >= threshold, target)


def train_safety_ensemble(
    dataset_path: Path,
    checkpoint_path: Path,
    portable_path: Path,
    metrics_path: Path,
    *,
    ensemble_size: int = 5,
    hidden_size: int = 32,
    epochs: int = 240,
    learning_rate: float = 3e-3,
    seed: int = 20260918,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    dataset = ManipulationSafetyDataset.load(dataset_path)
    train_mask = dataset.split == "train"
    validation_mask = dataset.split == "validation"
    train_x = dataset.features[train_mask]
    train_y = dataset.safe[train_mask].astype(np.float32)
    validation_x = dataset.features[validation_mask]
    validation_y = dataset.safe[validation_mask].astype(np.float32)
    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6)
    normalized_train_x = torch.from_numpy((train_x - feature_mean) / feature_std)
    train_y_tensor = torch.from_numpy(train_y)
    normalized_validation_x = torch.from_numpy((validation_x - feature_mean) / feature_std)
    validation_y_tensor = torch.from_numpy(validation_y)
    unsafe_count = max(1, int(np.sum(~dataset.safe[train_mask])))
    safe_count = max(1, int(np.sum(dataset.safe[train_mask])))
    # 위험 표본을 놓치는 오류를 더 크게 벌점화한다.
    sample_weights = torch.where(
        train_y_tensor > 0.5,
        torch.tensor(1.0),
        torch.tensor(1.5 * safe_count / unsafe_count),
    )
    state_dicts: list[dict[str, torch.Tensor]] = []
    best_losses: list[float] = []
    for member in range(ensemble_size):
        torch.manual_seed(seed + member)
        model = TableSafetyMLP(hidden_size=hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed + 1000 + member)
        bootstrap = torch.randint(len(train_x), (len(train_x),), generator=generator)
        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(normalized_train_x[bootstrap])
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, train_y_tensor[bootstrap], reduction="none"
            )
            loss = torch.mean(losses * sample_weights[bootstrap])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        model(normalized_validation_x), validation_y_tensor
                    )
                )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if epoch == 1 or epoch % 40 == 0 or epoch == epochs:
                print(
                    f"안전망 {member + 1}/{ensemble_size} epoch {epoch}/{epochs} "
                    f"검증손실 {validation_loss:.6f}",
                    flush=True,
                )
        assert best_state is not None
        state_dicts.append(best_state)
        best_losses.append(best_loss)

    def predict(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inputs = torch.from_numpy((features - feature_mean) / feature_std)
        probabilities = []
        with torch.no_grad():
            for state in state_dicts:
                model = TableSafetyMLP(hidden_size=hidden_size)
                model.load_state_dict(state)
                model.eval()
                probabilities.append(torch.sigmoid(model(inputs)).numpy())
        stacked = np.stack(probabilities)
        return stacked.mean(axis=0), stacked.std(axis=0)

    validation_probability, validation_uncertainty = predict(validation_x)
    unsafe_validation = validation_probability[validation_y < 0.5]
    # 검증 위험 자세 중 가장 낙관적인 값보다도 높은 보수적 허용 문턱이다.
    safe_probability_threshold = float(
        min(0.995, max(0.5, float(unsafe_validation.max(initial=0.0)) + 0.02))
    )
    uncertainty_threshold = float(
        max(0.01, np.quantile(validation_uncertainty, 0.95))
    )
    checkpoint = {
        "format_version": "1.0",
        "model_type": "TableSafetyMLPEnsemble",
        "feature_names": list(SAFETY_FEATURE_NAMES),
        "input_size": 4,
        "hidden_size": hidden_size,
        "ensemble_size": ensemble_size,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_min": train_x.min(axis=0),
        "feature_max": train_x.max(axis=0),
        "safe_probability_threshold": safe_probability_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "state_dicts": state_dicts,
        "seed": seed,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    portable: dict[str, Any] = {
        "feature_names": np.asarray(SAFETY_FEATURE_NAMES),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_min": train_x.min(axis=0),
        "feature_max": train_x.max(axis=0),
        "safe_probability_threshold": np.asarray([safe_probability_threshold], dtype=np.float32),
        "uncertainty_threshold": np.asarray([uncertainty_threshold], dtype=np.float32),
        "ensemble_size": np.asarray([ensemble_size], dtype=np.int64),
    }
    for member, state in enumerate(state_dicts):
        for layer_index, layer_name in enumerate(("layers.0", "layers.2", "layers.4")):
            portable[f"member_{member}_weight_{layer_index}"] = state[f"{layer_name}.weight"].numpy()
            portable[f"member_{member}_bias_{layer_index}"] = state[f"{layer_name}.bias"].numpy()
    portable_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(portable_path, **portable)

    results: dict[str, Any] = {
        "model_type": checkpoint["model_type"],
        "dataset_count": int(len(dataset.features)),
        "safe_count": int(np.sum(dataset.safe)),
        "unsafe_count": int(np.sum(~dataset.safe)),
        "parameter_count_per_member": sum(value.numel() for value in state_dicts[0].values()),
        "ensemble_size": ensemble_size,
        "best_validation_losses": best_losses,
        "safe_probability_threshold": safe_probability_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "policy": "학습망은 사전 거부만 조언하며 MuJoCo 접촉 하드 게이트를 대체하지 않음",
        "splits": {},
    }
    for split_name in ("train", "validation", "test"):
        mask = dataset.split == split_name
        features = dataset.features[mask]
        probability, uncertainty = predict(features)
        raw_metrics = _binary_metrics(probability, dataset.safe[mask], safe_probability_threshold)
        in_distribution = np.all(
            (features >= checkpoint["feature_min"]) & (features <= checkpoint["feature_max"]),
            axis=1,
        )
        policy_allows = (
            (probability >= safe_probability_threshold)
            & (uncertainty <= uncertainty_threshold)
            & in_distribution
        )
        metrics = _decision_metrics(policy_allows, dataset.safe[mask])
        metrics["raw_probability_gate"] = raw_metrics
        metrics["mean_probability"] = float(np.mean(probability))
        metrics["mean_uncertainty"] = float(np.mean(uncertainty))
        metrics["uncertainty_fallback_rate"] = float(
            np.mean(uncertainty > uncertainty_threshold)
        )
        results["splits"][split_name] = metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Panda 테이블 접촉 안전 앙상블 학습")
    parser.add_argument("--dataset", type=Path, default=Path("data/v5/panda_table_safety.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/v5/panda_table_safety.pt"))
    parser.add_argument("--portable", type=Path, default=Path("models/v5/panda_table_safety.npz"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/0016_v5_safety_classifier.metrics.json"))
    args = parser.parse_args()
    results = train_safety_ensemble(args.dataset, args.checkpoint, args.portable, args.metrics)
    test = results["splits"]["test"]
    print(
        f"완료: 최종 안전정책 테스트 정확도 {test['accuracy']:.3f}, "
        f"위험 통과 {test['false_safe_count']}건 "
        f"(확률만 사용 시 {test['raw_probability_gate']['false_safe_count']}건)",
        flush=True,
    )


if __name__ == "__main__":
    main()
