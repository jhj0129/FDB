from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .data import PushDataset
from .network import PushDynamicsMLP


def _metrics(prediction: np.ndarray, target: np.ndarray, features: np.ndarray) -> dict[str, float]:
    error = prediction - target
    distances = np.linalg.norm(error, axis=1)
    predicted_goal_distance = np.linalg.norm(prediction - features[:, 2:4], axis=1)
    actual_goal_distance = np.linalg.norm(target - features[:, 2:4], axis=1)
    return {
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(math.sqrt(np.mean(error**2))),
        "mean_position_error_m": float(np.mean(distances)),
        "p95_position_error_m": float(np.quantile(distances, 0.95)),
        "success_classification_accuracy": float(
            np.mean((predicted_goal_distance <= 0.14) == (actual_goal_distance <= 0.14))
        ),
    }


def train_ensemble(
    dataset_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    portable_path: Path | None = None,
    *,
    ensemble_size: int = 5,
    hidden_size: int = 64,
    epochs: int = 120,
    learning_rate: float = 2e-3,
    seed: int = 20260917,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    dataset = PushDataset.load(dataset_path)
    train_mask = dataset.split == "train"
    validation_mask = dataset.split == "validation"
    train_x, train_y = dataset.features[train_mask], dataset.targets[train_mask]
    validation_x, validation_y = dataset.features[validation_mask], dataset.targets[validation_mask]
    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6)
    target_mean = train_y.mean(axis=0)
    target_std = np.maximum(train_y.std(axis=0), 1e-6)

    normalized_x = torch.from_numpy((train_x - feature_mean) / feature_std)
    normalized_y = torch.from_numpy((train_y - target_mean) / target_std)
    val_x_tensor = torch.from_numpy((validation_x - feature_mean) / feature_std)
    val_y_tensor = torch.from_numpy((validation_y - target_mean) / target_std)
    loss_function = torch.nn.MSELoss()
    state_dicts: list[dict[str, torch.Tensor]] = []
    best_validation_losses: list[float] = []
    for member in range(ensemble_size):
        torch.manual_seed(seed + member)
        model = PushDynamicsMLP(input_size=train_x.shape[1], hidden_size=hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        generator = torch.Generator().manual_seed(seed + 1000 + member)
        bootstrap = torch.randint(len(normalized_x), (len(normalized_x),), generator=generator)
        member_x, member_y = normalized_x[bootstrap], normalized_y[bootstrap]
        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(len(member_x), generator=generator)
            for start in range(0, len(member_x), 128):
                indices = permutation[start : start + 128]
                optimizer.zero_grad()
                loss = loss_function(model(member_x[indices]), member_y[indices])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(model(val_x_tensor), val_y_tensor))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                print(
                    f"앙상블 {member + 1}/{ensemble_size} epoch {epoch}/{epochs} "
                    f"검증손실 {validation_loss:.6f}",
                    flush=True,
                )
        assert best_state is not None
        state_dicts.append(best_state)
        best_validation_losses.append(best_loss)

    def ensemble_predict(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.from_numpy((features - feature_mean) / feature_std)
        outputs = []
        with torch.no_grad():
            for state in state_dicts:
                model = PushDynamicsMLP(input_size=train_x.shape[1], hidden_size=hidden_size)
                model.load_state_dict(state)
                model.eval()
                outputs.append(model(tensor).numpy() * target_std + target_mean)
        stacked = np.stack(outputs)
        return stacked.mean(axis=0), stacked.std(axis=0)

    validation_prediction, validation_std = ensemble_predict(validation_x)
    uncertainty_threshold = max(0.002, float(np.quantile(np.max(validation_std, axis=1), 0.95)))
    checkpoint = {
        "format_version": "1.0",
        "model_type": "PushDynamicsMLPEnsemble",
        "input_size": train_x.shape[1],
        "hidden_size": hidden_size,
        "ensemble_size": ensemble_size,
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "feature_min": train_x.min(axis=0),
        "feature_max": train_x.max(axis=0),
        "uncertainty_threshold_m": uncertainty_threshold,
        "state_dicts": state_dicts,
        "seed": seed,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    if portable_path is not None:
        portable: dict[str, Any] = {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "feature_min": train_x.min(axis=0),
            "feature_max": train_x.max(axis=0),
            "uncertainty_threshold_m": np.asarray([uncertainty_threshold], dtype=np.float32),
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
        "parameter_count_per_member": sum(value.numel() for value in state_dicts[0].values()),
        "ensemble_size": ensemble_size,
        "train_count": int(train_mask.sum()),
        "validation_count": int(validation_mask.sum()),
        "best_validation_losses": best_validation_losses,
        "uncertainty_threshold_m": uncertainty_threshold,
        "portable_model": str(portable_path) if portable_path else None,
        "splits": {},
    }
    for split_name in ("validation", "test", "ood"):
        mask = dataset.split == split_name
        prediction, uncertainty = ensemble_predict(dataset.features[mask])
        split_metrics = _metrics(prediction, dataset.targets[mask], dataset.features[mask])
        split_metrics["mean_uncertainty_m"] = float(np.mean(np.max(uncertainty, axis=1)))
        split_metrics["fallback_rate"] = float(
            np.mean(np.max(uncertainty, axis=1) > uncertainty_threshold)
        )
        results["splits"][split_name] = split_metrics

    # A deliberately simple constant baseline makes learned improvement explicit.
    test_mask = dataset.split == "test"
    constant = np.repeat(train_y.mean(axis=0, keepdims=True), int(test_mask.sum()), axis=0)
    results["constant_baseline_test"] = _metrics(
        constant, dataset.targets[test_mask], dataset.features[test_mask]
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results
