from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch


class BalanceMLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size), torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size), torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def _metrics(allowed: np.ndarray, stable: np.ndarray) -> dict[str, float | int]:
    false_stable = int(np.sum(allowed & ~stable))
    false_fall = int(np.sum(~allowed & stable))
    return {
        "count": int(len(stable)),
        "accuracy": float(np.mean(allowed == stable)),
        "false_stable_count": false_stable,
        "false_fall_count": false_fall,
        "stable_precision": float(np.sum(allowed & stable) / max(1, int(np.sum(allowed)))),
        "stable_recall": float(np.sum(allowed & stable) / max(1, int(np.sum(stable)))),
    }


def train(
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
    *,
    ensemble_size: int = 5,
    epochs: int = 220,
    seed: int = 20260917,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with np.load(dataset_path) as payload:
        features = payload["features"].copy()
        stable = payload["stable"].copy()
        split = payload["split"].copy()
        robots = payload["robot"].copy()
        feature_names = payload["feature_names"].copy()
    train_mask, validation_mask = split == "train", split == "validation"
    mean = features[train_mask].mean(axis=0)
    std = np.maximum(features[train_mask].std(axis=0), 1e-6)
    train_x = torch.from_numpy((features[train_mask] - mean) / std)
    train_y = torch.from_numpy(stable[train_mask].astype(np.float32))
    validation_x = torch.from_numpy((features[validation_mask] - mean) / std)
    validation_y = torch.from_numpy(stable[validation_mask].astype(np.float32))
    states = []
    for member in range(ensemble_size):
        torch.manual_seed(seed + member)
        model = BalanceMLP(features.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed + 1000 + member)
        bootstrap = torch.randint(len(train_x), (len(train_x),), generator=generator)
        best_loss, best_state = float("inf"), None
        for epoch in range(1, epochs + 1):
            model.train(); optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(train_x[bootstrap]), train_y[bootstrap]
            )
            loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
                    model(validation_x), validation_y
                ))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if epoch == 1 or epoch % 55 == 0 or epoch == epochs:
                print(f"균형망 {member + 1}/5 epoch {epoch}/{epochs} 검증손실 {val_loss:.5f}", flush=True)
        states.append(best_state)

    def predict(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.from_numpy((values - mean) / std)
        outputs = []
        with torch.no_grad():
            for state in states:
                model = BalanceMLP(features.shape[1]); model.load_state_dict(state); model.eval()
                outputs.append(torch.sigmoid(model(tensor)).numpy())
        stacked = np.stack(outputs)
        return stacked.mean(axis=0), stacked.std(axis=0)

    val_probability, val_uncertainty = predict(features[validation_mask])
    unstable_probabilities = val_probability[~stable[validation_mask]]
    # 작은 다중 로봇 자료에서는 검증 낙상 확률 최대치만으로는 지나치게 낙관적이다.
    # 안정 승인은 90% 이상의 합의까지 요구하고 나머지는 물리 검증으로 보낸다.
    probability_threshold = float(min(0.995, max(0.9, unstable_probabilities.max(initial=0.0) + 0.03)))
    uncertainty_threshold = float(max(0.02, np.quantile(val_uncertainty, 0.9)))
    portable: dict[str, object] = {
        "feature_names": feature_names, "feature_mean": mean, "feature_std": std,
        "feature_min": features[train_mask].min(axis=0), "feature_max": features[train_mask].max(axis=0),
        "stable_probability_threshold": np.asarray([probability_threshold], dtype=np.float32),
        "uncertainty_threshold": np.asarray([uncertainty_threshold], dtype=np.float32),
        "ensemble_size": np.asarray([ensemble_size], dtype=np.int64),
    }
    for member, state in enumerate(states):
        for layer_index, layer_name in enumerate(("layers.0", "layers.2", "layers.4")):
            portable[f"member_{member}_weight_{layer_index}"] = state[f"{layer_name}.weight"].numpy()
            portable[f"member_{member}_bias_{layer_index}"] = state[f"{layer_name}.bias"].numpy()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, **portable)
    report: dict[str, object] = {
        "model_type": "MultiHumanoidBalanceMLPEnsemble",
        "ensemble_size": ensemble_size,
        "parameter_count_per_member": sum(value.numel() for value in states[0].values()),
        "stable_probability_threshold": probability_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "policy": "90% 안정 합의와 불확실성 gate를 모두 통과해야 하며, 허용 후에도 물리 검증 필요",
        "splits": {}, "test_by_robot": {},
    }
    for split_name in ("train", "validation", "test"):
        mask = split == split_name
        probability, uncertainty = predict(features[mask])
        in_distribution = np.all((features[mask] >= portable["feature_min"]) & (features[mask] <= portable["feature_max"]), axis=1)
        allowed = (probability >= probability_threshold) & (uncertainty <= uncertainty_threshold) & in_distribution
        report["splits"][split_name] = {**_metrics(allowed, stable[mask]), "fallback_rate": float(np.mean(~allowed))}
    for robot in np.unique(robots):
        mask = (split == "test") & (robots == robot)
        probability, uncertainty = predict(features[mask])
        in_distribution = np.all((features[mask] >= portable["feature_min"]) & (features[mask] <= portable["feature_max"]), axis=1)
        allowed = (probability >= probability_threshold) & (uncertainty <= uncertainty_threshold) & in_distribution
        report["test_by_robot"][str(robot)] = _metrics(allowed, stable[mask])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="다중 휴머노이드 낙상 예측 앙상블 학습")
    parser.add_argument("--dataset", type=Path, default=Path("data/v5/humanoid_balance.npz"))
    parser.add_argument("--model", type=Path, default=Path("models/v5/humanoid_balance_ensemble.npz"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/0018_humanoid_balance.metrics.json"))
    args = parser.parse_args()
    report = train(args.dataset, args.model, args.metrics)
    test = report["splits"]["test"]
    print(f"완료: 테스트 정확도 {test['accuracy']:.3f}, 낙상을 안정으로 오판 {test['false_stable_count']}건", flush=True)


if __name__ == "__main__":
    main()
