from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(seed: int = 17) -> tuple[dict[str, object], dict[str, object]]:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prototypes = np.array([
        [0.95, 0.08, 0.04, 0.018, 0.018, 0.022],
        [0.08, 0.85, 0.15, 0.023, 0.017, 0.022],
        [0.08, 0.25, 0.95, 0.017, 0.017, 0.029],
    ], dtype=np.float32)
    labels = ("red_small", "green_wide", "blue_tall")
    features, targets = [], []
    for label, prototype in enumerate(prototypes):
        samples = np.repeat(prototype[None, :], 240, axis=0)
        samples[:, :3] += rng.normal(0.0, 0.055, size=(240, 3))
        samples[:, 3:] += rng.normal(0.0, 0.0018, size=(240, 3))
        features.append(samples)
        targets.append(np.full(240, label, dtype=np.int64))
    x = np.concatenate(features)
    y = np.concatenate(targets)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    split = 576
    mean, std = x[:split].mean(axis=0), x[:split].std(axis=0) + 1e-6
    train_x = torch.tensor((x[:split] - mean) / std)
    train_y = torch.tensor(y[:split])
    test_x = torch.tensor((x[split:] - mean) / std)
    test_y = torch.tensor(y[split:])
    ensemble = []
    predictions = []
    for member in range(5):
        torch.manual_seed(seed + member)
        network = torch.nn.Sequential(
            torch.nn.Linear(6, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 3),
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
        for _ in range(180):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(network(train_x), train_y)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            predictions.append(torch.softmax(network(test_x), dim=1).numpy())
        ensemble.append(network.state_dict())
    probabilities = np.mean(predictions, axis=0)
    predicted = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predicted == y[split:]))
    confidence = probabilities.max(axis=1)
    metrics = {
        "dataset": {"train": split, "test": len(x) - split, "classes": list(labels)},
        "features": ["red", "green", "blue", "half_x", "half_y", "half_z"],
        "ensemble_members": 5,
        "test_accuracy": accuracy,
        "minimum_test_confidence": float(confidence.min()),
        "mean_test_confidence": float(confidence.mean()),
        "scope": "합성 색·크기 잡음 범위 안의 세 클래스 후보 모델",
        "hard_gate": "분류 결과와 무관하게 테이블 충돌·안전 높이는 MuJoCo가 최종 검증",
    }
    checkpoint = {
        "state_dicts": ensemble,
        "mean": torch.tensor(mean),
        "std": torch.tensor(std),
        "labels": labels,
        "prototypes": torch.tensor(prototypes),
        "ood_scale": torch.tensor([0.055, 0.055, 0.055, 0.0018, 0.0018, 0.0018]),
        "metrics": metrics,
    }
    return metrics, checkpoint


def main() -> None:
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description="다중 물체 색·형상 MLP 앙상블 학습")
    parser.add_argument("--model", type=Path, default=Path("models/v5/object_sort_ensemble.pt"))
    parser.add_argument("--portable", type=Path, default=Path("models/v5/object_sort_ensemble.npz"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/0019_object_sort_classifier.metrics.json"))
    args = parser.parse_args()
    metrics, checkpoint = train()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model)
    arrays: dict[str, object] = {
        "mean": checkpoint["mean"].numpy(),
        "std": checkpoint["std"].numpy(),
        "labels": np.asarray(checkpoint["labels"]),
        "prototypes": checkpoint["prototypes"].numpy(),
        "ood_scale": checkpoint["ood_scale"].numpy(),
    }
    for member, state in enumerate(checkpoint["state_dicts"]):
        for key, value in state.items():
            arrays[f"member_{member}_{key.replace('.', '_')}"] = value.detach().numpy()
    args.portable.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.portable, **arrays)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metrics, "model": str(args.model), "portable": str(args.portable), "metrics": str(args.metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
