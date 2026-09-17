from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import train_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB v5 신경망 앙상블 학습")
    parser.add_argument("--data", type=Path, default=Path("data/v5/push_dynamics.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/v5/push_dynamics_ensemble.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/0014_v5_neural_metrics.json"))
    parser.add_argument("--portable", type=Path, default=Path("models/v5/push_dynamics_ensemble.npz"))
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()
    results = train_ensemble(
        args.data,
        args.checkpoint,
        args.metrics,
        portable_path=args.portable,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
