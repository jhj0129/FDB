from __future__ import annotations

import argparse
from pathlib import Path
import time

from .data import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB v5 MuJoCo 학습 데이터 생성")
    parser.add_argument("--output", type=Path, default=Path("data/v5/push_dynamics.npz"))
    parser.add_argument("--train", type=int, default=2400)
    parser.add_argument("--validation", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--ood", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    started = time.perf_counter()
    dataset = generate_dataset(
        train_count=args.train,
        validation_count=args.validation,
        test_count=args.test,
        ood_count=args.ood,
        seed=args.seed,
    )
    dataset.save(args.output)
    elapsed = time.perf_counter() - started
    print(f"저장: {args.output}")
    print(f"표본: {len(dataset.features)}, 경과: {elapsed:.2f}초")


if __name__ == "__main__":
    main()
