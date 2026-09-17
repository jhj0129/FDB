"""FDB v5: 학습형 동역학 예측과 검증된 물리 fallback."""

from .data import FEATURE_NAMES, TARGET_NAMES, PushDataset, generate_dataset

__all__ = ["FEATURE_NAMES", "TARGET_NAMES", "PushDataset", "generate_dataset"]
