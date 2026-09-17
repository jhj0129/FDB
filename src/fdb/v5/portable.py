from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PortablePrediction:
    final_position: tuple[float, float]
    uncertainty_m: float
    in_distribution: bool
    use_physics_fallback: bool
    fallback_reasons: tuple[str, ...]


class PortableDynamicsEnsemble:
    """Torch 없이 MuJoCo 환경에서 실행되는 검증용 NumPy 추론기."""

    def __init__(self, path: Path) -> None:
        with np.load(path) as payload:
            self.feature_mean = payload["feature_mean"]
            self.feature_std = payload["feature_std"]
            self.target_mean = payload["target_mean"]
            self.target_std = payload["target_std"]
            self.feature_min = payload["feature_min"]
            self.feature_max = payload["feature_max"]
            self.uncertainty_threshold_m = float(payload["uncertainty_threshold_m"][0])
            ensemble_size = int(payload["ensemble_size"][0])
            self.members = tuple(
                tuple(
                    (payload[f"member_{member}_weight_{layer}"].copy(), payload[f"member_{member}_bias_{layer}"].copy())
                    for layer in range(3)
                )
                for member in range(ensemble_size)
            )

    @staticmethod
    def _silu(values: np.ndarray) -> np.ndarray:
        return values / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

    def predict(self, features: np.ndarray) -> PortablePrediction:
        vector = np.asarray(features, dtype=np.float32)
        normalized = (vector - self.feature_mean) / self.feature_std
        outputs = []
        for layers in self.members:
            value = normalized
            for index, (weight, bias) in enumerate(layers):
                value = weight @ value + bias
                if index < 2:
                    value = self._silu(value)
            outputs.append(value * self.target_std + self.target_mean)
        predictions = np.stack(outputs)
        mean = predictions.mean(axis=0)
        uncertainty = float(np.max(predictions.std(axis=0)))
        in_distribution = bool(
            np.all(vector >= self.feature_min) and np.all(vector <= self.feature_max)
        )
        reasons: list[str] = []
        if not in_distribution:
            reasons.append("학습 분포 밖 입력")
        if uncertainty > self.uncertainty_threshold_m:
            reasons.append("앙상블 불확실성 초과")
        return PortablePrediction(
            final_position=(float(mean[0]), float(mean[1])),
            uncertainty_m=uncertainty,
            in_distribution=in_distribution,
            use_physics_fallback=bool(reasons),
            fallback_reasons=tuple(reasons),
        )
