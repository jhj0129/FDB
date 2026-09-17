from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SafetyAdvice:
    safe_probability: float
    uncertainty: float
    in_distribution: bool
    learned_gate_allows: bool
    requires_physics_check: bool
    reasons: tuple[str, ...]


class PortableSafetyEnsemble:
    """PyTorch 없이 실행되는 보수적인 조작 자세 안전 조언기."""

    def __init__(self, path: Path) -> None:
        with np.load(path) as payload:
            self.feature_mean = payload["feature_mean"].copy()
            self.feature_std = payload["feature_std"].copy()
            self.feature_min = payload["feature_min"].copy()
            self.feature_max = payload["feature_max"].copy()
            self.safe_probability_threshold = float(payload["safe_probability_threshold"][0])
            self.uncertainty_threshold = float(payload["uncertainty_threshold"][0])
            ensemble_size = int(payload["ensemble_size"][0])
            self.members = tuple(
                tuple(
                    (
                        payload[f"member_{member}_weight_{layer}"].copy(),
                        payload[f"member_{member}_bias_{layer}"].copy(),
                    )
                    for layer in range(3)
                )
                for member in range(ensemble_size)
            )

    @staticmethod
    def _silu(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return values / (1.0 + np.exp(-clipped))

    def advise(self, features: np.ndarray | list[float]) -> SafetyAdvice:
        vector = np.asarray(features, dtype=np.float32)
        normalized = (vector - self.feature_mean) / self.feature_std
        probabilities = []
        for layers in self.members:
            value = normalized
            for index, (weight, bias) in enumerate(layers):
                value = weight @ value + bias
                if index < 2:
                    value = self._silu(value)
            probabilities.append(float(1.0 / (1.0 + np.exp(-np.clip(value.item(), -40.0, 40.0)))))
        probability = float(np.mean(probabilities))
        uncertainty = float(np.std(probabilities))
        in_distribution = bool(
            np.all(vector >= self.feature_min) and np.all(vector <= self.feature_max)
        )
        reasons: list[str] = []
        if not in_distribution:
            reasons.append("학습 분포 밖 자세")
        if probability < self.safe_probability_threshold:
            reasons.append("안전 확률 문턱 미달")
        if uncertainty > self.uncertainty_threshold:
            reasons.append("앙상블 불확실성 초과")
        learned_gate_allows = not reasons
        # 허용 판정도 물리 검증을 면제하지 않는다. 거부 판정은 조기에 중단할 수 있다.
        return SafetyAdvice(
            safe_probability=probability,
            uncertainty=uncertainty,
            in_distribution=in_distribution,
            learned_gate_allows=learned_gate_allows,
            requires_physics_check=learned_gate_allows,
            reasons=tuple(reasons),
        )
