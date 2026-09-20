from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class PushDynamicsMLP(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


@dataclass(frozen=True)
class NeuralPrediction:
    final_position: tuple[float, float]
    uncertainty_m: float
    in_distribution: bool
    confident: bool
    use_physics_fallback: bool
    fallback_reasons: tuple[str, ...]


class NeuralDynamicsEnsemble:
    def __init__(self, checkpoint: dict[str, Any], *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        self.target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
        self.feature_min = np.asarray(checkpoint["feature_min"], dtype=np.float32)
        self.feature_max = np.asarray(checkpoint["feature_max"], dtype=np.float32)
        self.uncertainty_threshold_m = float(checkpoint["uncertainty_threshold_m"])
        self.models: list[PushDynamicsMLP] = []
        for state in checkpoint["state_dicts"]:
            model = PushDynamicsMLP(
                input_size=int(checkpoint["input_size"]),
                hidden_size=int(checkpoint["hidden_size"]),
            )
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.models.append(model)

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "NeuralDynamicsEnsemble":
        return cls(torch.load(path, map_location="cpu", weights_only=False), device=device)

    def predict(self, features: np.ndarray) -> NeuralPrediction:
        vector = np.asarray(features, dtype=np.float32)
        in_distribution = bool(
            np.all(vector >= self.feature_min) and np.all(vector <= self.feature_max)
        )
        normalized = (vector - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(normalized[None, :]).to(self.device)
        with torch.no_grad():
            normalized_predictions = torch.stack([model(tensor)[0] for model in self.models])
        predictions = normalized_predictions.cpu().numpy() * self.target_std + self.target_mean
        mean = predictions.mean(axis=0)
        uncertainty = float(np.max(predictions.std(axis=0)))
        confident = uncertainty <= self.uncertainty_threshold_m
        reasons: list[str] = []
        if not in_distribution:
            reasons.append("학습 분포 밖 입력")
        if not confident:
            reasons.append("앙상블 불확실성 초과")
        return NeuralPrediction(
            final_position=(float(mean[0]), float(mean[1])),
            uncertainty_m=uncertainty,
            in_distribution=in_distribution,
            confident=confident,
            use_physics_fallback=bool(reasons),
            fallback_reasons=tuple(reasons),
        )
