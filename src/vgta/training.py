from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from vgta.vim_model import VisionMamba20, vim_loss


@dataclass(frozen=True)
class SymmetricScaler:
    minimum: float
    maximum: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "SymmetricScaler":
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("cannot fit symmetric scaling without finite source values")
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        if maximum <= minimum:
            raise ValueError("symmetric scaling requires a nonzero source range")
        return cls(minimum=minimum, maximum=maximum)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return 2.0 * (values - self.minimum) / (self.maximum - self.minimum) - 1.0

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return (values + 1.0) * 0.5 * (self.maximum - self.minimum) + self.minimum


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_vim(
    model: VisionMamba20,
    features: np.ndarray,
    concentration: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> VisionMamba20:
    seed_everything(seed)
    model.train()
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(concentration, dtype=torch.float32)
    classes = torch.as_tensor(labels, dtype=torch.long)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = vim_loss(
            model(x), concentration=y, labels=classes
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def predict_vim(
    model: VisionMamba20, features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = model(torch.as_tensor(features, dtype=torch.float32))
    concentration = output.concentration.cpu().numpy()
    probabilities = torch.softmax(output.logits, dim=-1).cpu().numpy()
    labels = probabilities.argmax(axis=-1)
    return concentration, probabilities, labels
