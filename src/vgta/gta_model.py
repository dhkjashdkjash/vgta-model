from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def build_gta_inputs(
    values: np.ndarray,
    vim_prior: np.ndarray,
) -> np.ndarray:

    values = np.asarray(values, dtype=float)
    prior = np.asarray(vim_prior, dtype=float)
    if values.ndim != 2 or prior.shape != values.shape:
        raise ValueError("GTA arrays must both have shape [B,T]")
    observed = np.isfinite(values)
    prior_available = np.isfinite(prior)
    return np.concatenate(
        (
            np.where(observed, values, 0.0)[..., None],
            observed.astype(float)[..., None],
            np.where(prior_available, prior, 0.0)[..., None],
            prior_available.astype(float)[..., None],
        ),
        axis=-1,
    )


def masked_reconstruction_loss(prediction: Tensor, target: Tensor, supervised: Tensor) -> Tensor:
    if prediction.shape != target.shape or supervised.shape != target.shape:
        raise ValueError("prediction, target, and supervised mask must share a shape")
    selected = supervised.bool() & torch.isfinite(target)
    if not torch.any(selected):
        raise ValueError("no supervised reconstruction positions")
    return F.mse_loss(prediction[selected], target[selected])


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution1 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=0
        )
        self.convolution2 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=0
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout1 = nn.Dropout1d(dropout)
        self.dropout2 = nn.Dropout1d(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        value = self.convolution1(F.pad(sequence, (self.left_padding, 0)))
        value = self.norm1(value.transpose(1, 2)).transpose(1, 2)
        value = self.dropout1(F.relu(value))
        value = self.convolution2(F.pad(value, (self.left_padding, 0)))
        value = self.norm2(value.transpose(1, 2)).transpose(1, 2)
        value = self.dropout2(F.relu(value))
        return sequence + value


class GTA(nn.Module):
    def __init__(
        self,
        *,
        input_features: int = 4,
        filters: int = 24,
        blocks: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.05,
        attention_heads: int = 2,
        key_channels: int = 5,
    ) -> None:
        super().__init__()
        if not 2 <= blocks <= 4:
            raise ValueError("GTA requires two to four residual blocks")
        self.input_projection = nn.Conv1d(input_features, filters, 1)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(filters, kernel_size, 2 ** index, dropout)
            for index in range(blocks)
        )
        attention_dim = attention_heads * key_channels
        self.attention_input = nn.Linear(filters, attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Linear(attention_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("GTA input must have shape [batch, time, features]")
        sequence = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            sequence = block(sequence)
        tokens = self.attention_input(sequence.transpose(1, 2))
        causal = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        attended, _ = self.attention(tokens, tokens, tokens, attn_mask=causal, need_weights=False)
        return self.output(attended).squeeze(-1)
