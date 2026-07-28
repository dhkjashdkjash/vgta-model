from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class VimOutput:
    concentration: Tensor
    logits: Tensor
    embedding: Tensor


class SelectiveSSM(nn.Module):


    def __init__(self, dim: int, state_dim: int = 8, kernel_size: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.input_projection = nn.Linear(dim, 2 * dim)
        self.local_conv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size - 1, groups=dim
        )
        self.parameter_projection = nn.Linear(dim, 1 + 2 * state_dim)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float()).repeat(dim, 1))
        self.skip = nn.Parameter(torch.ones(dim))
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, tokens: Tensor) -> Tensor:
        projected, gate = self.input_projection(tokens).chunk(2, dim=-1)
        local = self.local_conv(projected.transpose(1, 2))[..., : tokens.shape[1]]
        local = F.silu(local.transpose(1, 2))
        parameters = self.parameter_projection(local)
        delta_raw, b_values, c_values = torch.split(
            parameters, (1, self.state_dim, self.state_dim), dim=-1
        )
        delta = F.softplus(delta_raw).unsqueeze(-1)
        a = -torch.exp(self.a_log).view(1, 1, self.dim, self.state_dim)
        state = torch.zeros(
            tokens.shape[0], self.dim, self.state_dim, device=tokens.device, dtype=tokens.dtype
        )
        outputs = []
        for step in range(tokens.shape[1]):
            dt = delta[:, step]
            transition = torch.exp(dt * a[:, 0])
            input_term = dt * b_values[:, step].unsqueeze(1) * local[:, step].unsqueeze(-1)
            state = transition * state + input_term
            value = (state * c_values[:, step].unsqueeze(1)).sum(-1)
            value = value + self.skip * local[:, step]
            outputs.append(value)
        scanned = torch.stack(outputs, dim=1)
        return self.output_projection(scanned * F.silu(gate))


class BidirectionalMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        state_dim: int = 8,
        bidirectional: bool = True,
        selective_ssm: bool = True,
        gating: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.bidirectional = bidirectional
        self.selective_ssm = selective_ssm
        self.gating = gating
        self.forward_scan = SelectiveSSM(dim, state_dim)
        self.backward_scan = SelectiveSSM(dim, state_dim) if bidirectional else None
        self.nonselective = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.SiLU(), nn.Conv1d(dim, dim, 1)
        )
        self.merge_gate = nn.Linear(dim, dim)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm(tokens)
        if self.selective_ssm:
            forward = self.forward_scan(normalized)
        else:
            forward = self.nonselective(normalized.transpose(1, 2)).transpose(1, 2)
        if not self.bidirectional:
            merged = forward
        else:
            backward_input = torch.flip(normalized, dims=(1,))
            if self.selective_ssm:
                backward = self.backward_scan(backward_input)
            else:
                backward = self.nonselective(backward_input.transpose(1, 2)).transpose(1, 2)
            backward = torch.flip(backward, dims=(1,))
            if self.gating:
                weight = torch.sigmoid(self.merge_gate(normalized))
                merged = weight * forward + (1.0 - weight) * backward
            else:
                merged = 0.5 * (forward + backward)
        return tokens + merged


class VisionMamba20(nn.Module):


    def __init__(
        self,
        *,
        channels: int = 20,
        classes: int = 4,
        dim: int = 192,
        depth: int = 24,
        patch_size: int = 16,
        state_dim: int = 16,
        max_tokens: int = 1025,
        bidirectional: bool = True,
        positional_embedding: bool = True,
        selective_ssm: bool = True,
        gating: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.positional_embedding = positional_embedding
        self.patch_embedding = nn.Conv2d(channels, dim, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, max_tokens, dim))
        self.blocks = nn.ModuleList(
            BidirectionalMambaBlock(
                dim,
                state_dim=state_dim,
                bidirectional=bidirectional,
                selective_ssm=selective_ssm,
                gating=gating,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.regression_head = nn.Linear(dim, 1)
        self.classification_head = nn.Linear(dim, classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, features: Tensor) -> VimOutput:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(f"VisionMamba20 expects {self.channels} channels in [B,C,H,W]")
        patches = self.patch_embedding(features).flatten(2).transpose(1, 2)
        token = self.class_token.expand(features.shape[0], -1, -1)
        tokens = torch.cat((token, patches), dim=1)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("input patch grid exceeds configured max_tokens")
        if self.positional_embedding:
            tokens = tokens + self.position[:, : tokens.shape[1]]
        for block in self.blocks:
            tokens = block(tokens)
        embedding = self.norm(tokens[:, 0])
        concentration = self.regression_head(embedding).squeeze(-1)
        logits = self.classification_head(embedding)
        return VimOutput(concentration=concentration, logits=logits, embedding=embedding)


def vim_loss(
    output: VimOutput,
    *,
    concentration: Tensor,
    labels: Tensor,
    regression_weight: float = 1.0,
    classification_weight: float = 1.0,
) -> Tensor:
    regression = F.mse_loss(output.concentration, concentration)
    classification = F.cross_entropy(output.logits, labels)
    return regression_weight * regression + classification_weight * classification
