from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DM
from .model import SchedulingNetwork

ABLATION_VARIANTS = ("ahd-rl-nca", "ahd-rl-nrt", "ahd-rl-ncr")


class _IdentityCrossDecoder(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del context, context_key_padding_mask
        return x


class _InputMaskedLinear(nn.Linear):
    @classmethod
    def from_linear(
        cls, source: nn.Linear, masked_indices: list[int]
    ) -> "_InputMaskedLinear":
        layer = cls(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
        )
        with torch.no_grad():
            layer.weight.copy_(source.weight)
            if source.bias is not None and layer.bias is not None:
                layer.bias.copy_(source.bias)
        mask = torch.ones(source.in_features, dtype=source.weight.dtype)
        mask[masked_indices] = 0.0
        layer.register_buffer("input_mask", mask, persistent=False)
        return layer

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        mask = self.input_mask.to(dtype=input_tensor.dtype)
        return F.linear(input_tensor * mask, self.weight, self.bias)


class AblationSchedulingNetwork(SchedulingNetwork):
    def __init__(self, variant: str, dropout: Optional[float] = None):
        if variant not in ABLATION_VARIANTS:
            raise ValueError("unknown ablation variant: {}".format(variant))
        self.ablation_variant = variant
        super().__init__(dropout=dropout)

        if self.ablation_variant == "ahd-rl-nca":
            # NCA removes both directions of task-agent cross-attention.
            self.cross_t2r = _IdentityCrossDecoder()
            self.cross_r2t = _IdentityCrossDecoder()
        elif self.ablation_variant == "ahd-rl-nrt":
            # The two decision phases place reference timing in different columns.
            self.task_emb_sel = _InputMaskedLinear.from_linear(
                self.task_emb_sel, [6, 7, 12, 13, 14]
            )
            self.task_emb_rob = _InputMaskedLinear.from_linear(
                self.task_emb_rob, [11, 12, 17, 18, 19]
            )
        elif self.ablation_variant == "ahd-rl-ncr":
            # NCR removes member flags and the current coalition embedding.
            self.robot_emb_rob = _InputMaskedLinear.from_linear(
                self.robot_emb_rob, [15]
            )
            self.W_global = _InputMaskedLinear.from_linear(
                self.W_global, list(range(2 * DM, 3 * DM))
            )
            self.W_proj = nn.Identity()
            self.ln_coal_proj = nn.Identity()

    def update_coal_state(
        self,
        coal_indices: list[int],
        h_r2_sq: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.ablation_variant == "ahd-rl-ncr":
            del coal_indices, h_r2_sq
            return torch.zeros(DM, device=device)
        return super().update_coal_state(coal_indices, h_r2_sq, device)
