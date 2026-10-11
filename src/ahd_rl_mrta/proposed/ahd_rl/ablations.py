"""与论文中的 AHD-RL-NCA、AHD-RL-NRT、AHD-RL-NCR 对应的消融网络。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DM
from .model import SchedulingNetwork


ABLATION_VARIANTS = ("ahd-rl-nca", "ahd-rl-nrt", "ahd-rl-ncr")


class _IdentityCrossDecoder(nn.Module):
    """NCA 用恒等映射代替任务与智能体之间的交叉注意力。"""

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del context, context_key_padding_mask
        return x


class _InputMaskedLinear(nn.Linear):
    """在线性投影前屏蔽指定特征列，同时保持原模型的参数形状。"""

    @classmethod
    def from_linear(
        cls, source: nn.Linear, masked_indices: list[int]
    ) -> "_InputMaskedLinear":
        """复制原投影参数，并建立不写入权重文件的输入掩码。"""
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
    """保持主网络结构，仅移除指定消融项对应的信息路径。"""

    def __init__(self, variant: str, dropout: Optional[float] = None):
        if variant not in ABLATION_VARIANTS:
            raise ValueError("unknown ablation variant: {}".format(variant))
        self.ablation_variant = variant
        super().__init__(dropout=dropout)

        if self.ablation_variant == "ahd-rl-nca":
            # 移除双向交叉注意力；两个编码器仍分别处理任务和智能体。
            self.cross_t2r = _IdentityCrossDecoder()
            self.cross_r2t = _IdentityCrossDecoder()
        elif self.ablation_variant == "ahd-rl-nrt":
            # 两种任务特征中的参考联盟信息相同，但列位置因决策阶段而异。
            self.task_emb_sel = _InputMaskedLinear.from_linear(
                self.task_emb_sel, [6, 7, 12, 13, 14]
            )
            self.task_emb_rob = _InputMaskedLinear.from_linear(
                self.task_emb_rob, [11, 12, 17, 18, 19]
            )
        elif self.ablation_variant == "ahd-rl-ncr":
            # 屏蔽已选成员标记，以及共享查询中的当前联盟嵌入。
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
