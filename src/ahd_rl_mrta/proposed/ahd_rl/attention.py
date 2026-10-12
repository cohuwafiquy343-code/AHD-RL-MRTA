from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.Wg = nn.Linear(d_model, d_hidden, bias=False)
        self.Wv = nn.Linear(d_model, d_hidden, bias=False)
        self.W2 = nn.Linear(d_hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.drop(F.silu(self.Wg(x)) * self.Wv(x)))


class PreLNLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = GatedFFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False,
        )
        x = x + self.drop(attn_out)
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int,
                 d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [PreLNLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i == 0 and attn_bias is not None:
                x = layer(x, key_padding_mask, attn_bias=attn_bias)
            else:
                x = layer(x, key_padding_mask)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = GatedFFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.ln_q(x)
        kv = self.ln_kv(context)
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=context_key_padding_mask)
        x = x + self.drop(attn_out)
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class CrossDecoder(nn.Module):
    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context, context_key_padding_mask)
        return x
