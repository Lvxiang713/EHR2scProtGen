# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:27:20 2025

@author: LvXiang
"""
import torch
import torch.nn as nn

class LayerNorm(nn.LayerNorm):
    """
    LayerNorm that promotes to float32 for stability
    then casts back to original dtype.
    """
    def forward(self, x: torch.Tensor):
        orig = x.dtype
        ret = super().forward(x.float())
        return ret.to(orig)

class QuickGELU(nn.Module):
    """Fast GELU approximation: x * sigmoid(1.702 * x)."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """
    Single block of a Transformer:
      - Pre-LN
      - Multihead self-attention
      - Residual add
      - MLP (4× expansion) + residual
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            QuickGELU(),
            nn.Linear(d_model*4, d_model)
        )


    def attention(self, x, key_padding_mask=None):
        return self.attn(x, x, x, need_weights=False,key_padding_mask=key_padding_mask)[0]
    
    def forward(self, x: torch.Tensor, key_padding_mask=None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """Stack of ResidualAttentionBlock for sequence modeling."""
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])
 
 
    def forward(self, x, key_padding_mask=None):
        for block in self.resblocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x

