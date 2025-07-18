# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 19:48:08 2025

@author: LvXiang
"""

# models/diff_layers.py

import torch
import torch.nn as nn
from abc import abstractmethod
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def norm_layer(dim: int) -> nn.LayerNorm:
    """LayerNorm 包装"""
    return nn.LayerNorm(dim)

# ——— 时序与条件模块 —————————————————————————————

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

# define TimestepEmbedSequential to support `time_emb` as extra input
class ConditioningBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, cd):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock, ConditioningBlock):
    """
    A Sequential container that routes inputs to child layers differently:
      - If the layer is a TimestepBlock, call layer(x, t)
      - If the layer is a ConditioningBlock, call layer(x, cond)
      - Otherwise, call layer(x)
    
    This allows interleaving of time-embedding-aware, conditioning-aware,
    and standard layers in one list.)
    """
    def forward(self, x, t, cond):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t)
            elif isinstance(layer, ConditioningBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x

# ——— 基础构件 —————————————————————————————————————————

class ResidualBlock(TimestepBlock):
    """
    A ResNet-style block that incorporates timestep embeddings.
    """
    def __init__(self, in_dims, out_dims, time_dims, dropout):
        super().__init__()
        self.Linear1 = nn.Sequential(
            norm_layer(in_dims),
            nn.SiLU(),
            nn.Linear(in_dims, out_dims)
        )
        
        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dims, out_dims)
        )
        
        self.Linear2 = nn.Sequential(
            norm_layer(out_dims),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dims, out_dims)
        )

        if in_dims != out_dims:
            self.shortcut = nn.Linear(in_dims, out_dims)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        """
        Args:
            x: Tensor of shape [batch*seq_len, in_dims]
            t: Tensor of shape [batch, time_dims]
        
        Returns:
            Tensor of shape [batch*seq_len, out_dims]
        """
        h = self.Linear1(x)
        h += self.time_emb(t)[:, None,:]
        h = self.Linear2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block with residual connection.
    """
    def __init__(self, dims, num_heads=1,dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        assert dims % num_heads == 0
        self.normq = nn.LayerNorm(dims)
        self.normk = nn.LayerNorm(dims)
        self.normv = nn.LayerNorm(dims)
        self.att = nn.MultiheadAttention(dims, num_heads, batch_first=True,dropout=dropout)
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, dims]
        Returns:
            Tensor of same shape, with self-attention applied and added residually.
        """
        q = self.normq(x)
        k = self.normk(x)
        v = self.normv(x)
        h,_=self.att(q,k,v)
        return h + x

class CrossAttentionBlock(ConditioningBlock):
    """
    Multi-head cross-attention block for conditioning on external embeddings.
    """
    def __init__(self, query_dim, key_value_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert query_dim % num_heads == 0
        assert key_value_dim % num_heads == 0
        self.normq = nn.LayerNorm(query_dim)
        self.prok =  nn.LayerNorm(key_value_dim)

        self.prov = nn.LayerNorm(key_value_dim)
        
        self.att = nn.MultiheadAttention(query_dim, num_heads, batch_first=True, kdim=key_value_dim, vdim=key_value_dim)

    def forward(self, query, key_value):
        """
        Args:
            query: Tensor of shape (batch_size, query_len, query_dim)
            key_value: Tensor of shape (batch_size, key_value_len, key_value_dim)
            attention_mask: Tensor of shape (batch_size, key_value_len) with 1 for valid and 0 for invalid positions
        """
        key_value_embeds, attention_mask = key_value
        q = self.normq(query)
        k = self.prok(key_value_embeds)
        v = self.prov(key_value_embeds)
        # Prepare attention mask for multi-head attention
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)  # Ensure boolean type
        # Apply multi-head attention
        h, _ = self.att(q, k, v, key_padding_mask=~attention_mask if attention_mask is not None else None)
        return h + query
