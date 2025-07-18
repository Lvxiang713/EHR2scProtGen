# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 19:51:31 2025

@author: LvXiang
"""
# models/scAttNet.py

import torch
import torch.nn as nn
from .diff_layers import (
    norm_layer,
    ResidualBlock,
    AttentionBlock,
    CrossAttentionBlock,
    TimestepEmbedSequential,
    timestep_embedding
)

class SingleCellAN(nn.Module):
    """
    The main U-Net–style diffusion model for single-cell protein generation,
    conditioned on EHR text embeddings via cross-attention.
    
    Args:
        feature_dims (int): Number of protein features per cell.
        EHR_embdims  (int): Dimensionality of the EHR text embeddings.
        model_dims   (int): Base channel width for all internal projections.
        dims_mult    (tuple): Multipliers for widening at each U-Net stage.
        num_res_blocks (int): Number of ResidualBlocks per level.
        attention_resolutions (tuple): Which downsampling factors include attention.
        dropout      (float): Dropout probability inside ResidualBlocks.
        dropoutAtt   (float): Dropout probability inside attention layers.
        num_heads    (int): Number of heads in all multi-head attention blocks.
    """
    def __init__(
        self,
        feature_dims = 36,
        EHR_embdims = 128,
        model_dims = 512,
        dims_mult=(1, 2, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(2, 4, 8, 16),
        dropout=0.0,
        dropoutAtt=0.1,
        num_heads=4,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        time_embed_dim = model_dims * 4
        self.model_dims = model_dims
        # Embedding incoming scalar protein measurements to model_dims
        self.proteinEmb = nn.Sequential(
            nn.Linear(1,model_dims),
            nn.LayerNorm(model_dims),
            nn.SiLU(),
            nn.Linear(model_dims,model_dims))
        # Initial transform to add positional context
        self.InitEmb = nn.Sequential(
            nn.Linear(model_dims,model_dims),
            nn.LayerNorm(model_dims),
            nn.SiLU(),
            nn.Linear(model_dims,model_dims))
        # Learnable 2D positional embedding: one vector per feature index
        self.position_emb = nn.Parameter(torch.zeros(feature_dims,model_dims))
        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Linear(model_dims, model_dims))
        ])
        # Time-embedding network: projects model_dims → 4×model_dims
        self.time_embed = nn.Sequential(
            nn.Linear(model_dims, time_embed_dim),
            nn.LayerNorm(time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # Build the “down” path of the U-Net
        down_block_dims = [model_dims]
        ch = model_dims
        ds = 1
        for level, mult in enumerate(dims_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_dims, time_embed_dim, dropout)
                ]
                ch = mult * model_dims
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads,dropout=dropoutAtt))
                    layers.append(CrossAttentionBlock(ch, EHR_embdims, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_dims.append(ch)
            if level != len(dims_mult) - 1: # don't use downsample for the last stage
                ds *= 2
                
         # Central “bottleneck” block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads,dropout=dropoutAtt),
            CrossAttentionBlock(ch, EHR_embdims, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )
    
        # Build the “up” path of the U-Net (mirror of down path)
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(dims_mult))[::-1]:
            for i in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        ch + down_block_dims.pop(),
                        model_dims * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_dims * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads,dropout=dropoutAtt))
                    layers.append(CrossAttentionBlock(ch, EHR_embdims, num_heads=num_heads))
                if level and i == num_res_blocks:
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        layers = [
            ResidualBlock(
                ch + down_block_dims.pop(),
                model_dims * mult,
                time_embed_dim,
                dropout
            )
        ]
        self.up_blocks.append(TimestepEmbedSequential(*layers))
        # Final block to bring channels back to 1 per feature
        self.out =  nn.Sequential(
                nn.Linear(model_dims, model_dims),
                norm_layer(model_dims),
                nn.SiLU(),
                nn.Linear(model_dims, model_dims),
                norm_layer(model_dims),
                nn.SiLU(),
                nn.Linear(model_dims, 1))
        
    def forward(self, x, timesteps,cd):
        """
        Forward pass through the conditioned diffusion U-Net.

        Args:
            x: Tensor of shape [batch, cell_num, feature_dims] — the noisy data.
            timesteps: 1-D Tensor [batch] of diffusion time steps.
            cd: Tuple (EHR_embeds, attention_mask):
                - EHR_embeds: [batch, 1, EHR_embdims]
                - attention_mask: [batch, 1] boolean mask

        Returns:
            Tensor of shape [batch, cell_num, feature_dims] — the denoised output.
        """
        hs = []
        # time step embedding
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_dims))
        EHR_embeds, attention_mask = cd
        B = x.shape[0]
        cell_num = x.shape[1] 
        time_emb = time_emb.unsqueeze(1).expand(-1,cell_num,-1).reshape(B*cell_num,-1)
        EHR_embeds = EHR_embeds.unsqueeze(1).expand(-1,cell_num,-1,-1).reshape(B*cell_num,EHR_embeds.shape[-2],-1)
        attention_mask = attention_mask.unsqueeze(1).expand(-1,cell_num,-1).reshape(B*cell_num,-1)
        x = x.reshape(B*cell_num,-1).unsqueeze(-1)
        x = self.proteinEmb(x)
        x = self.InitEmb(x + self.position_emb.unsqueeze(0))
        h = x
        for module in self.down_blocks:
            h = module(h, time_emb, (EHR_embeds, attention_mask))
            hs.append(h)
        h = self.middle_block(h, time_emb, (EHR_embeds, attention_mask))
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, time_emb, (EHR_embeds, attention_mask))   
        out = self.out(h).squeeze(-1).reshape([B,cell_num,-1])
        return out


