# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:28:38 2025

@author: LvXiang
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from .layers import LayerNorm

class CyTOF_encoder(nn.Module):
    """
    Encoder for CyTOF single-cell quantile patches:

      1) Compute quantiles per cell → (batch, Q, feat_dim)
      2) Build gradient maps via Sobel filters
      3) Pass through CNN layers, collect each conv output
      4) Extract sliding patches, flatten & embed
      5) Prepend class token + positional embedding
      6) Transformer encoder → LayerNorm + head → logits
    """
    def __init__(self,
                 out_dim:     int,
                 in_channels: int=36,
                 embed_dim:   int=128,
                 quantile_len:int=256,
                 depth:       int=6,
                 heads:       int=8,
                 mlp_ratio:   float=4,
                 dropout:     float=0.1,
                 cut_len:     int=16,
                 cut_stepsize:int=16,
                 cnn_depth:   int=4,
                 cnn_hidden:  int=32,
                 cnn_kernel:  int=3,
                 ):
        super().__init__()

        # 1) Prepare Sobel kernels
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
        ky = torch.tensor([[-1.,-2.,-1.],[ 0., 0., 0.],[ 1., 2., 1.]]).view(1,1,3,3)
        kx = kx.repeat(in_channels, 1, 1, 1)  # shape: (numFeas, 1, 3, 3)
        ky = ky.repeat(in_channels, 1, 1, 1)
        self.register_buffer('sobel_kernel_x',kx)
        self.register_buffer('sobel_kernel_y',ky)

        # 2) CNN feature extractor
        layers = []
        curr_ch = in_channels
        for _ in range(cnn_depth):
            layers += [
                nn.Conv2d(curr_ch, cnn_hidden, cnn_kernel, padding=cnn_kernel//2),
                nn.InstanceNorm2d(cnn_hidden),
                nn.ReLU()
            ]
            curr_ch = cnn_hidden
        self.cnnList = nn.ModuleList(layers)

        # 3) Store params for sliding-window & quantile
        self.quantile_len  = quantile_len
        self.cut_len       = cut_len
        self.cut_stepsize  = cut_stepsize
        self.q = torch.linspace(0.01, 0.99, quantile_len)
        # 4) Patch embedding
        patch_dim = cnn_hidden * cut_len * cut_len
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        # 5) Class token
        # number of windows per axis
        num_windows = ((quantile_len - cut_len)//cut_stepsize + 1)**2
        # total_patches = num_windows * cnn_depth
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+num_windows, embed_dim))
        # 6) Transformer stack and head
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)
        self._init_weights()
        # Initialize positional & token embeddings
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)


    def grad(self,x,conv_kernel):
        grad = F.conv2d(x, conv_kernel, bias=None, stride=1, padding=1, groups=x.shape[1])
        return grad
    
    
    def forward(self, x: torch.Tensor):
        """
        x: (batch, cell_count, feat_dim)
        returns: (batch, out_dim)
        """
        batch_size, _, _ = x.shape

        # -- Quantile & gradient maps --
        xq = x.quantile(q=self.q.to(x.device), dim=1, keepdim=False)     # (Q, batch, feat_dim)
        xq = xq.permute(1, 0, 2)                        # (batch, Q, feat_dim)

        # build gradient magnitude maps
        diff = xq.unsqueeze(1) - xq.unsqueeze(2)        # (batch, Q, Q, feat_dim)
        feat_maps = diff.permute(0, 3, 1, 2)            # (batch, feat_dim, Q, Q)
        gx = self.grad(feat_maps, self.sobel_kernel_x)
        gy = self.grad(feat_maps, self.sobel_kernel_y)
        g  = torch.sqrt(gx**2 + gy**2)                 # (batch, feat_dim, Q, Q)

        # -- CNN & collect convolution outputs --
        conv_feats = []
        for layer in self.cnnList:
            g = layer(g)
            if isinstance(layer, nn.Conv2d):
                conv_feats.append(g)  # one entry per conv layer

        # -- Sliding-window patch extraction & embedding --
        patches = []
        for fmap in conv_feats:
            fmap = fmap.permute(0,2,3,1)
            p = fmap.unfold(1,self.cut_len,self.cut_stepsize).unfold(2,self.cut_len,self.cut_stepsize).flatten(-3)
            L0 = p.size(1)
            p = p.reshape(batch_size,L0*L0,-1)
            patches.append(p)
        tokens = torch.cat(patches, dim=1)              # (batch, total_patches, patch_dim)
        x_tokens = self.patch_embed(tokens)             # (batch, total_patches, embed_dim)

        # -- Add class token & positional embeddings --
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (batch,1,embed_dim)
        x_tokens = torch.cat([cls_token, x_tokens], dim=1)    # (batch,1+total_patches,embed_dim)
        x_tokens = self.pos_drop(x_tokens)

        # -- Transformer & final head --
        out = self.transformer_encoder(x_tokens)                # (batch,1+patches,embed_dim)
        cls_out = self.norm(out[:, 0])                  # (batch,embed_dim)
        return self.head(cls_out)                       # (batch,out_dim)