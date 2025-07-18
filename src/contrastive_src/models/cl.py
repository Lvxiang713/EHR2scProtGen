# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:35:17 2025

@author: LvXiang
"""
import torch
import torch.nn as nn
import numpy as np
from .CyTOF_encoder import CyTOF_encoder
from .layers import Transformer, LayerNorm

class CL(nn.Module):
    """
    Contrastive model mapping CyTOF “images” and EHR “texts”
    into the same feature space for downstream alignment and classification.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        v = cfg['vision']
        # Use the renamed encoder and pull out_dim from vision config
        self.visual = CyTOF_encoder(
            in_channels   = v['in_channels'],    # number of CyTOF features
            embed_dim     = v['width'],          # token embedding size
            quantile_len  = v['quantile_len'],   # number of quantile tokens
            depth         = v['depth'],          # transformer layers in vision branch
            heads         = v['heads'],          # attention heads
            mlp_ratio     = v['mlp_ratio'],      # feed-forward expansion
            dropout       = v['dropout'],        # dropout rate
            cut_len       = v['cut_len'],        # patch height/width
            cut_stepsize  = v['cut_step'],       # patch stride
            cnn_depth     = v['cnn_depth'],      # number of CNN blocks
            cnn_hidden    = v['cnn_hidden'],     # CNN channels
            cnn_kernel    = v['cnn_kernel_size'],# CNN kernel size
            out_dim       = v['out_dim']         # output embedding dimension
        )

        t = cfg['text']
        self.transformer = Transformer(
            width  = t['transformer_width'],    # hidden size
            layers = t['transformer_layers'],   # number of layers
            heads  = t['transformer_heads']     # attention heads
        )
        # EHR → transformer tokens
        self.ehr_projection = nn.Linear(1, t['transformer_width'])
        self.ehr_cls_token  = nn.Parameter(torch.zeros(1,1,t['transformer_width']))
        self.ehr_pos_embed  = nn.Parameter(
            torch.zeros(1, 1 + t['feature_dim'], t['transformer_width'])
        )
        self.ln_final = LayerNorm(t['transformer_width'])

        # Final projection: use the same out_dim
        self.text_projection = nn.Parameter(
            torch.empty(t['transformer_width'], v['out_dim'])
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self._init_text()

    def _init_text(self):
        """Initialize EHR token & projection weights."""
        nn.init.trunc_normal_(self.ehr_cls_token, std=0.02)
        nn.init.trunc_normal_(self.ehr_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.ehr_projection.weight)
        nn.init.xavier_uniform_(self.text_projection)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode CyTOF tensor → image embedding."""
        return self.visual(x)

    def encode_text(self, ehr_data: torch.Tensor) -> torch.Tensor:
        """
        Encode EHR numerical features:
          1) reshape → (B, features)
          2) mask NaNs to build padding mask
          3) project each scalar → token
          4) prepend class token and add positional embeddings
          5) run through Transformer with mask
          6) take [cls] token, layer-norm, then final projection
        """
        # ensure shape (B, feature_dim)
        ehr = ehr_data.view(ehr_data.size(0), -1)
        B, _ = ehr.shape
        device = ehr.device

        # build padding mask: True where missing
        missing = torch.isnan(ehr)
        cls_mask = torch.zeros((B, 1), dtype=missing.dtype, device=device)
        key_padding_mask = torch.cat([cls_mask, missing], dim=1)
        # transpose if needed
        if key_padding_mask.shape[0] != B:
            key_padding_mask = key_padding_mask.transpose(0, 1)

        # replace NaN with 0 before projection
        ehr_clean = torch.where(missing, torch.zeros_like(ehr), ehr)
        x = ehr_clean.unsqueeze(-1)            # (B, feature_dim, 1)
        x = self.ehr_projection(x)             # → (B, feature_dim, width)

        # prepend class token and add positional embeddings
        cls = self.ehr_cls_token.expand(B, -1, -1)  # (B,1,width)
        x = torch.cat([cls, x], dim=1) + self.ehr_pos_embed

        # Transformer encoding
        x = self.transformer(x, key_padding_mask=key_padding_mask)

        # take class token, normalize, project
        cls_out = self.ln_final(x[:, 0])
        return cls_out @ self.text_projection  # (B, out_dim)

    def forward(self, img: torch.Tensor, ehr: torch.Tensor):
        """
        Compute both embeddings and normalize:
          - image_emb: from CyTOF data
          - text_emb:  from EHR data
        Returns (image_emb, text_emb).
        """
        img_emb = self.encode_image(img)
        txt_emb = self.encode_text(ehr)

        img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)
        return img_emb, txt_emb