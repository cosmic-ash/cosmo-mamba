"""
Vision Transformer baseline for cosmological parameter inference.
Patches a 256×256 map into tokens, processes with standard transformer
encoder, and regresses (Omega_m, sigma_8) with uncertainty.
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=384):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) → (B, n_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTRegressor(nn.Module):
    """
    Standard ViT adapted for regression on cosmological maps.
    Uses a [CLS] token and outputs mean + log-variance.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_params: int = 2,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean_out = nn.Linear(embed_dim // 2, n_params)
        self.logvar_out = nn.Linear(embed_dim // 2, n_params)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.encoder(tokens)
        cls_out = self.norm(tokens[:, 0])
        h = self.head(cls_out)
        return self.mean_out(h), self.logvar_out(h)
