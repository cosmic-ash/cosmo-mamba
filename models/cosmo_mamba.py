"""
CosmoMamba: 2D State-Space Model for Cosmological Parameter Inference.

Adapts selective state-space models (Mamba) for 2D cosmological field maps.
Key idea: 2D maps are scanned along multiple directions (horizontal, vertical,
and both diagonals) to capture spatial structure, then processed by SSM blocks.
This gives O(n) complexity vs O(n^2) for attention, while preserving spatial
relationships that matter for large-scale structure.

Architecture:
  PatchEmbed → [MambaBlock × depth] → global pool → regression head

Each MambaBlock contains:
  1. Multi-directional 2D scan (4 scan routes)
  2. Selective SSM per direction
  3. Merge + project

This is a pure-PyTorch implementation that runs on any device.
When mamba-ssm is available on Linux+CUDA, the selective scan kernel
is automatically used for faster training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def _parallel_scan(gates, values):
    """Hillis-Steele parallel prefix scan for linear recurrence:
       h[t] = gates[t] * h[t-1] + values[t],  h[-1] = 0.
    Runs in O(log L) parallel steps instead of O(L) sequential steps."""
    B, L = gates.shape[:2]
    trailing = gates.shape[2:]
    log2L = int(math.ceil(math.log2(max(L, 2))))
    Lp = 2 ** log2L
    if Lp > L:
        pad_shape = (B, Lp - L) + trailing
        gates = torch.cat([gates, gates.new_ones(pad_shape)], dim=1)
        values = torch.cat([values, values.new_zeros(pad_shape)], dim=1)
    for d in range(log2L):
        k = 2 ** d
        g_shift = torch.cat([gates.new_ones(B, k, *trailing), gates[:, :-k]], dim=1)
        v_shift = torch.cat([values.new_zeros(B, k, *trailing), values[:, :-k]], dim=1)
        values = gates * v_shift + values
        gates = gates * g_shift
    return values[:, :L]


try:
    from mamba_ssm import Mamba as CUDAMamba
    HAS_CUDA_MAMBA = True
except ImportError:
    HAS_CUDA_MAMBA = False


class SelectiveSSM(nn.Module):
    """
    Pure-PyTorch selective state-space model block.
    Implements the core S6 selective scan in a sequential manner.
    Used as fallback when mamba-ssm CUDA kernels are not available.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand
        self.d_inner = d_inner

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_inner, -1)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(d_inner))
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, -4.0)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = rearrange(x_branch, "b l d -> b d l")
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = rearrange(x_branch, "b d l -> b l d")
        x_branch = F.silu(x_branch)

        proj = self.x_proj(x_branch)
        dt_input = proj[..., :1]
        B_param = proj[..., 1 : 1 + self.d_state]
        C_param = proj[..., 1 + self.d_state :]

        dt = F.softplus(self.dt_proj(dt_input))

        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(-2)
        x_scaled = x_branch.unsqueeze(-1) * dB

        h_all = _parallel_scan(dA.float(), x_scaled.float()).to(x_branch.dtype)
        y = (h_all * C_param.unsqueeze(-2)).sum(-1)
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = self.norm(y)
        y = y * F.silu(z)
        return self.out_proj(y)


def _get_ssm_block(d_model, d_state, d_conv, expand, use_cuda=True):
    if HAS_CUDA_MAMBA and use_cuda and torch.cuda.is_available():
        return CUDAMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return SelectiveSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class MultiDirectionalScan(nn.Module):
    """
    Scans a 2D feature map along 4 directions to capture spatial structure:
      1. Left-to-right (row-major)
      2. Right-to-left (reversed row-major)
      3. Top-to-bottom (column-major)
      4. Bottom-to-top (reversed column-major)

    Each direction gets its own SSM. Outputs are merged via learned projection.
    This is the key architectural difference from 1D Mamba and analogous to
    the cross-scan in VMamba, but simplified for the regression task.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.ssm_fwd = _get_ssm_block(d_model, d_state, d_conv, expand)
        self.ssm_bwd = _get_ssm_block(d_model, d_state, d_conv, expand)
        self.ssm_col_fwd = _get_ssm_block(d_model, d_state, d_conv, expand)
        self.ssm_col_bwd = _get_ssm_block(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(d_model * 4, d_model)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, H*W, D) with spatial dims H, W."""
        B, L, D = x.shape

        y_fwd = self.ssm_fwd(x)
        y_bwd = self.ssm_bwd(x.flip(1)).flip(1)

        x_col = rearrange(x.view(B, H, W, D), "b h w d -> b (w h) d")
        y_col_fwd = self.ssm_col_fwd(x_col)
        y_col_fwd = rearrange(y_col_fwd.view(B, W, H, D), "b w h d -> b (h w) d")

        x_col_rev = x_col.flip(1)
        y_col_bwd = self.ssm_col_bwd(x_col_rev).flip(1)
        y_col_bwd = rearrange(y_col_bwd.view(B, W, H, D), "b w h d -> b (h w) d")

        merged = torch.cat([y_fwd, y_bwd, y_col_fwd, y_col_bwd], dim=-1)
        return self.merge(merged)


class MambaBlock(nn.Module):
    """Single CosmoMamba block: norm → multi-directional SSM → residual → norm → FFN → residual."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.scan = MultiDirectionalScan(d_model, d_state, d_conv, expand)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        ffn_dim = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop1(self.scan(self.norm1(x), H, W))
        x = x + self.ffn(self.norm2(x))
        return x


class CosmoMamba(nn.Module):
    """
    CosmoMamba: Full 2D State-Space Model for cosmological parameter inference.

    Architecture:
      Conv patch embedding → stack of MambaBlocks → global average pool → regression head

    Outputs (mean, log_variance) for (Omega_m, sigma_8) to support
    heteroscedastic uncertainty estimation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_params: int = 2,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.H = img_size // patch_size
        self.W = img_size // patch_size
        n_patches = self.H * self.W

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=patch_size // 4, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, d_state, d_conv, expand, ffn_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean_out = nn.Linear(embed_dim, n_params)
        self.logvar_out = nn.Linear(embed_dim, n_params)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x = self.patch_embed(x)
        _, _, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        actual_H, actual_W = H, W
        if x.shape[1] != self.H * self.W:
            pos = rearrange(self.pos_embed, "1 (h w) c -> 1 c h w", h=self.H, w=self.W)
            pos = F.interpolate(pos, size=(actual_H, actual_W), mode="bilinear", align_corners=False)
            pos = rearrange(pos, "1 c h w -> 1 (h w) c")
        else:
            pos = self.pos_embed
            actual_H, actual_W = self.H, self.W

        x = self.pos_drop(x + pos)

        for block in self.blocks:
            x = block(x, actual_H, actual_W)

        x = self.norm(x)
        x = x.mean(dim=1)

        h = self.head(x)
        return self.mean_out(h), self.logvar_out(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        total = self.count_parameters()
        return (
            f"CosmoMamba: {total / 1e6:.1f}M params | "
            f"depth={len(self.blocks)} | embed_dim={self.norm.normalized_shape[0]} | "
            f"patches={self.H}x{self.W} | CUDA SSM={'yes' if HAS_CUDA_MAMBA else 'no'}"
        )
