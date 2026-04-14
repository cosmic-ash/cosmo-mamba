"""
CNN baseline for cosmological parameter inference.
Reproduces the CMD benchmark architecture (o3_err variant):
  Conv2d blocks with increasing channels → adaptive avg pool → FC head
  predicting mean and log-variance for (Omega_m, sigma_8).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CNNBaseline(nn.Module):
    """
    CMD-style CNN: stacked conv blocks with periodic padding awareness,
    pooling layers, and a regression head that outputs mean + log-variance
    for uncertainty estimation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_params: int = 2,
        hidden_channels: list[int] | None = None,
        fc_hidden: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [16, 32, 64, 128, 256, 256]

        layers = []
        ch_in = in_channels
        for i, ch_out in enumerate(hidden_channels):
            layers.append(ConvBlock(ch_in, ch_out, kernel_size=3, stride=1, padding=1))
            if i < len(hidden_channels) - 1:
                layers.append(nn.AvgPool2d(2))
            ch_in = ch_out

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(hidden_channels[-1], fc_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        self.mean_out = nn.Linear(fc_hidden, n_params)
        self.logvar_out = nn.Linear(fc_hidden, n_params)

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        h = self.head(h)
        mean = self.mean_out(h)
        logvar = self.logvar_out(h)
        return mean, logvar
