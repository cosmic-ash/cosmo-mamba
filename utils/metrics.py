"""Evaluation metrics for cosmological parameter inference."""

import torch
import torch.nn as nn
import numpy as np


class R2Loss(nn.Module):
    """
    1 - R^2 as a loss function. Encourages the model to explain variance
    in the targets rather than just minimizing MSE around the mean.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ss_res = torch.sum((target - pred) ** 2, dim=0)
        ss_tot = torch.sum((target - target.mean(dim=0, keepdim=True)) ** 2, dim=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        return (1.0 - r2).mean()


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    param_names: tuple[str, ...] = ("Omega_m", "sigma_8"),
) -> dict:
    """
    Compute per-parameter and aggregate regression metrics.

    Returns dict with keys like 'Omega_m_rmse', 'sigma_8_r2', 'mean_rmse', etc.
    """
    results = {}
    for i, name in enumerate(param_names):
        pred = predictions[:, i]
        true = targets[:, i]
        residual = pred - true

        rmse = np.sqrt(np.mean(residual**2))
        mae = np.mean(np.abs(residual))
        bias = np.mean(residual)

        ss_res = np.sum(residual**2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        rel_err = np.mean(np.abs(residual) / (np.abs(true) + 1e-8))

        results[f"{name}_rmse"] = rmse
        results[f"{name}_mae"] = mae
        results[f"{name}_bias"] = bias
        results[f"{name}_r2"] = r2
        results[f"{name}_rel_err"] = rel_err

    results["mean_rmse"] = np.mean([results[f"{n}_rmse"] for n in param_names])
    results["mean_r2"] = np.mean([results[f"{n}_r2"] for n in param_names])

    return results
