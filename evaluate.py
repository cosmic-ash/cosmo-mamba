"""
Evaluation and visualization script.

Usage:
  python evaluate.py --checkpoint runs/cosmo_mamba_Mcdm_TNG/best_model.pt
  python evaluate.py --checkpoint runs/cosmo_mamba_Mcdm_TNG/best_model.pt --cross_suite SIMBA
"""

import argparse
import json
import os

import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import get_data_loaders
from models import build_model
from utils.metrics import compute_metrics


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict(model, loader, device):
    all_preds, all_logvars, all_targets = [], [], []
    for x, y in tqdm(loader, desc="Predicting"):
        x = x.to(device)
        mean, logvar = model(x)
        all_preds.append(mean.cpu().numpy())
        all_logvars.append(logvar.cpu().numpy())
        all_targets.append(y.numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_logvars),
        np.concatenate(all_targets),
    )


def plot_predictions(preds, targets, uncertainties, param_names, save_path):
    fig, axes = plt.subplots(1, len(param_names), figsize=(6 * len(param_names), 5))
    if len(param_names) == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        p, t = preds[:, i], targets[:, i]
        u = uncertainties[:, i]

        ax.errorbar(t, p, yerr=u, fmt=".", alpha=0.3, markersize=2, elinewidth=0.5, color="steelblue")

        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        r2 = 1 - np.sum((p - t) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-8)
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ax.set_title(f"{name}\nR² = {r2:.4f} | RMSE = {rmse:.4f}")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cross_suite", type=str, default=None,
                        help="Test on a different suite for robustness analysis")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)

    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    out_dir = os.path.dirname(args.checkpoint)

    if args.cross_suite:
        cfg_cross = json.loads(json.dumps(cfg))
        cfg_cross["data"]["suite"] = args.cross_suite
        _, _, test_loader = get_data_loaders(cfg_cross)
        suffix = f"_cross_{args.cross_suite}"
        print(f"\nCross-suite evaluation: trained on {cfg['data']['suite']}, testing on {args.cross_suite}")
    else:
        _, _, test_loader = get_data_loaders(cfg)
        suffix = ""
        print(f"\nEvaluating on {cfg['data']['suite']} test set")

    preds, logvars, targets = predict(model, test_loader, device)
    uncertainties = np.sqrt(np.exp(logvars))

    metrics = compute_metrics(preds, targets)
    print("\nResults:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.6f}")

    metrics_path = os.path.join(out_dir, f"eval_metrics{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    plot_path = os.path.join(out_dir, f"predictions{suffix}.png")
    plot_predictions(preds, targets, uncertainties, ["Omega_m", "sigma_8"], plot_path)

    np.savez(
        os.path.join(out_dir, f"predictions{suffix}.npz"),
        preds=preds, logvars=logvars, targets=targets,
    )


if __name__ == "__main__":
    main()
