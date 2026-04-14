"""
Training script for CosmoMamba and baselines on CMD.

Usage:
  python train.py                              # use default config
  python train.py --config configs/cnn.yaml    # use custom config
  python train.py --model mamba --suite SIMBA  # CLI overrides
"""

import argparse
import os
import time
import json
import yaml
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from data import get_data_loaders
from models import build_model
from utils.metrics import compute_metrics


def get_device(cfg):
    dev = cfg["experiment"].get("device", "auto")
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)


def gaussian_nll_loss(mean, logvar, target):
    """Heteroscedastic Gaussian NLL: learns per-sample uncertainty."""
    var = torch.exp(logvar).clamp(min=1e-6)
    return 0.5 * (logvar + (target - mean) ** 2 / var).mean()


def train_one_epoch(model, loader, optimizer, device, loss_fn, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mean, logvar = model(x)

        if loss_fn == "gaussian_nll":
            loss = gaussian_nll_loss(mean, logvar, y)
        else:
            loss = nn.functional.mse_loss(mean, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mean, logvar = model(x)

        if loss_fn == "gaussian_nll":
            loss = gaussian_nll_loss(mean, logvar, y)
        else:
            loss = nn.functional.mse_loss(mean, y)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(mean.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train CosmoMamba on CMD")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--suite", type=str, default=None, help="Override simulation suite")
    parser.add_argument("--fields", nargs="+", default=None, help="Override fields")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.suite:
        cfg["data"]["suite"] = args.suite
    if args.fields:
        cfg["data"]["fields"] = args.fields
        cfg["model"]["params"]["in_channels"] = len(args.fields)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    torch.manual_seed(cfg["experiment"]["seed"])
    np.random.seed(cfg["experiment"]["seed"])
    device = get_device(cfg)
    print(f"Device: {device}")

    exp_name = cfg["experiment"]["name"]
    out_dir = os.path.join("runs", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    wandb_run = None
    if cfg["experiment"].get("use_wandb"):
        import wandb
        wandb_run = wandb.init(project="cosmo-mamba", name=exp_name, config=cfg)

    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = build_model(cfg).to(device)
    if hasattr(model, "summary"):
        print(model.summary())
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {cfg['model']['name']} | {n_params / 1e6:.1f}M params")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    warmup = cfg["training"].get("warmup_epochs", 10)
    total_epochs = cfg["training"]["epochs"]
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])

    loss_fn = cfg["training"].get("loss", "gaussian_nll")
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    patience = cfg["training"].get("patience", 50)
    save_every = cfg["training"].get("save_every", 50)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    start_epoch = 1

    resume_path = os.path.join(out_dir, "resume_ckpt.pt")
    best_path = os.path.join(out_dir, "best_model.pt")

    if os.path.exists(best_path) and not os.path.exists(resume_path):
        print(f"\nAlready trained — loading {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, test_loader, device, loss_fn)
        for k, v in sorted(test_metrics.items()):
            print(f"  {k}: {v:.6f}")
        return

    if os.path.exists(resume_path):
        print(f"\nResuming from {resume_path}...")
        r = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(r["model_state"])
        optimizer.load_state_dict(r["optim_state"])
        scheduler.load_state_dict(r["sched_state"])
        start_epoch = r["epoch"] + 1
        best_val_loss = r["best_val_loss"]
        best_state = r["best_state"]
        no_improve = r["no_improve"]
        print(f"  Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.5f}")

    print(f"\nTraining {cfg['model']['name']} for epochs {start_epoch}-{total_epochs}...")
    print(f"  Loss: {loss_fn} | LR: {cfg['training']['lr']} | Batch: {cfg['training']['batch_size']}")
    print("-" * 80)

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn, grad_clip)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()

        elapsed = time.time() - t0
        val_loss = val_metrics["loss"]
        lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch <= 5:
            print(
                f"Epoch {epoch:4d}/{total_epochs} | "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"R2(Ωm): {val_metrics['Omega_m_r2']:.4f} | R2(σ8): {val_metrics['sigma_8_r2']:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.1f}s"
            )

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                **{f"val/{k}": v for k, v in val_metrics.items()},
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if save_every and epoch % save_every == 0:
            ckpt = os.path.join(out_dir, f"checkpoint_ep{epoch}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg}, ckpt)

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss, "best_state": best_state,
                "no_improve": no_improve,
            }, resume_path)

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    torch.save({"model": best_state, "config": cfg}, best_path)
    if os.path.exists(resume_path):
        os.remove(resume_path)

    print("\n" + "=" * 80)
    print("Final evaluation on test set (best model):")
    test_metrics = evaluate(model, test_loader, device, loss_fn)
    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.6f}")

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    if wandb_run:
        wandb_run.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
