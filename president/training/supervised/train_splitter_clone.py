"""
training/supervised/train_splitter_clone.py

Usage:
    python train_splitter_clone.py <experiment_name>

Loads config from:
    training/experiments/<experiment_name>/config.json

Reads data from:
    training/experiments/<experiment_name>/data/X_train.npy  etc.

Writes to:
    training/experiments/<experiment_name>/
        splitter_clone_v1.pt   – best checkpoint
        metrics.csv              – per-epoch loss/acc
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

SUPERVISED_DIR  = Path(__file__).parent
EXPERIMENTS_DIR = SUPERVISED_DIR.parent / "experiments"


def experiment_dir(name: str) -> Path:
    d = EXPERIMENTS_DIR / name
    if not d.is_dir():
        sys.exit(f"Experiment directory not found: {d}")
    return d


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "input_size":   108,
    "output_size":  55,
    "hidden_sizes": [256, 256],
    "activation":   "relu",
    "batch_size":   256,
    "lr":           1e-3,
    "max_epochs":   50,
    "patience":     5,
}


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found in {exp_dir}")
    with open(config_path) as f:
        cfg = json.load(f)
    # Fill in any keys not present in the file with defaults
    for key, value in DEFAULT_CONFIG.items():
        cfg.setdefault(key, value)
    return cfg


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

def resolve_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        try:
            torch.zeros(1).cuda()
            print(f"  {torch.cuda.get_device_name(0)}")
        except RuntimeError:
            print("  GPU not compatible, falling back to CPU")
            device = torch.device("cpu")
    return device


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def load_split(data_dir: Path, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    print(f"Loading {split} set...", end=" ", flush=True)
    X = torch.tensor(np.load(data_dir / f"X_{split}.npy"), dtype=torch.float32)
    Y = torch.tensor(np.load(data_dir / f"Y_{split}.npy"), dtype=torch.float32)
    print(f"{len(X):,} examples")
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

ACTIVATIONS = {
    "relu":    nn.ReLU,
    "tanh":    nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}


def build_model(cfg: dict) -> nn.Module:
    act_cls = ACTIVATIONS.get(cfg["activation"].lower())
    if act_cls is None:
        sys.exit(f"Unknown activation '{cfg['activation']}'. Choose from: {list(ACTIVATIONS)}")

    layers: list[nn.Module] = []
    in_size = cfg["input_size"]
    for hidden in cfg["hidden_sizes"]:
        layers += [nn.Linear(in_size, hidden), act_cls()]
        in_size = hidden
    layers.append(nn.Linear(in_size, cfg["output_size"]))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()


def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    loss_fn:   nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    desc:      str = "",
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_acc, n = 0.0, 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.set_grad_enabled(training):
        for X, Y in bar:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss   = loss_fn(logits, Y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X)
            total_acc  += accuracy(logits, Y) * len(X)
            n += len(X)
            bar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{total_acc/n:.4f}")
    return total_loss / n, total_acc / n


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clone expert system into an MLP.")
    parser.add_argument("experiment", help="Name of the experiment directory under training/experiments/")
    args = parser.parse_args()

    exp_dir  = experiment_dir(args.experiment)
    cfg      = load_config(exp_dir)
    data_dir = exp_dir / "data"

    print(f"\nExperiment : {args.experiment}")
    print(f"Config     : {json.dumps(cfg, indent=2)}\n")

    device = resolve_device()

    # Data
    train_loader = load_split(data_dir, "train", cfg["batch_size"], shuffle=True)
    val_loader   = load_split(data_dir, "val",   cfg["batch_size"], shuffle=False)
    test_loader  = load_split(data_dir, "test",  cfg["batch_size"], shuffle=False)

    # Model
    model = build_model(cfg).to(device)
    print(f"\nArchitecture:\n{model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn   = nn.CrossEntropyLoss()

    checkpoint_path = exp_dir / f"{args.experiment}.pt"
    metrics_path    = exp_dir / "metrics.csv"

    best_val_acc     = 0.0
    patience_counter = 0

    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, cfg["max_epochs"] + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, loss_fn, optimizer, device,
                desc=f"Epoch {epoch:>2} train",
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, loss_fn, None, device,
                desc=f"Epoch {epoch:>2} val  ",
            )

            improved = val_acc > best_val_acc
            print(
                f"Epoch {epoch:>2}  "
                f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
                + (" *" if improved else "")
            )

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            csvfile.flush()

            if improved:
                best_val_acc     = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                # Keep a copy of the config that produced this checkpoint
                shutil.copy(exp_dir / "config.json", exp_dir / "config_checkpoint.json")
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    # ── Test ──────────────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    _, test_acc = run_epoch(model, test_loader, loss_fn, None, device, desc="Test")
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Append test result to metrics
    with open(metrics_path, "a", newline="") as csvfile:
        csv.writer(csvfile).writerow(["test", "", "", "", test_acc])


if __name__ == "__main__":
    main()
