from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    try:
        torch.zeros(1).cuda()
        print(f"  {torch.cuda.get_device_name(0)}")
    except RuntimeError:
        print("  GPU not compatible, falling back to CPU")
        device = torch.device("cpu")

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

DATA       = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "player_splitter_mlp.pt"

def load(prefix):
    print(f"Loading {prefix} set...", end=" ", flush=True)
    X = torch.tensor(np.load(DATA / f"X_{prefix}.npy"), dtype=torch.float32)
    Y = torch.tensor(np.load(DATA / f"Y_{prefix}.npy"), dtype=torch.float32)
    print(f"{len(X):,} examples")
    return TensorDataset(X, Y)

train_loader = DataLoader(load("train"), batch_size=256, shuffle=True)
val_loader   = DataLoader(load("val"),   batch_size=256)
test_loader  = DataLoader(load("test"),  batch_size=256)

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

model = nn.Sequential(
    nn.Linear(108, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 55),   # 0-53 melds, 54 = pass
).to(device)

print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters())
loss_fn   = nn.CrossEntropyLoss()

def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()

def run_epoch(loader, train=True, desc=""):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.set_grad_enabled(train):
        for X, Y in bar:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss   = loss_fn(logits, Y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X)
            total_acc  += accuracy(logits, Y) * len(X)
            n += len(X)
            bar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{total_acc/n:.4f}")
    return total_loss / n, total_acc / n

best_val_acc     = 0.0
patience         = 5
patience_counter = 0

print()
for epoch in range(1, 51):
    train_loss, train_acc = run_epoch(train_loader, train=True,
                                      desc=f"Epoch {epoch:>2} train")
    val_loss,   val_acc   = run_epoch(val_loader,   train=False,
                                      desc=f"Epoch {epoch:>2} val  ")

    improved = val_acc > best_val_acc
    print(f"Epoch {epoch:>2}  "
          f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
          f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
          + (" *" if improved else ""))

    if improved:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
_, test_acc = run_epoch(test_loader, train=False, desc="Test")
print(f"\nTest accuracy: {test_acc:.4f}")
