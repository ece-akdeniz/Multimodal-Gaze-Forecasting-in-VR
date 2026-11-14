"""
train_aem.py

Training script for HAGI++ Regression with AEM spatial audio features.

Model: GRU encoder over head xyz sequence + MLP encoder over AEM spatial map
       → fused regression head → predicted gaze direction (3D unit vector)

Loss: 1 - cos(theta)  (angular loss)
Metric: Mean Angular Error in degrees

Usage:
    # Full fold 0
    python train_aem.py --data data/processed/hagi_aem_regression_fold0

    # Tiny debug run
    python train_aem.py --data data/processed/hagi_aem_regression_tiny --epochs 5

    # Ablation: current frame instead of next frame
    python train_aem.py --data data/processed/hagi_aem_regression_fold0_current
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NpyDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = torch.from_numpy(np.load(x_path)).float()
        self.y = torch.from_numpy(np.load(y_path)).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HAGIAEMRegression(nn.Module):
    """
    HAGI++ Regression with AEM spatial audio features.

    Input X is a flat vector:
        [head_xyz_window (seq_len*3) | aem_feature (aem_dim)]

    Architecture:
        - GRU encodes the head motion sequence → hidden_dim
        - MLP encodes the AEM spatial feature → 64
        - Concatenate → regression head → 3D gaze unit vector

    Args:
        seq_len:    Number of past head frames (default 10)
        head_dim:   Dimension per head frame (3 for xyz)
        aem_dim:    AEM feature dimension (default 128 = 8×16 grid)
        hidden_dim: GRU hidden size (default 128)
        num_layers: GRU layers (default 2)
        dropout:    Dropout rate (default 0.2)
    """

    def __init__(
        self,
        seq_len    = 10,
        head_dim   = 3,
        aem_dim    = 128,
        hidden_dim = 128,
        num_layers = 2,
        dropout    = 0.2,
    ):
        super().__init__()

        self.seq_len  = seq_len
        self.head_dim = head_dim
        self.aem_dim  = aem_dim

        # --- head motion encoder ---
        self.gru = nn.GRU(
            input_size  = head_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # --- AEM spatial audio encoder ---
        # 128-dim spatial map → 64-dim embedding
        self.aem_encoder = nn.Sequential(
            nn.Linear(aem_dim, 64),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.3), 
        )

        # --- fusion + regression head ---
        # GRU output (hidden_dim) + AEM embedding (64) → 3D gaze
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len*head_dim + aem_dim)

        Returns:
            gaze: (batch, 3) normalized unit vector
        """
        # split flat input back into head sequence and AEM feature
        head_flat   = x[:, : self.seq_len * self.head_dim]          # (B, seq_len*3)
        aem_feature = x[:, self.seq_len * self.head_dim :]           # (B, aem_dim)

        # reshape head for GRU
        head_seq = head_flat.view(x.shape[0], self.seq_len, self.head_dim)  # (B, T, 3)

        # encode head motion
        _, h          = self.gru(head_seq)
        head_emb      = h[-1]                    # (B, hidden_dim) — last GRU layer

        # encode AEM spatial feature
        aem_emb       = self.aem_encoder(aem_feature)  # (B, 64)

        # fuse and predict
        fused         = torch.cat([head_emb, aem_emb], dim=-1)  # (B, hidden_dim+64)
        gaze_raw      = self.regression_head(fused)              # (B, 3)

        # normalize to unit vector
        gaze = gaze_raw / (torch.norm(gaze_raw, dim=-1, keepdim=True) + 1e-8)
        return gaze


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def angular_loss(pred, target):
    """1 - cos(theta) — differentiable angular loss."""
    pred   = pred   / (torch.norm(pred,   dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    cos    = torch.clamp(torch.sum(pred * target, dim=-1), -1.0, 1.0)
    return torch.mean(1.0 - cos)


def mean_angular_error_deg(pred, target):
    """Mean angular error in degrees."""
    pred   = pred   / (torch.norm(pred,   dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    cos    = torch.clamp(torch.sum(pred * target, dim=-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            pred  = model(x)
            err   = mean_angular_error_deg(pred, y)
            errors.append(err.cpu().numpy())

    errors = np.concatenate(errors)
    return {
        "mean":   float(errors.mean()),
        "median": float(np.median(errors)),
        "std":    float(errors.std()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train HAGI++ Regression with AEM audio features.")
    parser.add_argument("--data",       default="data/processed/hagi_aem_regression_fold0")
    parser.add_argument("--out",        default="results/hagi_aem_regression_fold0")
    parser.add_argument("--seq-len",    default=10,    type=int)
    parser.add_argument("--aem-dim",    default=128,   type=int,
                        help="AEM feature dimension (grid_rows * grid_cols, default 128)")
    parser.add_argument("--hidden-dim", default=128,   type=int)
    parser.add_argument("--num-layers", default=2,     type=int)
    parser.add_argument("--dropout",    default=0.2,   type=float)
    parser.add_argument("--epochs",     default=20,    type=int)
    parser.add_argument("--batch-size", default=512,   type=int)
    parser.add_argument("--lr",         default=1e-3,  type=float)
    parser.add_argument("--num-workers",default=2,     type=int)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    train_ds = NpyDataset(data_dir / "trainingX.npy", data_dir / "trainingY.npy")
    test_ds  = NpyDataset(data_dir / "testX.npy",     data_dir / "testY.npy")

    print(f"Train samples: {len(train_ds):,}")
    print(f"Test samples:  {len(test_ds):,}")
    print(f"Feature dim:   {train_ds.x.shape[1]}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # --- model ---
    model = HAGIAEMRegression(
        seq_len    = args.seq_len,
        head_dim   = 3,
        aem_dim    = args.aem_dim,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        dropout    = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- training loop ---
    best_mean = float("inf")
    best_epoch = -1
    patience = 3
    epochs_no_improve = 0

    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = angular_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        metrics = evaluate(model, test_loader, device)

        epoch_end = time.time()
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1:>3}/{args.epochs} | "
            f"train_loss={np.mean(losses):.6f} | "
            f"MAE={metrics['mean']:.3f}° | "
            f"median={metrics['median']:.3f}° | "
            f"std={metrics['std']:.3f}° | "
            f"time={epoch_time:.1f}s"
        )
        
        if metrics["mean"] < best_mean:
            best_mean = metrics["mean"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest MAE: {best_mean:.3f}° at epoch {best_epoch}")
    print(f"Checkpoint saved to: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
