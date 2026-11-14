"""
train_ambisonic.py

Training script for HAGI++ Regression with ambisonic direction features.

Input X: [head_xyz_window (seq_len*3) | audio_direction_future (audio_seq_len*3)]
         e.g. (N, 120) for seq_len=30, audio_seq_len=10

Architecture:
    - GRU over head xyz sequence      → head embedding (hidden_dim)
    - GRU over audio direction sequence → audio embedding (hidden_dim // 2)
    - Concatenate → regression head → 3D gaze unit vector

Both head and audio are in the same xyz coordinate space —
the model can directly compare head direction to audio direction.
"""

from pathlib import Path
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

class HAGIAmbisonicRegression(nn.Module):
    """
    HAGI++ Regression with ambisonic direction features.

    Both head motion and audio direction are 3D unit vectors —
    same coordinate space, enabling direct comparison.

    Architecture:
        head_seq  (B, seq_len, 3)       → GRU → h_head  (B, hidden_dim)
        audio_seq (B, audio_seq_len, 3) → GRU → h_audio (B, hidden_dim//2)
        concat    (B, hidden_dim + hidden_dim//2)
        → regression head → gaze (B, 3)
    """

    def __init__(
        self,
        seq_len       = 30,
        head_dim      = 3,
        audio_dim     = 3,
        audio_seq_len = 10,
        hidden_dim    = 128,
        num_layers    = 2,
        dropout       = 0.3,
    ):
        super().__init__()

        self.seq_len       = seq_len
        self.head_dim      = head_dim
        self.audio_dim     = audio_dim
        self.audio_seq_len = audio_seq_len

        # head motion encoder
        self.head_gru = nn.GRU(
            input_size  = head_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # audio direction encoder (smaller — audio is lower info)
        self.audio_gru = nn.GRU(
            input_size  = audio_dim,
            hidden_size = hidden_dim // 2,
            num_layers  = 1,
            batch_first = True,
        )

        # fusion + regression head
        fused_dim = hidden_dim + hidden_dim // 2  # e.g. 128 + 64 = 192

        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        """
        Args:
            x: (B, seq_len*3 + audio_seq_len*3)  flat input vector

        Returns:
            gaze: (B, 3) normalized unit vector
        """
        head_size  = self.seq_len * self.head_dim          # e.g. 90
        audio_size = self.audio_seq_len * self.audio_dim   # e.g. 30

        head_flat  = x[:, :head_size]                      # (B, 90)
        audio_flat = x[:, head_size : head_size + audio_size]  # (B, 30)

        # reshape for GRUs
        head_seq  = head_flat.view(x.shape[0], self.seq_len, self.head_dim)              # (B, 30, 3)
        audio_seq = audio_flat.view(x.shape[0], self.audio_seq_len, self.audio_dim)     # (B, 10, 3)

        # encode
        _, h_head  = self.head_gru(head_seq)
        head_emb   = h_head[-1]                            # (B, hidden_dim)

        _, h_audio = self.audio_gru(audio_seq)
        audio_emb  = h_audio[-1]                           # (B, hidden_dim//2)

        # fuse and predict
        fused    = torch.cat([head_emb, audio_emb], dim=-1)  # (B, 192)
        gaze_raw = self.regression_head(fused)               # (B, 3)

        # normalize to unit vector
        gaze = gaze_raw / (torch.norm(gaze_raw, dim=-1, keepdim=True) + 1e-8)
        return gaze


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def angular_loss(pred, target):
    pred   = pred   / (torch.norm(pred,   dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    cos    = torch.clamp(torch.sum(pred * target, dim=-1), -1.0, 1.0)
    return torch.mean(1.0 - cos)


def mean_angular_error_deg(pred, target):
    pred   = pred   / (torch.norm(pred,   dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    cos    = torch.clamp(torch.sum(pred * target, dim=-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            err  = mean_angular_error_deg(pred, y)
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
    parser = argparse.ArgumentParser(
        description="Train HAGI++ Regression with ambisonic direction features."
    )
    parser.add_argument("--data",          default="data/processed/hagi_ambisonic_fold0")
    parser.add_argument("--out",           default="results/hagi_ambisonic_fold0")
    parser.add_argument("--seq-len",       default=30,   type=int)
    parser.add_argument("--audio-seq-len", default=10,   type=int)
    parser.add_argument("--hidden-dim",    default=128,  type=int)
    parser.add_argument("--num-layers",    default=2,    type=int)
    parser.add_argument("--dropout",       default=0.3,  type=float)
    parser.add_argument("--epochs",        default=20,   type=int)
    parser.add_argument("--batch-size",    default=512,  type=int)
    parser.add_argument("--lr",            default=1e-3, type=float)
    parser.add_argument("--num-workers",   default=2,    type=int)
    parser.add_argument("--patience",      default=3,    type=int)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # data
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

    # model
    model = HAGIAmbisonicRegression(
        seq_len       = args.seq_len,
        head_dim      = 3,
        audio_dim     = 3,
        audio_seq_len = args.audio_seq_len,
        hidden_dim    = args.hidden_dim,
        num_layers    = args.num_layers,
        dropout       = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mean        = float("inf")
    best_epoch       = -1
    epochs_no_improve = 0

    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...\n")

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

        metrics    = evaluate(model, test_loader, device)
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
            best_mean         = metrics["mean"]
            best_epoch        = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nBest MAE: {best_mean:.3f}° at epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
