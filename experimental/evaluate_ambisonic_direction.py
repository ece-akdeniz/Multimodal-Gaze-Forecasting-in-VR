"""
evaluate_ambisonic.py

Evaluation script for HAGI++ Regression with ambisonic direction features.

Usage:
    python evaluate_ambisonic.py \
        --data data/processed/hagi_ambisonic_fold0 \
        --checkpoint results/hagi_ambisonic_fold0/best_model.pt \
        --method-name "HAGI++ Ambisonic Direction"
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
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
# Model — must match train_ambisonic.py exactly
# ---------------------------------------------------------------------------

class HAGIAmbisonicRegression(nn.Module):
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

        self.head_gru = nn.GRU(
            input_size  = head_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.audio_gru = nn.GRU(
            input_size  = audio_dim,
            hidden_size = hidden_dim // 2,
            num_layers  = 1,
            batch_first = True,
        )

        fused_dim = hidden_dim + hidden_dim // 2

        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        head_size  = self.seq_len * self.head_dim
        audio_size = self.audio_seq_len * self.audio_dim

        head_flat  = x[:, :head_size]
        audio_flat = x[:, head_size : head_size + audio_size]

        head_seq  = head_flat.view(x.shape[0], self.seq_len, self.head_dim)
        audio_seq = audio_flat.view(x.shape[0], self.audio_seq_len, self.audio_dim)

        _, h_head  = self.head_gru(head_seq)
        head_emb   = h_head[-1]

        _, h_audio = self.audio_gru(audio_seq)
        audio_emb  = h_audio[-1]

        fused    = torch.cat([head_emb, audio_emb], dim=-1)
        gaze_raw = self.regression_head(fused)
        return gaze_raw / (torch.norm(gaze_raw, dim=-1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def angular_error_deg(pred, target):
    pred   = pred   / (torch.norm(pred,   dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    cos    = torch.clamp(torch.sum(pred * target, dim=-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def make_row(label, errors):
    if len(errors) == 0:
        return {
            "method":                   label,
            "samples":                  0,
            "mean_angular_error_deg":   float("nan"),
            "median_angular_error_deg": float("nan"),
            "std_angular_error_deg":    float("nan"),
        }
    return {
        "method":                   label,
        "samples":                  len(errors),
        "mean_angular_error_deg":   float(errors.mean()),
        "median_angular_error_deg": float(np.median(errors)),
        "std_angular_error_deg":    float(errors.std()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HAGI++ Ambisonic Regression model."
    )
    parser.add_argument("--data",          default="data/processed/hagi_ambisonic_fold0")
    parser.add_argument("--checkpoint",    default="results/hagi_ambisonic_fold0/best_model.pt")
    parser.add_argument("--out",           default="results/hagi_ambisonic_fold0")
    parser.add_argument("--seq-len",       default=30,  type=int)
    parser.add_argument("--audio-seq-len", default=10,  type=int)
    parser.add_argument("--hidden-dim",    default=128, type=int)
    parser.add_argument("--num-layers",    default=2,   type=int)
    parser.add_argument("--dropout",       default=0.3, type=float)
    parser.add_argument("--batch-size",    default=512, type=int)
    parser.add_argument("--num-workers",   default=2,   type=int)
    parser.add_argument("--method-name",   default=None, type=str)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_name  = args.method_name or out_dir.name.replace("_", " ").title()
    summary_path = out_dir / f"{out_dir.name}_summary.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Method:  {method_name}")

    # data
    test_ds = NpyDataset(data_dir / "testX.npy", data_dir / "testY.npy")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Test samples: {len(test_ds):,}")

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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # inference
    all_errors = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y   = x.to(device), y.to(device)
            pred   = model(x)
            errors = angular_error_deg(pred, y)
            all_errors.append(errors.cpu().numpy())

    all_errors = np.concatenate(all_errors)

    # summary rows
    rows = [make_row(f"{method_name} — All", all_errors)]

    fixation_path = data_dir / "testFixation.npy"
    if fixation_path.exists():
        fixation = np.load(fixation_path).astype(bool)
        if len(fixation) == len(all_errors):
            rows.append(make_row(f"{method_name} — Fixations Only", all_errors[fixation]))
            rows.append(make_row(f"{method_name} — Saccades Only",  all_errors[~fixation]))
        else:
            print(f"Warning: fixation length mismatch, skipping breakdown.")
    else:
        print("Warning: testFixation.npy not found.")

    summary = pd.DataFrame(rows)

    print("\n=== HAGI++ AMBISONIC DIRECTION — EVALUATION RESULTS ===")
    print(summary.to_string(index=False))

    summary.to_csv(summary_path, index=False)
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
