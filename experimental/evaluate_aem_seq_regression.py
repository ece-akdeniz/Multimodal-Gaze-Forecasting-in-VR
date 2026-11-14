"""
evaluate_aem.py

Evaluation script for HAGI++ Regression with AEM spatial audio features.

Loads best_model.pt and reports:
    - Overall mean angular error (MAE)
    - MAE on fixations only
    - MAE on saccades only
    - Median and std for each

Saves results to a CSV summary file.

Usage:
    python evaluate_aem.py \
        --data data/processed/hagi_aem_regression_fold0 \
        --checkpoint results/hagi_aem_regression_fold0/best_model.pt \
        --method-name "HAGI++ AEM (t+1)"
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset (same as training)
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
# Model (must match training exactly)
# ---------------------------------------------------------------------------

import torch.nn as nn

class HAGIAEMRegression(nn.Module):
    def __init__(
        self,
        seq_len         = 30, # head seq len
        head_dim        = 3,
        aem_dim         = 128,
        audio_seq_len   = 10, # audio seq len
        hidden_dim      = 128,
        num_layers      = 2,
        dropout         = 0.2,
    ):
        super().__init__()

        self.seq_len         = seq_len
        self.head_dim        = head_dim
        self.aem_dim         = aem_dim
        self.audio_seq_len   = audio_seq_len

        self.gru = nn.GRU(
            input_size  = head_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # --- AEM spatial audio encoder ---
        self.aem_gru = nn.GRU(
            input_size = aem_dim,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),  # 192 -> 128
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
        head_flat   = x[:, : self.seq_len * self.head_dim]          # (B, 90)
        aem_flat = x[:, self.seq_len * self.head_dim :]           # (B, 1280)

        head_seq = head_flat.view(x.shape[0], self.seq_len, self.head_dim)  # (B, 30, 3)
        aem_seq = aem_flat.view(x.shape[0], self.audio_seq_len, self.aem_dim)  # (B, 10, 128)

        _, h          = self.gru(head_seq)
        head_emb      = h[-1]                    # (B, 128) — last GRU layer

        _, h_aem = self.aem_gru(aem_seq)
        aem_emb  = h_aem[-1]  # (B, 64)

        fused         = torch.cat([head_emb, aem_emb], dim=-1)  # (B, hidden_dim+64)
        gaze_raw      = self.regression_head(fused)              # (B, 3)

        gaze = gaze_raw / (torch.norm(gaze_raw, dim=-1, keepdim=True) + 1e-8)
        return gaze


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def angular_error_deg(pred, target):
    """Per-sample angular error in degrees."""
    pred   = pred   / (torch.norm(pred,   dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    cos    = torch.clamp(torch.sum(pred * target, dim=-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def make_row(label, errors):
    """Build one summary row from an array of per-sample errors."""
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
    parser = argparse.ArgumentParser(description="Evaluate HAGI++ AEM Regression model.")
    parser.add_argument("--data",         default="data/processed/hagi_aem_regression_fold0")
    parser.add_argument("--checkpoint",   default="results/hagi_aem_regression_fold0/best_model.pt")
    parser.add_argument("--out",          default="results/hagi_aem_regression_fold0")
    parser.add_argument("--seq-len",      default=30,  type=int)
    parser.add_argument("--aem-dim",      default=128, type=int)
    parser.add_argument("--audio-seq-len",    default=10,    type=int)
    parser.add_argument("--hidden-dim",   default=128, type=int)
    parser.add_argument("--num-layers",   default=2,   type=int)
    parser.add_argument("--dropout",      default=0.3, type=float)
    parser.add_argument("--batch-size",   default=512, type=int)
    parser.add_argument("--num-workers",  default=2,   type=int)
    parser.add_argument("--method-name",  default=None, type=str,
                        help="Label for this run in the summary CSV (e.g. 'HAGI++ AEM t+1')")
    args = parser.parse_args()

    data_dir   = Path(args.data)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_name  = args.method_name or out_dir.name.replace("_", " ").title()
    summary_path = out_dir / f"{out_dir.name}_summary.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Method: {method_name}")

    # --- data ---
    test_ds = NpyDataset(data_dir / "testX.npy", data_dir / "testY.npy")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Test samples: {len(test_ds):,}")

    # --- load model ---
    model = HAGIAEMRegression(
        seq_len    = args.seq_len,
        head_dim   = 3,
        aem_dim    = args.aem_dim,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        dropout    = args.dropout,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- run inference ---
    all_errors = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y   = x.to(device), y.to(device)
            pred   = model(x)
            errors = angular_error_deg(pred, y)
            all_errors.append(errors.cpu().numpy())

    all_errors = np.concatenate(all_errors)  # (N,)

    # --- build summary rows ---
    rows = []
    rows.append(make_row(f"{method_name} — All", all_errors))

    # fixation breakdown
    fixation_path = data_dir / "testFixation.npy"
    if fixation_path.exists():
        fixation = np.load(fixation_path).astype(bool)

        if len(fixation) != len(all_errors):
            print(
                f"Warning: fixation length {len(fixation)} != errors length {len(all_errors)}. "
                "Skipping fixation breakdown."
            )
        else:
            fix_errors = all_errors[fixation]
            sac_errors = all_errors[~fixation]
            rows.append(make_row(f"{method_name} — Fixations Only", fix_errors))
            rows.append(make_row(f"{method_name} — Saccades Only",  sac_errors))
            print(f"  Fixation samples: {fix_errors.shape[0]:,}")
            print(f"  Saccade samples:  {sac_errors.shape[0]:,}")
    else:
        print("Warning: testFixation.npy not found — skipping fixation breakdown.")

    # --- print and save ---
    summary = pd.DataFrame(rows)

    print("\n=== HAGI++ AEM REGRESSION — EVALUATION RESULTS ===")
    print(summary.to_string(index=False))

    summary.to_csv(summary_path, index=False)
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
