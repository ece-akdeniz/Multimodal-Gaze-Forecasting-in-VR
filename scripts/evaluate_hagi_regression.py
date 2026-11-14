from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class NpyDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = torch.from_numpy(np.load(x_path)).float()
        self.y = torch.from_numpy(np.load(y_path)).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class HAGIRegression(nn.Module):
    def __init__(self, seq_len=10, input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        x = x.view(x.shape[0], self.seq_len, self.input_dim)
        _, h = self.gru(x)
        h_last = h[-1]

        y = self.head(h_last)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)

        return y


def angular_error_deg(pred, target):
    pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)

    cos = torch.sum(pred * target, dim=-1)
    cos = torch.clamp(cos, -1.0, 1.0)

    return torch.rad2deg(torch.acos(cos))


def make_row(method_name, errors):
    return {
        "method": method_name,
        "samples": len(errors),
        "mean_angular_error_deg": float(errors.mean()) if len(errors) > 0 else np.nan,
        "median_angular_error_deg": float(np.median(errors)) if len(errors) > 0 else np.nan,
        "std_angular_error_deg": float(errors.std()) if len(errors) > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/processed/hagi_regression_fold0")
    parser.add_argument("--checkpoint", default="results/hagi_regression_fold0/best_model.pt")
    parser.add_argument("--out", default="results/hagi_regression_fold0")
    parser.add_argument("--seq-len", default=10, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--method-name", default=None, type=str)

    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_name = args.method_name or out_dir.name.replace("_", " ").title()

    summary_filename = f"{out_dir.name}_summary.csv"
    summary_path = out_dir / summary_filename

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_ds = NpyDataset(
        data_dir / "testX.npy",
        data_dir / "testY.npy",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = HAGIRegression(seq_len=args.seq_len).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    all_errors = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            errors = angular_error_deg(pred, y)

            all_errors.append(errors.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    errors = np.concatenate(all_errors)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    rows = []

    rows.append(make_row(f"{method_name} - All", errors))

    fixation_path = data_dir / "testFixation.npy"

    if fixation_path.exists():
        fixation = np.load(fixation_path).astype(bool)

        if len(fixation) != len(errors):
            raise ValueError(
                f"Length mismatch: testFixation has {len(fixation)} rows, "
                f"but errors has {len(errors)} rows."
            )

        fixation_errors = errors[fixation]
        non_fixation_errors = errors[~fixation]

        rows.append(make_row(f"{method_name} - Fixations Only", fixation_errors))
        rows.append(make_row(f"{method_name} - Saccades / Non-Fixations", non_fixation_errors))
    else:
        print("Warning: testFixation.npy not found. Only overall evaluation will be saved.")

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)

    np.save(out_dir / "predictions.npy", preds)
    np.save(out_dir / "targets.npy", targets)
    np.save(out_dir / "errors.npy", errors)

    print("\n=== HAGI REGRESSION EVALUATION ===")
    print(summary.to_string(index=False))
    print("\nSaved summary to:", summary_path)


if __name__ == "__main__":
    main()