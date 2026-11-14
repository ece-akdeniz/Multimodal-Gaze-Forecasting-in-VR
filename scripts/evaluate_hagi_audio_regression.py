from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hagi_audio_model import build_audio_model


class NpyDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = torch.from_numpy(np.load(x_path)).float()
        self.y = torch.from_numpy(np.load(y_path)).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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


def load_model_config(out_dir: Path, data_dir: Path, args):
    config_path = out_dir / "model_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())

    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        return {
            "fusion": args.fusion,
            "audio_coord_dim": metadata.get("audio_coord_dim", args.audio_dim),
            "audio_validity_dim": metadata.get("audio_validity_dim", args.audio_validity_dim),
            "seq_len": args.seq_len,
        }

    return {
        "fusion": args.fusion,
        "audio_coord_dim": args.audio_coord_dim or args.audio_dim,
        "audio_validity_dim": args.audio_validity_dim,
        "seq_len": args.seq_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/hagi_audio_regression_fold0")
    parser.add_argument("--checkpoint", default="results/hagi_audio_regression_fold0/best_model.pt")
    parser.add_argument("--out", default="results/hagi_audio_regression_fold0")
    parser.add_argument("--seq-len", default=10, type=int)
    parser.add_argument("--audio-dim", default=9, type=int)
    parser.add_argument("--audio-coord-dim", default=None, type=int)
    parser.add_argument("--audio-validity-dim", default=0, type=int)
    parser.add_argument("--fusion", default="concat", choices=["concat", "gated"])
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--method-name", default=None, type=str)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_name = args.method_name or out_dir.name.replace("_", " ").title()
    summary_path = out_dir / f"{out_dir.name}_summary.csv"

    config = load_model_config(out_dir, data_dir, args)
    fusion = config.get("fusion", args.fusion)
    audio_coord_dim = config.get("audio_coord_dim", args.audio_coord_dim or args.audio_dim)
    audio_validity_dim = config.get("audio_validity_dim", args.audio_validity_dim)
    seq_len = config.get("seq_len", args.seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(
        f"Fusion={fusion} | audio_coord_dim={audio_coord_dim} | "
        f"audio_validity_dim={audio_validity_dim}"
    )

    test_ds = NpyDataset(data_dir / "testX.npy", data_dir / "testY.npy")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_audio_model(
        fusion=fusion,
        seq_len=seq_len,
        audio_dim=args.audio_dim,
        audio_coord_dim=audio_coord_dim,
        audio_validity_dim=audio_validity_dim,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    all_errors = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            errors = angular_error_deg(pred, y)
            all_errors.append(errors.cpu().numpy())

    errors = np.concatenate(all_errors)

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

    print("\n=== HAGI + AUDIO REGRESSION EVALUATION ===")
    print(summary.to_string(index=False))
    print("\nSaved summary to:", summary_path)


if __name__ == "__main__":
    main()
