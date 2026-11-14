from pathlib import Path
import argparse
import json
import sys

import numpy as np
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


def angular_loss(pred, target):
    pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)

    cos = torch.sum(pred * target, dim=-1)
    cos = torch.clamp(cos, -1.0, 1.0)

    return torch.mean(1.0 - cos)


def evaluate(model, loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
            y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)

            cos = torch.sum(pred * y, dim=-1)
            cos = torch.clamp(cos, -1.0, 1.0)

            err = torch.rad2deg(torch.acos(cos))
            errors.append(err.cpu().numpy())

    errors = np.concatenate(errors)

    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
    }


def load_audio_dims(data_dir: Path, args):
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        audio_coord_dim = metadata.get("audio_coord_dim", args.audio_dim)
        audio_validity_dim = metadata.get("audio_validity_dim", args.audio_validity_dim)
        return audio_coord_dim, audio_validity_dim

    if args.audio_validity_dim > 0:
        return args.audio_coord_dim, args.audio_validity_dim
    return args.audio_dim, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/hagi_audio_regression_fold0")
    parser.add_argument("--out", default="results/hagi_audio_regression_fold0")
    parser.add_argument("--seq-len", default=10, type=int)
    parser.add_argument("--audio-dim", default=9, type=int)
    parser.add_argument("--audio-coord-dim", default=None, type=int)
    parser.add_argument("--audio-validity-dim", default=0, type=int)
    parser.add_argument("--fusion", default="concat", choices=["concat", "gated"])
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_coord_dim, audio_validity_dim = load_audio_dims(data_dir, args)
    if args.audio_coord_dim is not None:
        audio_coord_dim = args.audio_coord_dim
    if args.audio_validity_dim > 0:
        audio_validity_dim = args.audio_validity_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(
        f"Fusion={args.fusion} | audio_coord_dim={audio_coord_dim} | "
        f"audio_validity_dim={audio_validity_dim}"
    )

    train_ds = NpyDataset(data_dir / "trainingX.npy", data_dir / "trainingY.npy")
    test_ds = NpyDataset(data_dir / "testX.npy", data_dir / "testY.npy")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_audio_model(
        fusion=args.fusion,
        seq_len=args.seq_len,
        audio_dim=args.audio_dim,
        audio_coord_dim=audio_coord_dim,
        audio_validity_dim=audio_validity_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mean = float("inf")

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = angular_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        metrics = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={np.mean(losses):.6f} | "
            f"mean={metrics['mean']:.3f}° | "
            f"median={metrics['median']:.3f}° | "
            f"std={metrics['std']:.3f}°"
        )

        if metrics["mean"] < best_mean:
            best_mean = metrics["mean"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            config = {
                "fusion": args.fusion,
                "audio_coord_dim": audio_coord_dim,
                "audio_validity_dim": audio_validity_dim,
                "seq_len": args.seq_len,
            }
            (out_dir / "model_config.json").write_text(json.dumps(config, indent=2))

    print("Best mean angular error:", best_mean)


if __name__ == "__main__":
    main()
