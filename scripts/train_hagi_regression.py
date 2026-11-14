from pathlib import Path
import argparse
import numpy as np
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

        # normalize output to unit vector
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y


def angular_loss(pred, target):
    pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)

    cos = torch.sum(pred * target, dim=-1)
    cos = torch.clamp(cos, -1.0, 1.0)

    # 1 - cos(theta)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/hagi_regression_fold0")
    parser.add_argument("--out", default="results/hagi_regression_fold0")
    parser.add_argument("--seq-len", default=10, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = NpyDataset(data_dir / "trainingX.npy", data_dir / "trainingY.npy")
    test_ds = NpyDataset(data_dir / "testX.npy", data_dir / "testY.npy")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HAGIRegression(seq_len=args.seq_len).to(device)
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
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={np.mean(losses):.6f} | "
            f"mean={metrics['mean']:.3f}° | "
            f"median={metrics['median']:.3f}° | "
            f"std={metrics['std']:.3f}°"
        )

        if metrics["mean"] < best_mean:
            best_mean = metrics["mean"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    print("Best mean angular error:", best_mean)


if __name__ == "__main__":
    main()