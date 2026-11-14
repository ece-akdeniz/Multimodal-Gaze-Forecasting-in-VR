import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append("external/DGaze/DGaze/models")

from DGazeModels import DGaze


class NpyDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = torch.from_numpy(np.load(x_path)).float()
        self.y = torch.from_numpy(np.load(y_path)).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    dataset_dir = Path("data/processed/dgaze_full_fold0_nearest")
    results_dir = Path("results/dgaze_full_fold0_nearest")
    results_dir.mkdir(parents=True, exist_ok=True)

    seq_len = 10
    seq_feature_num = 2
    output_dim = 2
    batch_size = 64
    epochs = 5
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = NpyDataset(
        dataset_dir / "trainingX.npy",
        dataset_dir / "trainingY.npy",
    )
    test_dataset = NpyDataset(
        dataset_dir / "testX.npy",
        dataset_dir / "testY.npy",
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DGaze(
        seqLength=seq_len,
        seqFeatureNum=seq_feature_num,
        saliencyWidth=24,
        saliencyNum=1,
        n_output=output_dim,
        dropoutRate=0.3,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        test_losses = []
        preds = []
        targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)

                test_losses.append(loss.item())
                preds.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {np.mean(train_losses):.6f} | "
            f"Test loss: {np.mean(test_losses):.6f}"
        )

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    np.save(results_dir / "predictions.npy", preds)
    np.save(results_dir / "targets.npy", targets)

    torch.save(model.state_dict(), results_dir / "model.pt")

    print("Saved results to:", results_dir)


if __name__ == "__main__":
    main()