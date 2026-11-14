from pathlib import Path
import numpy as np
import pandas as pd


def uv_to_xyz(u, v):
    lon = (u - 0.5) * 2.0 * np.pi
    lat = (0.5 - v) * np.pi

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    return np.stack([x, y, z], axis=-1)


def angular_error_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    cos_theta = np.sum(a * b, axis=-1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.degrees(np.arccos(cos_theta))


def main():
    results_dir = Path("results/dgaze_head_only_fold0")

    preds = np.load(results_dir / "predictions.npy")
    targets = np.load(results_dir / "targets.npy")

    # keep predictions inside valid panorama coordinate range
    preds = np.clip(preds, 0.0, 1.0)

    pred_vec = uv_to_xyz(preds[:, 0], preds[:, 1])
    target_vec = uv_to_xyz(targets[:, 0], targets[:, 1])

    errors = angular_error_deg(pred_vec, target_vec)

    summary = pd.DataFrame([{
        "method": "DGaze Head-only",
        "samples": len(errors),
        "mean_angular_error_deg": errors.mean(),
        "median_angular_error_deg": np.median(errors),
        "std_angular_error_deg": errors.std(),
    }])

    summary.to_csv(results_dir / "dgaze_head_only_fold0_summary.csv", index=False)

    print("\n=== DGAZE HEAD-ONLY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()