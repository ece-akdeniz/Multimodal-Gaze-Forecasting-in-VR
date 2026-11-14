from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/raw/D-SAV360")
GAZE_DIR = ROOT / "gaze"
HEAD_DIR = ROOT / "head"

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

def load_all_csvs(folder, cols):
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)

def main():
    print("Loading gaze...")
    gaze = load_all_csvs(GAZE_DIR, ["video", "stereo", "frame", "t", "u", "v", "id"])
    gaze = gaze.rename(columns={"u": "gaze_u", "v": "gaze_v", "t": "gaze_t"})
    gaze["gaze_t"] = gaze["gaze_t"] / 1000.0 # convert ms to s

    print("Loading head...")
    head = load_all_csvs(HEAD_DIR, ["video", "stereo", "t", "u", "v", "id"])
    head = head.rename(columns={"u": "head_u", "v": "head_v", "t": "head_t"})

    print("Sorting...")
    gaze = gaze.sort_values(["video", "id", "gaze_t"])
    head = head.sort_values(["video", "id", "head_t"])

    print("Merging nearest timestamps...")
    merged_parts = []

    for (video, pid), g in gaze.groupby(["video", "id"]):
        h = head[(head["video"] == video) & (head["id"] == pid)]

        if h.empty:
            continue

        g = g.sort_values("gaze_t")
        h = h.sort_values("head_t")

        m = pd.merge_asof(
            g,
            h,
            left_on="gaze_t",
            right_on="head_t",
            by=["video", "id"],
            direction="nearest",
            tolerance=0.05,  # seconds; adjust if needed
        )

        m = m.dropna(subset=["head_u", "head_v"])
        if len(m) > 0:
            merged_parts.append(m)

    if not merged_parts:
        raise RuntimeError("No gaze/head samples matched. Check timestamps, ids, or tolerance.")

    merged = pd.concat(merged_parts, ignore_index=True)

    print("Matched samples:", len(merged))

    gaze_vec = uv_to_xyz(merged["gaze_u"].to_numpy(), merged["gaze_v"].to_numpy())
    head_vec = uv_to_xyz(merged["head_u"].to_numpy(), merged["head_v"].to_numpy())

    errors = angular_error_deg(head_vec, gaze_vec)

    print("\n=== HEAD PROXY BASELINE ===")
    print(f"Samples: {len(errors)}")
    print(f"Mean angular error:   {errors.mean():.3f} deg")
    print(f"Median angular error: {np.median(errors):.3f} deg")
    print(f"Std angular error:    {errors.std():.3f} deg")

    results_dir = Path("results/head_proxy")
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "video": merged["video"],
        "id": merged["id"],
        "gaze_t": merged["gaze_t"],
        "head_t": merged["head_t"],
        "angular_error_deg": errors,
    }).to_csv(results_dir / "head_proxy_errors.csv", index=False)

    summary = pd.DataFrame([{
        "method": "Head Proxy",
        "samples": len(errors),
        "mean_angular_error_deg": errors.mean(),
        "median_angular_error_deg": np.median(errors),
        "std_angular_error_deg": errors.std(),
    }])

    summary.to_csv(results_dir / "head_proxy_summary.csv", index=False)
    print("\nSaved results to results/head_proxy/")

if __name__ == "__main__":
    main()