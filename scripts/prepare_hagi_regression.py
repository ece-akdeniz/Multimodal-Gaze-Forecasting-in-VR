from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_all_csvs(folder: Path, cols):
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)


def load_split_videos(split_file: Path):
    text = split_file.read_text()
    tokens = (
        text.replace(",", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace("'", " ")
        .replace('"', " ")
        .split()
    )

    videos = []
    for token in tokens:
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            videos.append(int(digits))
    return set(videos)


def synchronize_gaze_head(gaze, head, tolerance=0.05):
    gaze = gaze.rename(columns={"u": "gaze_u", "v": "gaze_v", "t": "gaze_t"})
    head = head.rename(columns={"u": "head_u", "v": "head_v", "t": "head_t"})

    # gaze timestamps are ms, head timestamps are seconds
    gaze["gaze_t"] = gaze["gaze_t"] / 1000.0

    gaze = gaze.sort_values(["video", "id", "gaze_t"])
    head = head.sort_values(["video", "id", "head_t"])

    merged_parts = []

    for (video, pid), g in gaze.groupby(["video", "id"]):
        h = head[(head["video"] == video) & (head["id"] == pid)]
        if h.empty:
            continue

        m = pd.merge_asof(
            g.sort_values("gaze_t"),
            h.sort_values("head_t"),
            left_on="gaze_t",
            right_on="head_t",
            by=["video", "id"],
            direction="nearest",
            tolerance=tolerance,
        )

        m = m.dropna(subset=["head_u", "head_v"])
        if len(m) > 0:
            merged_parts.append(m)

    if not merged_parts:
        raise RuntimeError("No gaze/head samples matched.")

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.sort_values(["video", "id", "gaze_t"]).reset_index(drop=True)
    return merged


def uv_to_xyz(u, v):
    lon = (u - 0.5) * 2.0 * np.pi
    lat = (0.5 - v) * np.pi

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    return np.stack([x, y, z], axis=-1).astype(np.float32)


def create_windows(df, seq_len=10, max_samples=None):
    xs = []
    ys = []
    fixation_labels = []
    
    stop = False

    for (video, pid), group in df.groupby(["video", "id"]):
        if stop:
            break

        group = group.sort_values("gaze_t").reset_index(drop=True)

        head_uv = group[["head_u", "head_v"]].to_numpy(dtype=np.float32)
        gaze_uv = group[["gaze_u", "gaze_v"]].to_numpy(dtype=np.float32)
        fix = group["fixation"].to_numpy(dtype=np.int64)

        head_xyz = uv_to_xyz(head_uv[:, 0], head_uv[:, 1])
        gaze_xyz = uv_to_xyz(gaze_uv[:, 0], gaze_uv[:, 1])

        for i in range(seq_len, len(group)):
            x = head_xyz[i - seq_len:i].reshape(-1)
            y = gaze_xyz[i]

            xs.append(x)
            ys.append(y)
            fixation_labels.append(fix[i])

            if max_samples is not None and len(xs) >= max_samples:
                stop = True
                break

    feature_dim = seq_len * 3

    if not xs:
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.array(fixation_labels, dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/D-SAV360")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--out", default="data/processed/hagi_regression_fold0")
    parser.add_argument("--seq-len", default=10, type=int)
    parser.add_argument("--tolerance", default=0.05, type=float)
    parser.add_argument("--max-train-samples", default=1000000, type=int)
    parser.add_argument("--max-test-samples", default=200000, type=int)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-videos", default=2, type=int)
    parser.add_argument("--max-users", default=2, type=int)
    args = parser.parse_args()

    root = Path(args.root)
    gaze_dir = root / "gaze"
    head_dir = root / "head"
    split_dir = Path(args.splits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_videos = load_split_videos(split_dir / f"train_{args.fold}.txt")
    test_videos = load_split_videos(split_dir / f"test_{args.fold}.txt")

    print("Loading gaze...")
    gaze = load_all_csvs(
        gaze_dir,
        ["video", "stereo", "frame", "t", "u", "v", "id", "fixation", "valid_fix_classification"],
    )

    print("Loading head...")
    head = load_all_csvs(
        head_dir,
        ["video", "stereo", "t", "u", "v", "id"],
    )
    
    print("Cleaning gaze data while keeping both fixations and saccades...")
    before = len(gaze)

    gaze = gaze[
        (gaze["t"] >= 0)
        & (gaze["u"].between(0, 1))
        & (gaze["v"].between(0, 1))
    ].copy()

    gaze["fixation"] = gaze["fixation"].astype(bool).astype(np.int64)

    after = len(gaze)

    print("Gaze samples before cleaning:", before)
    print("Gaze samples after cleaning:", after)
    print("Removed corrupted samples:", before - after)
    print("\nFixation distribution:")
    print(gaze["fixation"].value_counts())

    if args.tiny:
        common_videos = sorted(set(gaze["video"].unique()) & set(head["video"].unique()))
        tiny_videos = set(common_videos[: args.max_videos])

        common_users = sorted(set(gaze["id"].unique()) & set(head["id"].unique()))
        tiny_users = set(common_users[: args.max_users])

        print("Tiny mode ON")
        print("Videos:", sorted(tiny_videos))
        print("Users:", sorted(tiny_users))

        gaze = gaze[gaze["video"].isin(tiny_videos) & gaze["id"].isin(tiny_users)]
        head = head[head["video"].isin(tiny_videos) & head["id"].isin(tiny_users)]

        train_videos = tiny_videos
        test_videos = tiny_videos

    print("Synchronizing gaze/head...")
    merged = synchronize_gaze_head(gaze, head, tolerance=args.tolerance)
    print("Merged samples:", len(merged))

    train_df = merged[merged["video"].isin(train_videos)].copy()
    test_df = merged[merged["video"].isin(test_videos)].copy()

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    print("Creating train windows...")
    trainingX, trainingY, trainingFixation = create_windows(
        train_df,
        seq_len=args.seq_len,
        max_samples=args.max_train_samples,
    )

    print("Creating test windows...")
    testX, testY, testFixation = create_windows(
        test_df,
        seq_len=args.seq_len,
        max_samples=args.max_test_samples,
    )

    print("\n=== OUTPUT SHAPES ===")
    print("trainingX:", trainingX.shape)
    print("trainingY:", trainingY.shape)
    print("trainingFixation:", trainingFixation.shape)
    print("testX:", testX.shape)
    print("testY:", testY.shape)
    print("testFixation:", testFixation.shape)

    np.save(out_dir / "trainingX.npy", trainingX)
    np.save(out_dir / "trainingY.npy", trainingY)
    np.save(out_dir / "trainingFixation.npy", trainingFixation)
    np.save(out_dir / "testX.npy", testX)
    np.save(out_dir / "testY.npy", testY)
    np.save(out_dir / "testFixation.npy", testFixation)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()