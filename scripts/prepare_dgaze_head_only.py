from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_all_csvs(folder: Path, cols):
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return pd.concat(
        [pd.read_csv(f, usecols=cols) for f in files],
        ignore_index=True,
    )

def load_split_videos(split_file: Path):
    text = split_file.read_text()

    # Split on commas, spaces, tabs, and newlines
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

    # D-SAV360 gaze timestamps are in ms; head timestamps are in seconds.
    gaze["gaze_t"] = gaze["gaze_t"] / 1000.0

    gaze = gaze.sort_values(["video", "id", "gaze_t"])
    head = head.sort_values(["video", "id", "head_t"])

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
            tolerance=tolerance,
        )

        m = m.dropna(subset=["head_u", "head_v"])
        if len(m) > 0:
            merged_parts.append(m)

    if not merged_parts:
        raise RuntimeError("No gaze/head samples matched. Check timestamps or tolerance.")

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.sort_values(["video", "id", "gaze_t"]).reset_index(drop=True)
    return merged

def downsample_group(df, step):
    if step <= 1:
        return df.reset_index(drop=True)

    parts = []
    for (_, _), group in df.groupby(["video", "id"]):
        parts.append(group.iloc[::step].copy())

    return pd.concat(parts, ignore_index=True)

def create_windows(df, seq_len=10):
    xs = []
    ys = []

    for (video, pid), group in df.groupby(["video", "id"]):
        group = group.sort_values("gaze_t")

        head = group[["head_u", "head_v"]].to_numpy(dtype=np.float32)
        gaze = group[["gaze_u", "gaze_v"]].to_numpy(dtype=np.float32)

        # Input: 10 past/current head samples.
        # Target: next gaze sample.
        for i in range(seq_len, len(group)):
            x = head[i - seq_len:i].reshape(-1)
            y = gaze[i]
            xs.append(x)
            ys.append(y)

    if not xs:
        return np.empty((0, seq_len * 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/D-SAV360", type=str)
    parser.add_argument("--splits", default="data/splits", type=str)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--out", default="data/processed/dgaze_head_only_tiny", type=str)
    parser.add_argument("--seq-len", default=10, type=int)
    parser.add_argument("--downsample-step", default=2, type=int, help="60 Hz -> 30 Hz means step=2")
    parser.add_argument("--tolerance", default=0.05, type=float)
    parser.add_argument("--tiny", action="store_true", help="Use only 2 videos and 2 users for debugging")
    parser.add_argument("--max-videos", default=2, type=int)
    parser.add_argument("--max-users", default=2, type=int)
    args = parser.parse_args()

    root = Path(args.root)
    gaze_dir = root / "gaze"
    head_dir = root / "head"
    splits_dir = Path(args.splits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading split files...")
    train_videos = load_split_videos(splits_dir / f"train_{args.fold}.txt")
    test_videos = load_split_videos(splits_dir / f"test_{args.fold}.txt")

    print(f"Train videos in split: {len(train_videos)}")
    print(f"Test videos in split: {len(test_videos)}")

    print("Loading gaze...")
    gaze = load_all_csvs(
        gaze_dir,
        ["video", "stereo", "frame", "t", "u", "v", "id"],
    )

    print("Loading head...")
    head = load_all_csvs(
        head_dir,
        ["video", "stereo", "t", "u", "v", "id"],
    )

    if args.tiny:
        common_videos = sorted(set(gaze["video"].unique()) & set(head["video"].unique()))
        tiny_videos = set(common_videos[:args.max_videos])

        common_users = sorted(set(gaze["id"].unique()) & set(head["id"].unique()))
        tiny_users = set(common_users[:args.max_users])

        print(f"Tiny mode ON")
        print(f"Using videos: {sorted(tiny_videos)}")
        print(f"Using users: {sorted(tiny_users)}")

        gaze = gaze[gaze["video"].isin(tiny_videos) & gaze["id"].isin(tiny_users)]
        head = head[head["video"].isin(tiny_videos) & head["id"].isin(tiny_users)]

        # For tiny debug mode, use same videos for train/test split if official split
        # does not contain the selected videos.
        train_videos = tiny_videos
        test_videos = tiny_videos

    print("Synchronizing gaze/head...")
    merged = synchronize_gaze_head(gaze, head, tolerance=args.tolerance)
    print(f"Matched samples before downsampling: {len(merged)}")

    print(f"Downsampling with step={args.downsample_step}...")
    merged = downsample_group(merged, step=args.downsample_step)
    print(f"Matched samples after downsampling: {len(merged)}")
    print("Merged columns:", merged.columns.tolist())

    train_df = merged[merged["video"].isin(train_videos)].copy()
    test_df = merged[merged["video"].isin(test_videos)].copy()

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    print("Creating train windows...")
    trainingX, trainingY = create_windows(train_df, seq_len=args.seq_len)

    print("Creating test windows...")
    testX, testY = create_windows(test_df, seq_len=args.seq_len)

    print("\n=== OUTPUT SHAPES ===")
    print("trainingX:", trainingX.shape)
    print("trainingY:", trainingY.shape)
    print("testX:", testX.shape)
    print("testY:", testY.shape)

    np.save(out_dir / "trainingX.npy", trainingX)
    np.save(out_dir / "trainingY.npy", trainingY)
    np.save(out_dir / "testX.npy", testX)
    np.save(out_dir / "testY.npy", testY)

    print(f"\nSaved DGaze Head-only dataset to: {out_dir}")


if __name__ == "__main__":
    main()