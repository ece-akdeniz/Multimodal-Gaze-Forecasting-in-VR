from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image


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


def load_saliency_feature(
    saliency_root: Path,
    saliency_index,
    video,
    frame,
    size=24,
    max_frame_distance=4,
):
    video = int(video)
    frame = int(frame)

    if video not in saliency_index:
        return None, None

    available_frames = saliency_index[video]

    nearest_idx = np.abs(available_frames - frame).argmin()
    nearest_frame = int(available_frames[nearest_idx])

    if abs(nearest_frame - frame) > max_frame_distance:
        return None, None

    saliency_file = saliency_root / f"{video:04d}" / f"{video:04d}_{nearest_frame:04d}.png"

    if not saliency_file.exists():
        return None, None

    img = Image.open(saliency_file).convert("L")
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr.reshape(-1), nearest_frame


def build_saliency_index(saliency_root: Path):
    """
    Builds a dictionary:
    {
        video_id: np.array([available saliency frame numbers])
    }
    """
    saliency_index = {}

    for video_dir in saliency_root.iterdir():
        if not video_dir.is_dir() or not video_dir.name.isdigit():
            continue

        video_id = int(video_dir.name)
        frames = []

        for file in video_dir.glob("*.png"):
            try:
                frame = int(file.stem.split("_")[1])
                frames.append(frame)
            except (IndexError, ValueError):
                continue

        if frames:
            saliency_index[video_id] = np.array(sorted(frames), dtype=np.int32)

    return saliency_index


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

    if not parts:
        return df.iloc[0:0].copy()

    return pd.concat(parts, ignore_index=True)


def create_windows_with_saliency(
    df,
    saliency_root,
    saliency_index,
    seq_len=10,
    saliency_size=24,
    max_frame_distance=4,
    max_samples=None,
):
    xs = []
    ys = []
    missing_saliency = 0
    matched_saliency = 0
    frame_distances = []

    stop = False

    for (video, pid), group in df.groupby(["video", "id"]):
        if stop:
            break

        group = group.sort_values("gaze_t").reset_index(drop=True)

        head = group[["head_u", "head_v"]].to_numpy(dtype=np.float32)
        gaze = group[["gaze_u", "gaze_v"]].to_numpy(dtype=np.float32)
        frames = group["frame"].to_numpy()

        for i in range(seq_len, len(group)):
            target_frame = int(frames[i])

            saliency, nearest_frame = load_saliency_feature(
                saliency_root=saliency_root,
                saliency_index=saliency_index,
                video=video,
                frame=target_frame,
                size=saliency_size,
                max_frame_distance=max_frame_distance,
            )

            if saliency is None:
                missing_saliency += 1
                continue

            matched_saliency += 1
            frame_distances.append(abs(nearest_frame - target_frame))

            head_window = head[i - seq_len:i].reshape(-1).astype(np.float32)
            x = np.concatenate([head_window, saliency], axis=0).astype(np.float32)
            y = gaze[i].astype(np.float32)

            xs.append(x)
            ys.append(y)

            if max_samples is not None and len(xs) >= max_samples:
                stop = True
                break

    feature_dim = seq_len * 2 + saliency_size * saliency_size

    stats = {
        "missing_saliency": missing_saliency,
        "matched_saliency": matched_saliency,
        "mean_frame_distance": float(np.mean(frame_distances)) if frame_distances else None,
        "max_frame_distance_used": int(np.max(frame_distances)) if frame_distances else None,
        "sample_limit": max_samples,
    }

    if not xs:
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            stats,
        )

    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        stats,
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default="data/raw/D-SAV360", type=str)
    parser.add_argument("--splits", default="data/splits", type=str)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--out", default="data/processed/dgaze_full_tiny", type=str)


    parser.add_argument("--seq-len", default=10, type=int)
    parser.add_argument("--saliency-size", default=24, type=int)
    parser.add_argument("--downsample-step", default=2, type=int)
    parser.add_argument("--tolerance", default=0.05, type=float)

    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-videos", default=2, type=int)
    parser.add_argument("--max-users", default=2, type=int)
    parser.add_argument("--max-frame-distance", default=4, type=int)
    parser.add_argument("--max-train-samples", default=None, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int)

    args = parser.parse_args()

    root = Path(args.root)
    gaze_dir = root / "gaze"
    head_dir = root / "head"
    saliency_root = root / "saliency"
    
    print("Building saliency index...")
    saliency_index = build_saliency_index(saliency_root)
    print(f"Videos with saliency maps: {len(saliency_index)}")

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
        saliency_videos = {
            int(p.name)
            for p in saliency_root.iterdir()
            if p.is_dir() and p.name.isdigit()
        }

        common_videos = sorted(
            set(gaze["video"].unique())
            & set(head["video"].unique())
            & saliency_videos
        )

        tiny_videos = set(common_videos[: args.max_videos])

        common_users = sorted(set(gaze["id"].unique()) & set(head["id"].unique()))
        tiny_users = set(common_users[: args.max_users])

        print("Tiny mode ON")
        print(f"Using videos: {sorted(tiny_videos)}")
        print(f"Using users: {sorted(tiny_users)}")

        gaze = gaze[gaze["video"].isin(tiny_videos) & gaze["id"].isin(tiny_users)]
        head = head[head["video"].isin(tiny_videos) & head["id"].isin(tiny_users)]

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

    print("Creating train windows with saliency...")
    trainingX, trainingY, train_stats = create_windows_with_saliency(
        train_df,
        saliency_root=saliency_root,
        saliency_index=saliency_index,
        seq_len=args.seq_len,
        saliency_size=args.saliency_size,
        max_frame_distance=args.max_frame_distance,
        max_samples=args.max_train_samples,
    )

    print("Creating test windows with saliency...")
    testX, testY, test_stats = create_windows_with_saliency(
        test_df,
        saliency_root=saliency_root,
        saliency_index=saliency_index,
        seq_len=args.seq_len,
        saliency_size=args.saliency_size,
        max_frame_distance=args.max_frame_distance,
        max_samples=args.max_test_samples,
    )

    print("\n=== OUTPUT SHAPES ===")
    print("trainingX:", trainingX.shape)
    print("trainingY:", trainingY.shape)
    print("testX:", testX.shape)
    print("testY:", testY.shape)

    print("\n=== SALIENCY ===")
    print("Train saliency stats:", train_stats)
    print("Test saliency stats:", test_stats)

    expected_feature_dim = args.seq_len * 2 + args.saliency_size * args.saliency_size
    print("\nExpected feature dimension:", expected_feature_dim)

    np.save(out_dir / "trainingX.npy", trainingX)
    np.save(out_dir / "trainingY.npy", trainingY)
    np.save(out_dir / "testX.npy", testX)
    np.save(out_dir / "testY.npy", testY)

    print(f"\nSaved DGaze Full dataset to: {out_dir}")


if __name__ == "__main__":
    main()