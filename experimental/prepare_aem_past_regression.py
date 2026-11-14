"""
preprocess_aem.py

Preprocessing pipeline for HAGI++ Regression with AEM spatial audio features.

Audio source: Audio Energy Maps (AEM) — PNG images of the 360° panorama
              showing spatial sound energy distribution per video frame.

Output per split:
    trainingX.npy       (N, seq_len*3 + 128)  head xyz window + AEM feature
    trainingY.npy       (N, 3)                 target gaze xyz unit vector
    trainingFixation.npy (N,)                  fixation label (0/1)
    testX.npy
    testY.npy
    testFixation.npy

Key design decisions:
    - Temporal sync only: merge gaze (ms) and head (sec) via nearest-timestamp
    - Remove negative timestamps and out-of-range UV values
    - AEM at t+1 (next frame) — valid since videos are pre-recorded
    - AEM downsampled from 60fps → 30fps by taking every 2nd frame
    - AEM resized to (8, 16) grid → flattened to 128-dim feature
    - Head UV → 3D unit vector (xyz) before windowing
    - Sliding window: seq_len head frames → predict gaze at next frame
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_all_csvs(folder: Path, cols):
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return pd.concat(
        [pd.read_csv(f, usecols=cols) for f in files],
        ignore_index=True
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


# ---------------------------------------------------------------------------
# AEM loading
# ---------------------------------------------------------------------------

def load_aem_features(aem_dir: Path, grid_size=(8, 16)):
    """
    Load all AEM PNG frames for one video, downsample 60fps → 30fps,
    and resize each map to grid_size, returning a flattened feature array.

    Args:
        aem_dir:   Path to folder containing 0000.png, 0001.png, ...
        grid_size: (rows, cols) to resize each map to. Default (8,16) → 128-dim.

    Returns:
        np.ndarray of shape (N_frames_30hz, grid_size[0]*grid_size[1])
        Each row is a flattened, normalized energy map at 30Hz.
        Returns None if directory doesn't exist or has no PNG files.
    """
    if not aem_dir.exists():
        return None

    files = sorted(aem_dir.glob("*.png"))
    if not files:
        return None

    features = []
    for f in files:
        img = np.array(Image.open(f).convert('L'), dtype=np.float32)  # (H, W)
        img = img / 255.0

        # resize to small fixed grid
        small = np.array(
            Image.fromarray((img * 255).astype(np.uint8))
            .resize((grid_size[1], grid_size[0]), Image.BILINEAR)
        , dtype=np.float32) / 255.0                                    # (8, 16)

        features.append(small.flatten())                                # (128,)

    features = np.array(features, dtype=np.float32)                    # (N_60hz, 128)

    # downsample 60fps → 30fps
    features_30hz = features[::2]                                       # (N_30hz, 128)

    return features_30hz


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def uv_to_xyz(u, v):
    """Convert equirectangular UV coordinates to 3D unit vectors."""
    lon = (u - 0.5) * 2.0 * np.pi
    lat = (0.5 - v) * np.pi

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal synchronization
# ---------------------------------------------------------------------------

def synchronize_gaze_head(gaze: pd.DataFrame, head: pd.DataFrame, tolerance=0.05):
    """
    Align gaze and head signals by timestamp using nearest-match merge.

    gaze timestamps are in milliseconds → converted to seconds before merging.
    head timestamps are already in seconds.
    tolerance: max allowed time difference in seconds (default 50ms).
    """
    gaze = gaze.copy()
    head = head.copy()

    gaze = gaze.rename(columns={"u": "gaze_u", "v": "gaze_v", "t": "gaze_t"})
    head = head.rename(columns={"u": "head_u", "v": "head_v", "t": "head_t"})

    # convert gaze ms → seconds
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
        raise RuntimeError("No gaze/head samples matched after synchronization.")

    merged = pd.concat(merged_parts, ignore_index=True)
    return merged.sort_values(["video", "id", "gaze_t"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sliding window dataset creation
# ---------------------------------------------------------------------------

def create_windows(
    df: pd.DataFrame,
    aem_root: Path,
    seq_len: int = 30,            # head-seq-len
    audio_future_len: int = 10,   # audio-future-seq-len
    audio_past_len: int = 10,     # audio-past-seq-len
    fps: int = 30,
    grid_size: tuple = (8, 16),
    use_next_frame: bool = True,
    max_samples: int = None,
):
    """
    Build (X, Y) pairs using a sliding window over each (video, participant) recording.

    X = [head_xyz_window (seq_len*3,) | aem_feature (128,)]  shape: (seq_len*3 + 128,)
    Y = gaze_xyz at the target frame                           shape: (3,)

    Args:
        df:             Synchronized gaze+head dataframe.
        aem_root:       Root folder containing AEM subfolders per video (e.g. AEM/0001/).
        seq_len:        Number of past head frames to use as input.
        fps:            Target frame rate (30Hz after downsampling).
        grid_size:      AEM resize grid (rows, cols). Default (8,16) → 128-dim.
        use_next_frame: If True, use AEM at t+1 (next frame). Valid for pre-recorded video.
                        If False, use AEM at t (current frame).
        max_samples:    Stop after this many samples (for debugging).

    Returns:
        X:          np.ndarray (N, seq_len*3 + 128)
        Y:          np.ndarray (N, 3)
        fixations:  np.ndarray (N,) int64 — 1=fixation, 0=saccade
        stats:      dict with counts
    """
    audio_dim = grid_size[0] * grid_size[1]  # 128
    feature_dim = seq_len * 3 + (audio_future_len + audio_past_len) * audio_dim  # 30*3 + 20*128 = 2590

    xs, ys, fixation_labels = [], [], []
    missing_aem = 0
    stop = False

    aem_cache = {}

    for (video, pid), group in df.groupby(["video", "id"]):
        if stop:
            break

        video_int = int(video)
        video_str = f"{video_int:04d}"

        # load AEM for this video (cached)
        if video_int not in aem_cache:
            aem_dir = aem_root / video_str
            aem_cache[video_int] = load_aem_features(aem_dir, grid_size=grid_size)

        aem_feats = aem_cache[video_int]  # (N_30hz, 128) or None

        if aem_feats is None:
            print(f"  Warning: No AEM found for video {video_str}, skipping.")
            missing_aem += len(group)
            continue

        group = group.sort_values("gaze_t").reset_index(drop=True)

        head_uv  = group[["head_u", "head_v"]].to_numpy(dtype=np.float32)
        gaze_uv  = group[["gaze_u", "gaze_v"]].to_numpy(dtype=np.float32)
        fix      = group["fixation"].to_numpy(dtype=np.int64)
        times    = group["gaze_t"].to_numpy(dtype=np.float32)  # in seconds

        head_xyz = uv_to_xyz(head_uv[:, 0], head_uv[:, 1])    # (T, 3)
        gaze_xyz = uv_to_xyz(gaze_uv[:, 0], gaze_uv[:, 1])    # (T, 3)

        for i in range(seq_len, len(group)):
            if fix[i] == 0:   # skip if target is a saccade
                continue

            # AEM frame index for target timestep
            target_time = times[i]
            aem_offset  = 1 if use_next_frame else 0

            target_idx   = int(round(target_time * fps))

            # future: t+1 to t+audio_seq_len
            future_start = target_idx + 1
            future_end   = future_start + audio_future_len

            # past: t-audio_past_len to t (inclusive)
            past_start   = target_idx - audio_past_len
            past_end     = target_idx

            if (future_end > len(aem_feats) or past_start < 0):
                missing_aem += 1
                continue

            head_window  = head_xyz[i - seq_len : i].reshape(-1)  # (30*3,) = (90,)
            aem_past    = aem_feats[past_start:past_end].reshape(-1)    # (past_len*128,)
            aem_future  = aem_feats[future_start:future_end].reshape(-1) # (future_len*128,)
            aem_feature = np.concatenate([aem_past, aem_future], axis=0) # (20*128,)

            x = np.concatenate([head_window, aem_feature], axis=0).astype(np.float32)
            y = gaze_xyz[i].astype(np.float32)

            xs.append(x)
            ys.append(y)
            fixation_labels.append(fix[i])

            if max_samples is not None and len(xs) >= max_samples:
                stop = True
                break

    stats = {
        "samples": len(xs),
        "missing_aem": missing_aem,
        "feature_dim": feature_dim,
    }

    if not xs:
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, 3),           dtype=np.float32),
            np.empty((0,),             dtype=np.int64),
            stats,
        )

    return (
        np.stack(xs),
        np.stack(ys),
        np.array(fixation_labels, dtype=np.int64),
        stats,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess D-SAV360 with AEM spatial audio features.")
    parser.add_argument("--root",       default="data/raw/D-SAV360",
                        help="Root folder of D-SAV360 dataset (contains gaze/, head/ subfolders)")
    parser.add_argument("--aem-root",   default="data/raw/D-SAV360/AEM",
                        help="Root folder of AEM PNGs (contains subfolders 0001/, 0002/, ...)")
    parser.add_argument("--splits",     default="data/splits",
                        help="Folder containing train_0.txt, test_0.txt, etc.")
    parser.add_argument("--fold",       default=0, type=int)
    parser.add_argument("--out",        default="data/processed/hagi_aem_regression_fold0")
    parser.add_argument("--seq-len",    default=30, type=int,
                        help="Number of past head frames per input window (default 30 = 1s at 30Hz)")
    parser.add_argument("--audio-future-len", default=10, type=int,
                        help="Number of future AEM frames per input window (default 10 = 0.33s at 30Hz)")
    parser.add_argument("--audio-past-len", default=10, type=int,
                        help="Number of past AEM frames per input window (default 10 = 0.33s at 30Hz)")
    parser.add_argument("--fps",        default=30, type=int)
    parser.add_argument("--grid-rows",  default=8,  type=int, help="AEM resize grid rows")
    parser.add_argument("--grid-cols",  default=16, type=int, help="AEM resize grid cols")
    parser.add_argument("--use-next-frame", action="store_true", default=True,
                        help="Use AEM at t+1 (valid for pre-recorded video). Default: True.")
    parser.add_argument("--use-current-frame", dest="use_next_frame", action="store_false",
                        help="Use AEM at t instead of t+1 (for ablation).")
    parser.add_argument("--tolerance",  default=0.05, type=float,
                        help="Max gaze/head timestamp difference in seconds (default 50ms)")
    parser.add_argument("--max-train-samples", default=1000000, type=int)
    parser.add_argument("--max-test-samples",  default=200000,  type=int)
    parser.add_argument("--tiny", action="store_true",
                        help="Debug mode: use only 2 videos and 2 users")
    parser.add_argument("--max-videos", default=2, type=int)
    parser.add_argument("--max-users",  default=2, type=int)
    args = parser.parse_args()

    root     = Path(args.root)
    aem_root = Path(args.aem_root)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_size = (args.grid_rows, args.grid_cols)
    audio_dim = grid_size[0] * grid_size[1]

    print(f"AEM grid size: {grid_size} → {audio_dim}-dim feature per frame")
    print(f"Using AEM at {'t+1 (next frame)' if args.use_next_frame else 't (current frame)'}")

    # --- load split files ---
    split_dir    = Path(args.splits)
    train_videos = load_split_videos(split_dir / f"train_{args.fold}.txt")
    test_videos  = load_split_videos(split_dir / f"test_{args.fold}.txt")
    # use half the train videos
    train_videos = set(sorted(train_videos)[:len(train_videos)//2])
    print(f"Using {len(train_videos)} train videos (half fold)")
    # print(f"Train videos: {len(train_videos)} | Test videos: {len(test_videos)}")

    # --- load CSVs ---
    print("\nLoading gaze CSVs...")
    gaze = load_all_csvs(
        root / "gaze",
        ["video", "stereo", "frame", "t", "u", "v", "id", "fixation", "valid_fix_classification"],
    )

    print("Loading head CSVs...")
    head = load_all_csvs(
        root / "head",
        ["video", "stereo", "t", "u", "v", "id"],
    )

    # --- clean gaze ---
    print("\nCleaning gaze data...")
    before = len(gaze)

    gaze = gaze[
        (gaze["t"] >= 0)                  # remove negative timestamps
        & (gaze["u"].between(0, 1))        # remove out-of-range UV
        & (gaze["v"].between(0, 1))
    ].copy()

    gaze["fixation"] = gaze["fixation"].astype(bool).astype(np.int64)

    after = len(gaze)
    print(f"  Before: {before:,} | After: {after:,} | Removed: {before - after:,}")
    print(f"  Fixation distribution:\n{gaze['fixation'].value_counts().to_string()}")

    # --- tiny mode for debugging ---
    if args.tiny:
        common_videos = sorted(set(gaze["video"].unique()) & set(head["video"].unique()))
        tiny_videos   = set(common_videos[: args.max_videos])
        common_users  = sorted(set(gaze["id"].unique()) & set(head["id"].unique()))
        tiny_users    = set(common_users[: args.max_users])

        print(f"\nTiny mode: videos={sorted(tiny_videos)}, users={sorted(tiny_users)}")

        gaze = gaze[gaze["video"].isin(tiny_videos) & gaze["id"].isin(tiny_users)]
        head = head[head["video"].isin(tiny_videos) & head["id"].isin(tiny_users)]

        train_videos = tiny_videos
        test_videos  = tiny_videos

    # --- synchronize ---
    print("\nSynchronizing gaze and head timestamps...")
    merged = synchronize_gaze_head(gaze, head, tolerance=args.tolerance)
    print(f"  Merged samples: {len(merged):,}")

    train_df = merged[merged["video"].isin(train_videos)].copy()
    test_df  = merged[merged["video"].isin(test_videos)].copy()
    print(f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    # --- create windows ---
    print("\nCreating train windows...")
    trainX, trainY, trainFix, train_stats = create_windows(
        train_df,
        aem_root        = aem_root,
        seq_len         = args.seq_len,
        audio_future_len   = args.audio_future_len,
        audio_past_len = args.audio_past_len,
        fps             = args.fps,
        grid_size       = grid_size,
        use_next_frame  = args.use_next_frame,
        max_samples     = args.max_train_samples,
    )

    print("Creating test windows...")
    testX, testY, testFix, test_stats = create_windows(
        test_df,
        aem_root        = aem_root,
        seq_len         = args.seq_len, 
        audio_future_len   = args.audio_future_len,
        audio_past_len = args.audio_past_len,
        fps             = args.fps,
        grid_size       = grid_size,
        use_next_frame  = args.use_next_frame,
        max_samples     = args.max_test_samples,
    )

    # normalize features
    mean = trainX.mean(axis=0)
    std  = trainX.std(axis=0) + 1e-8
    np.save(out_dir / "feature_mean.npy", mean)
    np.save(out_dir / "feature_std.npy",  std)

    trainX = (trainX - mean) / std
    testX  = (testX  - mean) / std

    # --- report ---
    print("\n=== OUTPUT SHAPES ===")
    print(f"  trainX:   {trainX.shape}   (head_window + AEM feature)")
    print(f"  trainY:   {trainY.shape}   (gaze xyz unit vector)")
    print(f"  testX:    {testX.shape}")
    print(f"  testY:    {testY.shape}")
    print(f"  testFix:  {testFix.shape}  (1=fixation, 0=saccade)")

    print("\n=== STATS ===")
    print(f"  Train: {train_stats}")
    print(f"  Test:  {test_stats}")

    # --- save ---
    np.save(out_dir / "trainingX.npy",        trainX)
    np.save(out_dir / "trainingY.npy",        trainY)
    np.save(out_dir / "trainingFixation.npy", trainFix)
    np.save(out_dir / "testX.npy",            testX)
    np.save(out_dir / "testY.npy",            testY)
    np.save(out_dir / "testFixation.npy",     testFix)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
