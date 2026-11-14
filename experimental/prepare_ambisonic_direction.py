"""
preprocess_ambisonic.py

Preprocessing pipeline for HAGI++ Regression with ambisonic direction features.

Instead of AEM spatial maps (128-dim per frame), we extract the dominant
sound direction from the raw ambisonic WAV file as a 3D unit vector (3-dim),
directly comparable to head and gaze xyz representations.

Ambisonic format (AmbiX / B-format):
    W = omnidirectional (total pressure)
    X = front-back
    Y = left-right
    Z = up-down

Dominant sound direction per frame:
    azimuth   = arctan2(Y_rms, X_rms)
    elevation = arctan2(Z_rms, sqrt(X_rms^2 + Y_rms^2))
    → convert to 3D unit vector (same space as head/gaze)

Input X shape:  (N, seq_len*3 + audio_seq_len*3)  e.g. (N, 90+30) = (N, 120)
Output Y shape: (N, 3)  gaze xyz unit vector

Usage:
    # tiny debug
    python preprocess_ambisonic.py --tiny

    # full fold 0
    python preprocess_ambisonic.py \
        --root data/raw/D-SAV360 \
        --audio-root data/raw/D-SAV360/videos_extracted \
        --splits data/splits \
        --fold 0 \
        --out data/processed/hagi_ambisonic_fold0
"""

from pathlib import Path
import argparse
import wave
import numpy as np
import pandas as pd


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


def find_wav(audio_root: Path, video_id: int):
    vid = f"{video_id:04d}"
    matches = list(audio_root.glob(f"{vid}/{vid}/{vid}.wav"))
    if matches:
        return matches[0]
    matches = list(audio_root.glob(f"**/{vid}.wav"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Ambisonic direction extraction
# ---------------------------------------------------------------------------

def compute_ambisonic_direction(wav_path: Path, fps=30, window_sec=0.5):
    """
    Extract dominant sound direction per frame from ambisonic WAV file.

    Ambisonic channels (4-channel B-format):
        ch0 = W (omnidirectional)
        ch1 = X (front-back)
        ch2 = Y (left-right)
        ch3 = Z (up-down)

    For each frame:
        1. Compute RMS of X, Y, Z channels over a window
        2. Derive azimuth and elevation from X, Y, Z energy
        3. Convert to 3D unit vector (same coordinate space as head/gaze)

    Args:
        wav_path:   Path to 4-channel ambisonic WAV file
        fps:        Target frame rate (30Hz)
        window_sec: RMS window duration in seconds (default 0.5s)

    Returns:
        np.ndarray of shape (N_frames, 3) — xyz unit vectors at target fps
        Returns None if file can't be loaded or isn't 4-channel
    """
    try:
        with wave.open(str(wav_path), "rb") as w:
            channels  = w.getnchannels()
            sr        = w.getframerate()
            n_frames  = w.getnframes()
            raw       = w.readframes(n_frames)
    except Exception as e:
        print(f"  Warning: Could not load {wav_path}: {e}")
        return None

    if channels != 4:
        print(f"  Warning: Expected 4-channel ambisonic, got {channels} channels in {wav_path}")
        return None

    # decode to float
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    audio = audio.reshape(-1, channels) / 32768.0  # (N_samples, 4)

    duration = len(audio) / sr
    n_steps  = int(np.ceil(duration * fps))
    win      = int(window_sec * sr)

    directions = np.zeros((n_steps, 3), dtype=np.float32)

    for i in range(n_steps):
        center = int(i / fps * sr)
        start  = max(0, center)
        end    = min(len(audio), center + win)

        segment = audio[start:end]  # (win_samples, 4)
        if len(segment) == 0:
            continue

        # RMS per channel: W, X, Y, Z
        rms = np.sqrt(np.mean(segment ** 2, axis=0))  # (4,)
        W, X, Y, Z = rms[0], rms[1], rms[2], rms[3]

        # derive dominant sound direction from X, Y, Z energy
        # azimuth: horizontal angle (yaw), 0=front, pi/2=left
        azimuth   = np.arctan2(Y, X)

        # elevation: vertical angle (pitch), 0=horizon, pi/2=up
        horiz_mag = np.sqrt(X**2 + Y**2)
        elevation = np.arctan2(Z, horiz_mag + 1e-8)

        # convert spherical → 3D unit vector
        # same convention as uv_to_xyz:
        #   x = cos(lat)*sin(lon)
        #   y = sin(lat)
        #   z = cos(lat)*cos(lon)
        x = np.cos(elevation) * np.sin(azimuth)
        y = np.sin(elevation)
        z = np.cos(elevation) * np.cos(azimuth)

        vec = np.array([x, y, z], dtype=np.float32)

        # normalize (should already be unit but be safe)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        directions[i] = vec

    return directions  # (N_frames_at_fps, 3)


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
    gaze timestamps in ms → converted to seconds before merging.
    head timestamps already in seconds.
    """
    gaze = gaze.copy()
    head = head.copy()

    gaze = gaze.rename(columns={"u": "gaze_u", "v": "gaze_v", "t": "gaze_t"})
    head = head.rename(columns={"u": "head_u", "v": "head_v", "t": "head_t"})

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
# Sliding window dataset creation — vectorized
# ---------------------------------------------------------------------------

def create_windows(
    df: pd.DataFrame,
    audio_root: Path,
    seq_len: int = 30,
    audio_seq_len: int = 10,
    fps: int = 30,
    window_sec: float = 0.5,
    fixation_only: bool = True,
    max_samples: int = None,
):
    """
    Build (X, Y) pairs using a sliding window.

    X = [head_xyz_window (seq_len*3) | audio_direction_future (audio_seq_len*3)]
    Y = gaze_xyz at target frame

    Audio direction: future frames t+1 to t+audio_seq_len
    Valid since D-SAV360 videos are pre-recorded.

    Args:
        df:             Synchronized gaze+head dataframe
        audio_root:     Root folder containing WAV files
        seq_len:        Past head frames per window (default 30 = 1s at 30Hz)
        audio_seq_len:  Future audio frames (default 10 = 0.33s at 30Hz)
        fps:            Frame rate (30Hz)
        window_sec:     RMS window for ambisonic feature extraction
        fixation_only:  If True, only create samples where target is a fixation
        max_samples:    Stop after this many samples

    Returns:
        X, Y, fixations, stats
    """
    audio_dim   = 3  # xyz direction vector per frame
    feature_dim = seq_len * 3 + audio_seq_len * audio_dim  # e.g. 90 + 30 = 120

    xs, ys, fixation_labels = [], [], []
    missing_audio = 0
    audio_cache   = {}
    stop          = False

    for (video, pid), group in df.groupby(["video", "id"]):
        if stop:
            break

        video_int = int(video)

        # load ambisonic direction for this video (cached)
        if video_int not in audio_cache:
            wav_path = find_wav(audio_root, video_int)
            if wav_path is None:
                print(f"  Warning: No WAV found for video {video_int:04d}, skipping.")
                audio_cache[video_int] = None
            else:
                audio_cache[video_int] = compute_ambisonic_direction(
                    wav_path, fps=fps, window_sec=window_sec
                )

        audio_dirs = audio_cache[video_int]  # (N_frames, 3) or None

        if audio_dirs is None:
            missing_audio += len(group)
            continue

        group = group.sort_values("gaze_t").reset_index(drop=True)

        head_uv  = group[["head_u", "head_v"]].to_numpy(dtype=np.float32)
        gaze_uv  = group[["gaze_u", "gaze_v"]].to_numpy(dtype=np.float32)
        fix      = group["fixation"].to_numpy(dtype=np.int64)
        times    = group["gaze_t"].to_numpy(dtype=np.float32)

        head_xyz = uv_to_xyz(head_uv[:, 0], head_uv[:, 1])  # (T, 3)
        gaze_xyz = uv_to_xyz(gaze_uv[:, 0], gaze_uv[:, 1])  # (T, 3)

        # valid target indices
        if fixation_only:
            valid_i = np.array([
                i for i in range(seq_len, len(group))
                if fix[i] == 1
            ], dtype=np.int64)
        else:
            valid_i = np.arange(seq_len, len(group), dtype=np.int64)

        if len(valid_i) == 0:
            continue

        # compute audio frame indices
        target_times  = times[valid_i]
        target_frames = np.round(target_times * fps).astype(np.int64)
        future_starts = target_frames + 1
        future_ends   = future_starts + audio_seq_len

        # mask out-of-bounds
        in_bounds = (future_ends <= len(audio_dirs)) & (future_starts >= 0)
        valid_i       = valid_i[in_bounds]
        future_starts = future_starts[in_bounds]

        if len(valid_i) == 0:
            missing_audio += 1
            continue

        # build head windows — vectorized
        head_windows = np.stack([
            head_xyz[i - seq_len : i].reshape(-1)
            for i in valid_i
        ])  # (M, seq_len*3)

        # build future audio direction windows — vectorized
        future_idx  = future_starts[:, None] + np.arange(audio_seq_len)[None, :]  # (M, audio_seq_len)
        audio_future = audio_dirs[future_idx].reshape(len(valid_i), -1)            # (M, audio_seq_len*3)

        X_group = np.concatenate([head_windows, audio_future], axis=1).astype(np.float32)
        Y_group = gaze_xyz[valid_i].astype(np.float32)
        F_group = fix[valid_i]

        xs.append(X_group)
        ys.append(Y_group)
        fixation_labels.append(F_group)

        if max_samples is not None and sum(len(x) for x in xs) >= max_samples:
            stop = True
            break

    stats = {
        "samples":       sum(len(x) for x in xs) if xs else 0,
        "missing_audio": missing_audio,
        "feature_dim":   feature_dim,
    }

    if not xs:
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, 3),           dtype=np.float32),
            np.empty((0,),             dtype=np.int64),
            stats,
        )

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(fixation_labels, axis=0).astype(np.int64),
        stats,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess D-SAV360 with ambisonic direction features."
    )
    parser.add_argument("--root",        default="data/raw/D-SAV360",
                        help="D-SAV360 root folder (contains gaze/, head/)")
    parser.add_argument("--audio-root",  default="data/raw/D-SAV360/videos_extracted",
                        help="Root folder containing WAV files")
    parser.add_argument("--splits",      default="data/splits")
    parser.add_argument("--fold",        default=0, type=int)
    parser.add_argument("--out",         default="data/processed/hagi_ambisonic_fold0")
    parser.add_argument("--seq-len",     default=30,  type=int,
                        help="Past head frames per window (default 30 = 1s at 30Hz)")
    parser.add_argument("--audio-seq-len", default=10, type=int,
                        help="Future audio direction frames (default 10 = 0.33s)")
    parser.add_argument("--fps",         default=30,  type=int)
    parser.add_argument("--window-sec",  default=0.5, type=float,
                        help="RMS window for ambisonic extraction in seconds")
    parser.add_argument("--tolerance",   default=0.05, type=float)
    parser.add_argument("--fixation-only", action="store_true", default=True,
                        help="Only create samples where target is a fixation")
    parser.add_argument("--all-samples", dest="fixation_only", action="store_false",
                        help="Include saccade targets too")
    parser.add_argument("--max-train-samples", default=1000000, type=int)
    parser.add_argument("--max-test-samples",  default=200000,  type=int)
    parser.add_argument("--tiny",        action="store_true")
    parser.add_argument("--max-videos",  default=2, type=int)
    parser.add_argument("--max-users",   default=2, type=int)
    args = parser.parse_args()

    root       = Path(args.root)
    audio_root = Path(args.audio_root)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Head window:        {args.seq_len} frames ({args.seq_len/args.fps:.2f}s)")
    print(f"Audio future window:{args.audio_seq_len} frames ({args.audio_seq_len/args.fps:.2f}s)")
    print(f"Feature dim:        {args.seq_len*3 + args.audio_seq_len*3}")
    print(f"Fixation only:      {args.fixation_only}")

    # load splits
    split_dir    = Path(args.splits)
    train_videos = load_split_videos(split_dir / f"train_{args.fold}.txt")
    test_videos  = load_split_videos(split_dir / f"test_{args.fold}.txt")
    print(f"Train videos: {len(train_videos)} | Test videos: {len(test_videos)}")

    # load CSVs
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

    # clean gaze
    print("\nCleaning gaze data...")
    before = len(gaze)
    gaze = gaze[
        (gaze["t"] >= 0)
        & (gaze["u"].between(0, 1))
        & (gaze["v"].between(0, 1))
    ].copy()
    gaze["fixation"] = gaze["fixation"].astype(bool).astype(np.int64)
    after = len(gaze)
    print(f"  Before: {before:,} | After: {after:,} | Removed: {before - after:,}")
    print(f"  Fixation distribution:\n{gaze['fixation'].value_counts().to_string()}")

    # tiny mode
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

    # synchronize
    print("\nSynchronizing gaze and head timestamps...")
    merged = synchronize_gaze_head(gaze, head, tolerance=args.tolerance)
    print(f"  Merged samples: {len(merged):,}")

    train_df = merged[merged["video"].isin(train_videos)].copy()
    test_df  = merged[merged["video"].isin(test_videos)].copy()
    print(f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    # create windows
    print("\nCreating train windows...")
    trainX, trainY, trainFix, train_stats = create_windows(
        train_df,
        audio_root    = audio_root,
        seq_len       = args.seq_len,
        audio_seq_len = args.audio_seq_len,
        fps           = args.fps,
        window_sec    = args.window_sec,
        fixation_only = args.fixation_only,
        max_samples   = args.max_train_samples,
    )

    print("Creating test windows...")
    testX, testY, testFix, test_stats = create_windows(
        test_df,
        audio_root    = audio_root,
        seq_len       = args.seq_len,
        audio_seq_len = args.audio_seq_len,
        fps           = args.fps,
        window_sec    = args.window_sec,
        fixation_only = args.fixation_only,
        max_samples   = args.max_test_samples,
    )

    # report
    print("\n=== OUTPUT SHAPES ===")
    print(f"  trainX:  {trainX.shape}  (head_window + ambisonic_direction)")
    print(f"  trainY:  {trainY.shape}  (gaze xyz unit vector)")
    print(f"  testX:   {testX.shape}")
    print(f"  testY:   {testY.shape}")
    print(f"  testFix: {testFix.shape} (1=fixation, 0=saccade)")

    print("\n=== STATS ===")
    print(f"  Train: {train_stats}")
    print(f"  Test:  {test_stats}")

    # save
    np.save(out_dir / "trainingX.npy",        trainX)
    np.save(out_dir / "trainingY.npy",        trainY)
    np.save(out_dir / "trainingFixation.npy", trainFix)
    np.save(out_dir / "testX.npy",            testX)
    np.save(out_dir / "testY.npy",            testY)
    np.save(out_dir / "testFixation.npy",     testFix)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
