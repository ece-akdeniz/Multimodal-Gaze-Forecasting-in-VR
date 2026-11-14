from pathlib import Path
import argparse
import json
import sys
import wave

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aem_utils import (
    build_aem_index,
    horizon_to_frame_offset,
    load_aem_peak,
)


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


def find_wav(audio_root: Path, video_id: int):
    vid = f"{video_id:04d}"
    matches = list(audio_root.glob(f"{vid}/{vid}/{vid}.wav"))
    if matches:
        return matches[0]
    matches = list(audio_root.glob(f"**/{vid}.wav"))
    return matches[0] if matches else None


def load_audio_features(wav_path: Path, fps=30, window_sec=0.5):
    with wave.open(str(wav_path), "rb") as w:
        channels = w.getnchannels()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    audio = audio.reshape(-1, channels) / 32768.0

    duration = len(audio) / sr
    n_steps = int(np.ceil(duration * fps))
    win = int(window_sec * sr)

    feats = np.zeros((n_steps, channels), dtype=np.float32)

    for i in range(n_steps):
        center = int(i / fps * sr)
        start = max(0, center)
        end = min(len(audio), center + win)

        segment = audio[start:end]
        if len(segment) == 0:
            continue

        rms = np.sqrt(np.mean(segment ** 2, axis=0))
        feats[i] = rms.astype(np.float32)

    return feats


def synchronize_gaze_head(gaze, head, tolerance=0.05):
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
        raise RuntimeError("No gaze/head samples matched.")

    merged = pd.concat(merged_parts, ignore_index=True)
    return merged.sort_values(["video", "id", "gaze_t"]).reset_index(drop=True)


def uv_to_xyz(u, v):
    lon = (u - 0.5) * 2.0 * np.pi
    lat = (0.5 - v) * np.pi

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    return np.stack([x, y, z], axis=-1).astype(np.float32)


def audio_feature_dim(audio_mode, audio_horizons, use_xyz, include_aem_validity=False):
    if audio_mode == "rms":
        return 4

    per_horizon = 3 if use_xyz else 2
    coord_dim = len(audio_horizons) * per_horizon
    if include_aem_validity:
        return coord_dim + len(audio_horizons)
    return coord_dim


def extract_peak_aem_features(
    video,
    gaze_frame,
    aem_root,
    aem_index,
    audio_horizons,
    video_fps,
    max_frame_distance,
    use_xyz,
    peak_cache,
    peak_method="argmax",
    aem_mask_mode="drop",
    include_aem_validity=False,
):
    coord_parts = []
    validity_parts = []
    all_valid = True

    for horizon in audio_horizons:
        target_frame = int(gaze_frame) + horizon_to_frame_offset(horizon, video_fps=video_fps)
        peak, confidence = load_aem_peak(
            aem_root=aem_root,
            aem_index=aem_index,
            video=video,
            frame=target_frame,
            peak_method=peak_method,
            max_frame_distance=max_frame_distance,
            cache=peak_cache,
        )

        if peak is None:
            all_valid = False
            if aem_mask_mode == "drop":
                return None
            u, v = 0.0, 0.0
            validity = 0.0
        else:
            u, v = peak
            validity = float(confidence)

        if use_xyz:
            coord_parts.append(uv_to_xyz(np.array([u], dtype=np.float32), np.array([v], dtype=np.float32))[0])
        else:
            coord_parts.append(np.array([u, v], dtype=np.float32))

        if include_aem_validity:
            validity_parts.append(np.array([validity], dtype=np.float32))

    coords = np.concatenate(coord_parts).astype(np.float32)
    if include_aem_validity:
        validity = np.concatenate(validity_parts).astype(np.float32)
        return coords, validity, all_valid

    if not all_valid:
        return None
    return coords


def create_windows(
    df,
    seq_len=10,
    fps=30,
    max_samples=None,
    audio_mode="peak_aem",
    audio_root=None,
    aem_root=None,
    aem_index=None,
    audio_horizons=None,
    audio_window_sec=0.5,
    video_fps=60,
    max_frame_distance=8,
    use_xyz=True,
    peak_method="argmax",
    aem_mask_mode="drop",
    include_aem_validity=False,
):
    if audio_horizons is None:
        audio_horizons = [0.5, 1.0, 2.0]

    xs = []
    ys = []
    fixation_labels = []
    missing_audio = 0
    partial_aem = 0
    audio_cache = {}
    peak_cache = {}

    stop = False
    audio_dim = audio_feature_dim(
        audio_mode,
        audio_horizons,
        use_xyz,
        include_aem_validity=include_aem_validity,
    )
    feature_dim = seq_len * 3 + audio_dim

    for (video, pid), group in df.groupby(["video", "id"]):
        if stop:
            break

        video = int(video)

        if audio_mode == "rms":
            if video not in audio_cache:
                wav_path = find_wav(audio_root, video)
                audio_cache[video] = (
                    load_audio_features(wav_path, fps=fps, window_sec=audio_window_sec)
                    if wav_path is not None
                    else None
                )

            if audio_cache[video] is None:
                missing_audio += len(group)
                continue

        elif audio_mode == "peak_aem":
            if video not in aem_index and aem_mask_mode == "drop":
                missing_audio += len(group)
                continue
        else:
            raise ValueError(f"Unknown audio_mode: {audio_mode}")

        group = group.sort_values("gaze_t").reset_index(drop=True)

        head_uv = group[["head_u", "head_v"]].to_numpy(dtype=np.float32)
        gaze_uv = group[["gaze_u", "gaze_v"]].to_numpy(dtype=np.float32)
        fix = group["fixation"].to_numpy(dtype=np.int64)
        frames = group["frame"].to_numpy(dtype=np.int32)

        head_xyz = uv_to_xyz(head_uv[:, 0], head_uv[:, 1])
        gaze_xyz = uv_to_xyz(gaze_uv[:, 0], gaze_uv[:, 1])

        for i in range(seq_len, len(group)):
            head_window = head_xyz[i - seq_len:i].reshape(-1)

            if audio_mode == "rms":
                times = group["gaze_t"].to_numpy(dtype=np.float32)
                audio_idx = int(round(times[i] * fps))
                audio_feats = audio_cache[video]

                if audio_idx < 0 or audio_idx >= len(audio_feats):
                    missing_audio += 1
                    continue

                audio_feature = audio_feats[audio_idx]
            else:
                audio_result = extract_peak_aem_features(
                    video=video,
                    gaze_frame=int(frames[i]),
                    aem_root=aem_root,
                    aem_index=aem_index,
                    audio_horizons=audio_horizons,
                    video_fps=video_fps,
                    max_frame_distance=max_frame_distance,
                    use_xyz=use_xyz,
                    peak_cache=peak_cache,
                    peak_method=peak_method,
                    aem_mask_mode=aem_mask_mode,
                    include_aem_validity=include_aem_validity,
                )

                if audio_result is None:
                    missing_audio += 1
                    continue

                if include_aem_validity:
                    audio_coords, audio_validity, all_valid = audio_result
                    if not all_valid:
                        partial_aem += 1
                    audio_feature = np.concatenate([audio_coords, audio_validity], axis=0)
                else:
                    audio_feature = audio_result

            x = np.concatenate([head_window, audio_feature], axis=0).astype(np.float32)
            y = gaze_xyz[i].astype(np.float32)

            xs.append(x)
            ys.append(y)
            fixation_labels.append(fix[i])

            if max_samples is not None and len(xs) >= max_samples:
                stop = True
                break

    audio_coord_dim = audio_feature_dim(audio_mode, audio_horizons, use_xyz, include_aem_validity=False)
    audio_validity_dim = len(audio_horizons) if include_aem_validity else 0

    stats = {
        "samples": len(xs),
        "missing_audio": missing_audio,
        "partial_aem": partial_aem,
        "feature_dim": feature_dim,
        "audio_mode": audio_mode,
        "audio_horizons": list(audio_horizons),
        "use_xyz": use_xyz,
        "peak_method": peak_method,
        "aem_mask_mode": aem_mask_mode,
        "include_aem_validity": include_aem_validity,
        "audio_coord_dim": audio_coord_dim,
        "audio_validity_dim": audio_validity_dim,
    }

    if not xs:
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            stats,
        )

    return (
        np.stack(xs),
        np.stack(ys),
        np.array(fixation_labels, dtype=np.int64),
        stats,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/D-SAV360")
    parser.add_argument("--audio-root", default="data/raw/D-SAV360/videos_extracted")
    parser.add_argument("--aem-root", default="data/raw/D-SAV360/audio_energy_maps")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--out", default="data/processed/hagi_audio_peak_fold0")
    parser.add_argument("--seq-len", default=10, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--video-fps", default=60, type=int)
    parser.add_argument("--audio-mode", default="peak_aem", choices=["peak_aem", "rms"])
    parser.add_argument("--audio-horizons", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--audio-window-sec", default=0.5, type=float)
    parser.add_argument("--max-frame-distance", default=8, type=int)
    parser.add_argument("--peak-method", default="argmax", choices=["argmax", "centroid"])
    parser.add_argument("--aem-mask-mode", default="drop", choices=["drop", "zero"])
    parser.add_argument("--include-aem-validity", action="store_true")
    parser.add_argument("--aem-use-xyz", action="store_true", default=True)
    parser.add_argument("--aem-use-uv", action="store_true")
    parser.add_argument("--tolerance", default=0.05, type=float)
    parser.add_argument("--max-train-samples", default=1000000, type=int)
    parser.add_argument("--max-test-samples", default=200000, type=int)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-videos", default=2, type=int)
    parser.add_argument("--max-users", default=2, type=int)
    args = parser.parse_args()

    use_xyz = not args.aem_use_uv
    include_aem_validity = args.include_aem_validity or args.aem_mask_mode == "zero"

    root = Path(args.root)
    audio_root = Path(args.audio_root)
    aem_root = Path(args.aem_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    aem_index = None
    if args.audio_mode == "peak_aem":
        if not aem_root.exists():
            raise FileNotFoundError(
                f"AEM directory not found: {aem_root}\n"
                "Download AEM.zip from https://zenodo.org/records/10043919 "
                "and extract to data/raw/D-SAV360/audio_energy_maps/"
            )
        aem_index, aem_root = build_aem_index(aem_root)
        if not aem_index:
            raise RuntimeError(f"No AEM PNG files found under {aem_root}")
        print("AEM root:", aem_root)
        print("AEM videos indexed:", len(aem_index))

    split_dir = Path(args.splits)
    train_videos = load_split_videos(split_dir / f"train_{args.fold}.txt")
    test_videos = load_split_videos(split_dir / f"test_{args.fold}.txt")

    print("Loading gaze...")
    gaze = load_all_csvs(
        root / "gaze",
        ["video", "stereo", "frame", "t", "u", "v", "id", "fixation", "valid_fix_classification"],
    )

    print("Loading head...")
    head = load_all_csvs(
        root / "head",
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

    window_kwargs = dict(
        seq_len=args.seq_len,
        fps=args.fps,
        audio_mode=args.audio_mode,
        audio_root=audio_root,
        aem_root=aem_root,
        aem_index=aem_index,
        audio_horizons=args.audio_horizons,
        audio_window_sec=args.audio_window_sec,
        video_fps=args.video_fps,
        max_frame_distance=args.max_frame_distance,
        use_xyz=use_xyz,
        peak_method=args.peak_method,
        aem_mask_mode=args.aem_mask_mode,
        include_aem_validity=include_aem_validity,
    )

    print("Creating train windows with audio...")
    trainingX, trainingY, trainingFixation, train_stats = create_windows(
        train_df,
        max_samples=args.max_train_samples,
        **window_kwargs,
    )

    print("Creating test windows with audio...")
    testX, testY, testFixation, test_stats = create_windows(
        test_df,
        max_samples=args.max_test_samples,
        **window_kwargs,
    )

    print("\n=== OUTPUT SHAPES ===")
    print("trainingX:", trainingX.shape)
    print("trainingY:", trainingY.shape)
    print("testX:", testX.shape)
    print("testY:", testY.shape)
    print("testFixation:", testFixation.shape)
    print("\n=== AUDIO STATS ===")
    print("Train:", train_stats)
    print("Test:", test_stats)

    np.save(out_dir / "trainingX.npy", trainingX)
    np.save(out_dir / "trainingY.npy", trainingY)
    np.save(out_dir / "trainingFixation.npy", trainingFixation)
    np.save(out_dir / "testX.npy", testX)
    np.save(out_dir / "testY.npy", testY)
    np.save(out_dir / "testFixation.npy", testFixation)

    metadata = {
        "seq_len": args.seq_len,
        "head_dim": 3,
        "audio_mode": args.audio_mode,
        "audio_horizons": args.audio_horizons,
        "use_xyz": use_xyz,
        "peak_method": args.peak_method,
        "aem_mask_mode": args.aem_mask_mode,
        "include_aem_validity": include_aem_validity,
        "audio_coord_dim": train_stats["audio_coord_dim"],
        "audio_validity_dim": train_stats["audio_validity_dim"],
        "feature_dim": train_stats["feature_dim"],
        "train_stats": train_stats,
        "test_stats": test_stats,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
