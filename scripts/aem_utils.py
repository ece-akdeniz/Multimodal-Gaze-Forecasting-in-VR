from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def resolve_aem_root(aem_root: Path) -> Path:
    if (aem_root / "dataset" / "AEM").exists():
        return aem_root / "dataset" / "AEM"
    return aem_root


def build_aem_index(aem_root: Path):
    aem_root = resolve_aem_root(aem_root)
    aem_index = {}

    for video_dir in aem_root.iterdir():
        if not video_dir.is_dir() or not video_dir.name.isdigit():
            continue

        video_id = int(video_dir.name)
        frames = []

        for file in video_dir.glob("*.png"):
            try:
                if "_" in file.stem:
                    frame = int(file.stem.split("_")[1])
                else:
                    frame = int(file.stem)
                frames.append(frame)
            except ValueError:
                continue

        if frames:
            aem_index[video_id] = np.array(sorted(frames), dtype=np.int32)

    return aem_index, aem_root


def peak_aem_uv_from_array(arr: np.ndarray, intensity_threshold: float = 1e-3):
    flat = arr.reshape(-1)
    mask = flat > intensity_threshold

    if not np.any(mask):
        return None

    idx = int(np.argmax(np.where(mask, flat, -1.0)))
    height, width = arr.shape
    y, x = divmod(idx, width)

    u = (x + 0.5) / width
    v = (y + 0.5) / height
    return float(u), float(v)


def centroid_aem_uv_from_array(arr: np.ndarray, intensity_threshold: float = 1e-3):
    mask = arr > intensity_threshold
    if not np.any(mask):
        return None

    height, width = arr.shape
    ys, xs = np.nonzero(mask)
    weights = arr[ys, xs].astype(np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return None

    u = float(np.sum((xs.astype(np.float64) + 0.5) * weights) / weight_sum / width)
    v = float(np.sum((ys.astype(np.float64) + 0.5) * weights) / weight_sum / height)
    return u, v


def aem_peak_from_array(
    arr: np.ndarray,
    peak_method: str = "argmax",
    intensity_threshold: float = 1e-3,
):
    if peak_method == "argmax":
        peak = peak_aem_uv_from_array(arr, intensity_threshold=intensity_threshold)
    elif peak_method == "centroid":
        peak = centroid_aem_uv_from_array(arr, intensity_threshold=intensity_threshold)
    else:
        raise ValueError(f"Unknown peak_method: {peak_method}")

    if peak is None:
        return None, 0.0

    u, v = peak
    active = arr[arr > intensity_threshold]
    confidence = float(active.max()) if active.size > 0 else 0.0
    return (u, v), confidence


def peak_aem_uv(aem_path: Path, intensity_threshold: float = 1e-3):
    img = Image.open(aem_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return peak_aem_uv_from_array(arr, intensity_threshold=intensity_threshold)


def nearest_aem_frame(available_frames: np.ndarray, target_frame: int, max_frame_distance: int = 8):
    if available_frames.size == 0:
        return None

    nearest_idx = int(np.abs(available_frames - target_frame).argmin())
    nearest_frame = int(available_frames[nearest_idx])

    if abs(nearest_frame - target_frame) > max_frame_distance:
        return None

    return nearest_frame


def aem_file_path(aem_root: Path, video: int, frame: int) -> Path:
    video_dir = aem_root / f"{video:04d}"
    nested = video_dir / f"{video:04d}_{frame:04d}.png"
    if nested.exists():
        return nested
    return video_dir / f"{frame:04d}.png"


def load_aem_peak(
    aem_root: Path,
    aem_index,
    video: int,
    frame: int,
    peak_method: str = "argmax",
    max_frame_distance: int = 8,
    intensity_threshold: float = 1e-3,
    cache: Optional[dict] = None,
) -> Tuple[Optional[Tuple[float, float]], float]:
    video = int(video)
    frame = int(frame)

    if video not in aem_index:
        return None, 0.0

    nearest_frame = nearest_aem_frame(
        aem_index[video],
        frame,
        max_frame_distance=max_frame_distance,
    )
    if nearest_frame is None:
        return None, 0.0

    aem_file = aem_file_path(aem_root, video, nearest_frame)
    if not aem_file.exists():
        return None, 0.0

    if cache is not None:
        if aem_file not in cache:
            img = Image.open(aem_file).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            cache[aem_file] = aem_peak_from_array(
                arr,
                peak_method=peak_method,
                intensity_threshold=intensity_threshold,
            )
        return cache[aem_file]

    img = Image.open(aem_file).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return aem_peak_from_array(
        arr,
        peak_method=peak_method,
        intensity_threshold=intensity_threshold,
    )


def load_peak_aem_uv(
    aem_root: Path,
    aem_index,
    video: int,
    frame: int,
    max_frame_distance: int = 8,
    intensity_threshold: float = 1e-3,
    cache: Optional[dict] = None,
):
    peak, _ = load_aem_peak(
        aem_root=aem_root,
        aem_index=aem_index,
        video=video,
        frame=frame,
        peak_method="argmax",
        max_frame_distance=max_frame_distance,
        intensity_threshold=intensity_threshold,
        cache=cache,
    )
    return peak


def horizon_to_frame_offset(horizon_sec: float, video_fps: int = 60):
    return int(round(horizon_sec * video_fps))
