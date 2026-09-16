"""Mask decoding and geometry helpers independent of PyNeat."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

MASK_STRIDE = 4


def as_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    values = np.asarray(mask)
    cutoff = threshold * 255.0 if values.dtype == np.uint8 else threshold
    return values > cutoff


def area(mask: np.ndarray, threshold: float = 0.5) -> int:
    return int(np.count_nonzero(as_binary(mask, threshold)))


def centroid(mask: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    binary = as_binary(mask, threshold)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def iou(left: np.ndarray, right: np.ndarray, threshold: float = 0.5) -> float:
    a = as_binary(left, threshold)
    b = as_binary(right, threshold)
    if a.shape != b.shape:
        b = cv2.resize(
            b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    union = np.logical_or(a, b)
    denom = int(union.sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / denom)


def bbox_from_mask(mask: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(as_binary(mask, threshold))
    if xs.size == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def project_letterbox_mask(
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    frame_size: tuple[int, int],
    model_size: tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Project a full letterboxed mask head into a frame-sized binary mask."""
    frame_w, frame_h = frame_size
    model_w, model_h = model_size
    source = np.asarray(mask)
    mask_h, mask_w = source.shape[:2]
    scale = min(model_w / frame_w, model_h / frame_h)
    pad_x = (model_w - frame_w * scale) / 2.0
    pad_y = (model_h - frame_h * scale) / 2.0
    x1, y1, x2, y2 = bbox

    mx0 = max(0, min(mask_w - 1, int(np.floor((x1 * scale + pad_x) * mask_w / model_w))))
    my0 = max(0, min(mask_h - 1, int(np.floor((y1 * scale + pad_y) * mask_h / model_h))))
    mx1 = max(mx0 + 1, min(mask_w, int(np.ceil((x2 * scale + pad_x) * mask_w / model_w))))
    my1 = max(my0 + 1, min(mask_h, int(np.ceil((y2 * scale + pad_y) * mask_h / model_h))))

    fx0 = max(0, min(frame_w - 1, int(np.floor(x1))))
    fy0 = max(0, min(frame_h - 1, int(np.floor(y1))))
    fx1 = max(fx0 + 1, min(frame_w, int(np.ceil(x2))))
    fy1 = max(fy0 + 1, min(frame_h, int(np.ceil(y2))))
    roi = cv2.resize(source[my0:my1, mx0:mx1], (fx1 - fx0, fy1 - fy0), cv2.INTER_LINEAR)
    full = np.zeros((frame_h, frame_w), dtype=np.uint8)
    full[fy0:fy1, fx0:fx1] = roi
    return full


def jpeg_crop(
    frame: np.ndarray,
    mask: np.ndarray,
    margin: float = 0.12,
    quality: int = 88,
) -> bytes:
    x, y, width, height = bbox_from_mask(mask)
    if width <= 0 or height <= 0:
        raise ValueError("cannot crop an empty mask")
    extra_x, extra_y = int(width * margin), int(height * margin)
    x0, y0 = max(0, x - extra_x), max(0, y - extra_y)
    x1 = min(frame.shape[1], x + width + extra_x)
    y1 = min(frame.shape[0], y + height + extra_y)
    ok, encoded = cv2.imencode(
        ".jpg", np.ascontiguousarray(frame[y0:y1, x0:x1]), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("failed to encode JPEG crop")
    return encoded.tobytes()


def point_distance(mask: np.ndarray, point: tuple[float, float]) -> float:
    """Distance in pixels from a point to the nearest positive mask pixel."""
    binary = as_binary(mask)
    if not binary.any():
        return float("inf")
    x, y = point
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < binary.shape[0] and 0 <= xi < binary.shape[1] and binary[yi, xi]:
        return 0.0
    ys, xs = np.nonzero(binary)
    return float(np.sqrt(np.min((xs - x) ** 2 + (ys - y) ** 2)))


def encode_uncompressed_rle(mask: np.ndarray) -> list[int]:
    flat = as_binary(mask).flatten(order="F")
    counts: list[int] = []
    current = False
    run = 0
    for value in flat:
        bit = bool(value)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = bit
    counts.append(run)
    return counts
