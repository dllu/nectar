"""Small color and display helpers shared by Nectar's Python tools."""

from __future__ import annotations

import numpy as np


RAW_COLOR_CORRECTION = np.array(
    [
        [0.9, -0.3, -0.3],
        [-0.8, 1.5, -0.3],
        [-0.5, -0.5, 1.7],
    ],
    dtype=np.float32,
)
COLOR_TRANSFORM_BLOCK_ROWS = 64


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    adjusted = matrix.astype(np.float32).copy()
    deficit = 1.0 - adjusted.sum(axis=1)
    adjusted[np.arange(adjusted.shape[0]), np.arange(adjusted.shape[0])] += deficit
    return adjusted


COLOR_CORRECTION = _normalize_rows(RAW_COLOR_CORRECTION)


def apply_color_transform(
    raw_rgb: np.ndarray,
    white_balance: np.ndarray,
    color_matrix: np.ndarray = COLOR_CORRECTION,
) -> np.ndarray:
    """Apply per-channel white balance followed by a 3x3 color matrix."""
    raw_rgb = np.asarray(raw_rgb)
    if raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3:
        raise ValueError("raw_rgb must be an HxWx3 array")
    wb = np.asarray(white_balance, dtype=np.float32).reshape(1, 1, 3)
    color_matrix = np.asarray(color_matrix)
    output_dtype = np.result_type(raw_rgb.dtype, wb.dtype, color_matrix.dtype)
    transformed = np.empty(raw_rgb.shape, dtype=output_dtype)
    for row_start in range(0, raw_rgb.shape[0], COLOR_TRANSFORM_BLOCK_ROWS):
        row_stop = min(raw_rgb.shape[0], row_start + COLOR_TRANSFORM_BLOCK_ROWS)
        balanced = raw_rgb[row_start:row_stop] * wb
        transformed[row_start:row_stop] = np.tensordot(
            balanced,
            color_matrix.T,
            axes=([2], [0]),
        )
    return transformed


def auto_tone_map_to_uint8(
    rgb: np.ndarray,
    *,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    gamma: float = 2.2,
) -> np.ndarray:
    """Apply a simple display-only percentile stretch and gamma curve."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("automatic tone mapping expects an HxWx3 uint8 image")
    black_point, white_point = np.percentile(
        rgb,
        (low_percentile, high_percentile),
    )
    if white_point <= black_point:
        return rgb.copy()
    lut = np.arange(256, dtype=np.float32)
    lut -= black_point
    lut /= white_point - black_point
    np.clip(lut, 0.0, 1.0, out=lut)
    np.power(lut, 1.0 / gamma, out=lut)
    return np.rint(lut * 255.0).astype(np.uint8)[rgb]
