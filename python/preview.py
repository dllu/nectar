#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import time
from collections import OrderedDict
from pathlib import Path

import bottleneck as bn
import cv2
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
import itertools
import tifffile
from concurrent.futures import ThreadPoolExecutor, as_completed

# Color calibration components for RGB conversion
RAW_COLOR_CORRECTION = np.array(
    [
        [0.9, -0.3, -0.3],
        [-0.8, 1.5, -0.3],
        [-0.5, -0.5, 1.7],
    ],
    dtype=np.float32,
)

CAMERA_MAKE = "Alkeria"
CAMERA_MODEL_NAME = "Alkeria Necta N4K2-7C"
ENABLE_COLUMN_JITTER_CORRECTION = True
FORCE_CONTIGUOUS_SELECTION = True
FLIP_OUTPUT_180 = True
STRUCTURE_MASK_DOWNSAMPLE = 8
STRUCTURE_MASK_THRESHOLD = 0.05
STRUCTURE_MASK_DILATE_PX = 0
ENABLE_JITTER_DEBUG_PLOT = False
JITTER_DEBUG_ROW = 2000
JITTER_VERTICAL_BANDS = 256
JITTER_FINE_TEMPORAL_WINDOW = 31
JITTER_TEMPORAL_WINDOW = 1023
JITTER_BLOCK_COLUMNS = 4096
JITTER_MAX_CLIPPED_FRACTION = 0.10
JITTER_BLACK_MARGIN = 64.0
JITTER_WHITE_MARGIN_FRACTION = 0.03
JITTER_WHITE_MARGIN_MIN = 256.0
JITTER_FINE_MAX_CORRECTION_FRACTION = 0.02
JITTER_BROAD_MAX_CORRECTION_FRACTION = 0.01
SPEED_MAX_FIT_DYNAMIC_RANGE = 1.5
JITTER_RED = 0
JITTER_GREEN1 = 1
JITTER_GREEN2 = 2
JITTER_BLUE = 3


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    deficit = 1.0 - row_sums
    adjusted = matrix.astype(np.float32).copy()
    diag = np.arange(adjusted.shape[0])
    adjusted[diag, diag] += deficit[:, 0]
    return adjusted


COLOR_CORRECTION = _normalize_rows(RAW_COLOR_CORRECTION)

# DNG ColorMatrix values map CIE XYZ to camera-native coordinates, whereas
# COLOR_CORRECTION maps white-balanced camera coordinates to linear sRGB.
# These standard matrices bridge the two conventions. The D50 matrix is the
# ICC-adapted linear-sRGB matrix used for DNG's profile connection space.
XYZ_D65_TO_LINEAR_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

LINEAR_SRGB_TO_XYZ_D50 = np.array(
    [
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733],
    ],
    dtype=np.float64,
)

DNG_COLOR_MATRIX_1 = np.linalg.solve(
    COLOR_CORRECTION.astype(np.float64),
    XYZ_D65_TO_LINEAR_SRGB,
)
DNG_FORWARD_MATRIX_1 = (
    LINEAR_SRGB_TO_XYZ_D50 @ COLOR_CORRECTION.astype(np.float64)
)


def apply_color_transform(
    raw_rgb: np.ndarray,
    white_balance: np.ndarray,
    color_matrix: np.ndarray = COLOR_CORRECTION,
) -> np.ndarray:
    """Apply per-channel white balance followed by the 3x3 color matrix."""
    wb = np.asarray(white_balance, dtype=np.float32).reshape(1, 1, 3)
    balanced = raw_rgb * wb
    return np.tensordot(balanced, color_matrix.T, axes=([2], [0]))


def _rational_values(
    values: np.ndarray, *, signed: bool, denominator: int = 1_000_000
) -> tuple[tuple[int, int], ...]:
    values = np.asarray(values, dtype=np.float64).ravel()
    if not signed and np.any(values < 0):
        raise ValueError("unsigned rational values must be non-negative")
    return tuple((int(round(value * denominator)), denominator) for value in values)


def write_linear_dng(
    path: Path, raw_rgb: np.ndarray, white_balance: np.ndarray
) -> None:
    if tifffile is None:
        raise RuntimeError(
            "tifffile is required for DNG export. Install it via `uv pip install tifffile`."
        )

    flipped = raw_rgb[::-1, :, :]
    if flipped.dtype == np.uint16:
        linear = np.ascontiguousarray(flipped)
    else:
        linear = np.clip(flipped, 0, 65535).astype(np.uint16)
    wb = np.asarray(white_balance, dtype=np.float64)
    safe_wb = np.clip(wb, 1e-6, None)
    as_shot_neutral = 1.0 / safe_wb
    if as_shot_neutral[1] != 0:
        as_shot_neutral /= as_shot_neutral[1]

    extratags = [
        (271, "s", len(CAMERA_MAKE) + 1, CAMERA_MAKE),  # Make
        (272, "s", len(CAMERA_MODEL_NAME) + 1, CAMERA_MODEL_NAME),  # Model
        (274, "H", 1, 1),  # Orientation = top-left
        (50706, "B", 4, (1, 4, 0, 0)),  # DNGVersion
        (50707, "B", 4, (1, 2, 0, 0)),  # DNGBackwardVersion
        (
            50708,
            "s",
            len(CAMERA_MODEL_NAME) + 1,
            CAMERA_MODEL_NAME,
        ),  # UniqueCameraModel
        (
            50721,
            "2i",
            9,
            _rational_values(DNG_COLOR_MATRIX_1, signed=True),
        ),  # ColorMatrix1: XYZ D65 -> camera
        (
            50728,
            "2I",
            3,
            _rational_values(as_shot_neutral, signed=False),
        ),  # AsShotNeutral
        (50778, "H", 1, 21),  # CalibrationIlluminant1 = D65
        (
            50964,
            "2i",
            9,
            _rational_values(DNG_FORWARD_MATRIX_1, signed=True),
        ),  # ForwardMatrix1: white-balanced camera -> XYZ D50
    ]

    tifffile.imwrite(
        path,
        linear,
        photometric=tifffile.PHOTOMETRIC.LINEAR_RAW,
        planarconfig="contig",
        metadata=None,
        software="nectar-preview",
        extratags=extratags,
    )


# linescans = Path("/home/dllu/pictures/linescan")
linescans = Path("/mnt/dataz/pictures/linescan")


def format_contiguous_segments(indices):
    if not indices:
        return "none"

    segments = []
    start = indices[0]
    prev = start
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev))
        start = idx
        prev = idx
    segments.append((start, prev))

    return ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in segments
    )


class FileDataCache:
    def __init__(
        self,
        bin_paths,
        dtype,
        rows,
        max_cache_bytes: int = 512 * 1024 * 1024,
    ):
        self.bin_paths = bin_paths
        self.dtype = dtype
        self.rows = rows
        self.max_cache_bytes = max(1, max_cache_bytes)
        self.cached_bytes = 0
        # determine columns per file
        self.file_cols = []
        for p in bin_paths:
            size = p.stat().st_size
            n_elems = size // np.dtype(dtype).itemsize
            if n_elems % rows != 0:
                raise ValueError(f"File {p} size not multiple of rows")
            cols = n_elems // rows
            self.file_cols.append(cols)
        # cumulative column offsets
        self.cum_cols = np.concatenate(([0], np.cumsum(self.file_cols)))
        self.total_cols = int(self.cum_cols[-1])
        self.height = rows
        self.cache = OrderedDict()

    @property
    def shape(self):
        return (self.height, self.total_cols)

    def _load_file(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
        else:
            # load and reshape file idx
            data1d = np.fromfile(self.bin_paths[idx], dtype=self.dtype).astype(
                np.float32
            )
            # reshape to (rows, cols) then transpose to (height, cols)
            data2d = data1d.reshape(-1, self.height).T
            self.cache[idx] = data2d
            self.cached_bytes += data2d.nbytes
            while self.cached_bytes > self.max_cache_bytes and len(self.cache) > 1:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= evicted.nbytes
        return self.cache[idx]

    def __getitem__(self, key):
        # Expect key as (rows, cols)
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Invalid index for FileDataCache")
        row_idx, col_idx = key
        # handle integer column indexing
        if isinstance(col_idx, int):
            ci = col_idx if col_idx >= 0 else self.total_cols + col_idx
            # find file containing this column
            file_idx = np.searchsorted(self.cum_cols, ci, side="right") - 1
            local = ci - self.cum_cols[file_idx]
            arr_file = self._load_file(file_idx)
            return arr_file[row_idx, local]
        # handle slice of columns
        if isinstance(col_idx, slice):
            start = (
                0
                if col_idx.start is None
                else (
                    col_idx.start
                    if col_idx.start >= 0
                    else self.total_cols + col_idx.start
                )
            )
            stop = (
                self.total_cols
                if col_idx.stop is None
                else (
                    col_idx.stop
                    if col_idx.stop >= 0
                    else self.total_cols + col_idx.stop
                )
            )
            step = 1 if col_idx.step is None else col_idx.step
            # collect blocks from each file
            blocks = []
            for i, cols_i in enumerate(self.file_cols):
                fstart = self.cum_cols[i]
                fend = self.cum_cols[i + 1]
                if fend <= start or fstart >= stop:
                    continue
                lo = max(start, fstart) - fstart
                hi = min(stop, fend) - fstart
                arr_file = self._load_file(i)
                blocks.append(arr_file[row_idx, lo:hi])
            if not blocks:
                empty = np.empty((self.height, 0), dtype=np.float32)
                return empty[row_idx, :]
            arr = np.concatenate(blocks, axis=-1)
            # apply column step
            if step != 1:
                arr = arr[..., ::step]
            return arr
        # unsupported indexing
        raise IndexError("Unsupported index type for FileDataCache")


class ChannelView:
    """
    Lazy view of a single Bayer channel from FileDataCache.
    """

    def __init__(self, cache, row_offset, col_offset, row_step=2, col_step=2):
        self.cache = cache
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.row_step = row_step
        self.col_step = col_step
        # compute channel dimensions
        total_cols = cache.total_cols
        self.height = (cache.height - row_offset + row_step - 1) // row_step
        self.width = (total_cols - col_offset + col_step - 1) // col_step

    @property
    def shape(self):
        """Shape of this channel view as (rows, cols)."""
        return (self.height, self.width)

    def __getitem__(self, key):
        # key: (row_idx, col_idx)
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("ChannelView requires 2D indexing")
        row_key, col_key = key
        # map row_key to parent
        if isinstance(row_key, slice):
            start = 0 if row_key.start is None else row_key.start
            stop = self.height if row_key.stop is None else row_key.stop
            step = 1 if row_key.step is None else row_key.step
            abs_row = slice(
                self.row_offset + start * self.row_step,
                self.row_offset + stop * self.row_step,
                self.row_step * step,
            )
        else:
            # single index
            idx = row_key if row_key >= 0 else self.height + row_key
            abs_row = self.row_offset + idx * self.row_step
        # map col_key similarly
        if isinstance(col_key, slice):
            start = 0 if col_key.start is None else col_key.start
            stop = self.width if col_key.stop is None else col_key.stop
            step = 1 if col_key.step is None else col_key.step
            abs_col = slice(
                self.col_offset + start * self.col_step,
                self.col_offset + stop * self.col_step,
                self.col_step * step,
            )
        else:
            idx = col_key if col_key >= 0 else self.width + col_key
            abs_col = self.col_offset + idx * self.col_step
        # fetch from parent cache
        return self.cache[abs_row, abs_col]


def subpixel_peak(y, sigma, initial):
    peak = initial
    x = np.arange(len(y), dtype=np.float32)

    def split_gaussian(x_vals, mu, amp, sigma_l, sigma_r):
        sigma_vals = np.where(x_vals <= mu, sigma_l, sigma_r)
        return amp * np.exp(-0.5 * ((x_vals - mu) / sigma_vals) ** 2)

    window = int(max(3, round(3 * sigma)))
    left = max(0, initial - window)
    right = min(len(y) - 1, initial + window)
    x_local = x[left : right + 1]
    y_local = y[left : right + 1]

    fit_peak = None
    if y_local.size >= 5 and np.any(y_local > 0):
        amp0 = float(np.max(y_local))
        mu0 = float(initial)
        p0 = [mu0, amp0, max(0.5, sigma * 0.8), max(0.5, sigma * 1.2)]
        bounds = ([left, 0.0, 0.1, 0.1], [right, amp0 * 2.0, 20.0, 20.0])
        try:
            popt, _ = curve_fit(split_gaussian, x_local, y_local, p0=p0, bounds=bounds)
            fit_peak = float(popt[0])
        except Exception:
            fit_peak = None

    sigma2 = sigma**2
    for _ in range(10):
        weights = np.exp(-((x - peak) ** 2) / sigma2) * y
        denom = np.sum(weights)
        if denom == 0:
            break
        peak = np.sum(weights * x) / denom
    mean_shift_peak = float(peak)

    if fit_peak is not None:
        return fit_peak, mean_shift_peak
    return mean_shift_peak, mean_shift_peak


def windowed_cross_correlation(
    green_1: np.ndarray,
    green_2: np.ndarray,
    window_step: int = 256,
    window_size: int = 4096,
    corr_size: int = 15,
    output_dir: Path = None,
    jitter_models: ColumnJitterModel | None = None,
) -> np.ndarray:
    """
    green_1: (h, w) where w >> h and w >> window_size
    green_2: (h, w)
    """

    xs = np.arange(0, green_1.shape[1], window_step)
    ys = []
    refined_ys = []
    good_xs = []
    corr_half = corr_size // 2
    weights = []

    heatmap = np.zeros((len(xs), corr_size))

    def vertical_energy(block: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(block)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 0.1 * np.max(block)
        return grad_x / magnitude

    for xi, x in tqdm(
        enumerate(xs[1:-1]), desc="train speed estimation", total=len(xs)
    ):
        window_1 = green_1[:, x : x + window_step]
        if jitter_models is not None:
            window_1 = apply_column_models(
                window_1, jitter_models, x, JITTER_GREEN1
            )
        window_1 = vertical_energy(window_1)

        corrs = np.zeros((corr_size,))
        pad = corr_half
        window_2_base = green_2[:, x - pad : x + window_step + pad]
        if jitter_models is not None:
            window_2_base = apply_column_models(
                window_2_base,
                jitter_models,
                x - pad,
                JITTER_GREEN2,
            )
        window_2_base = vertical_energy(window_2_base)

        for corr_ind in range(corr_size):
            corr_x = corr_ind - corr_half
            start = corr_x + pad
            window_2 = window_2_base[:, start : start + window_step]

            # corr_value = np.sum(window_1 * window_2)
            error = np.sum(np.abs(window_1 - window_2))
            corrs[corr_ind] = -error

        peak_idx = np.argmax(corrs)

        if peak_idx <= 0 or peak_idx >= corr_size - 1:
            continue

        heatmap[xi, :] = corrs

        if xi >= window_size // window_step:
            ncorrs = np.sum(heatmap[xi - window_size // window_step : xi, :], axis=0)
            weights.append(np.max(ncorrs) - np.min(ncorrs))
            ncorrs = ncorrs - np.min(ncorrs)
            ncorrs = ncorrs / np.max(ncorrs)
            peak_idx = np.argmax(ncorrs)
            refined_peak_idx, mean_shift_peak_idx = subpixel_peak(ncorrs, 2.0, peak_idx)

            ys.append(peak_idx - corr_half)
            refined_ys.append(mean_shift_peak_idx - corr_half)
            good_xs.append(xi * window_step + window_size // 2)

            if output_dir is not None:
                plt.figure(figsize=(16, 9), dpi=300)
                plt.plot(
                    np.arange(-corr_half, corr_half + 1),
                    ncorrs,
                    "b.",
                    label="Raw peaks",
                )
                plt.plot(
                    [peak_idx - corr_half], [ncorrs[peak_idx]], "r.", label="Naive peak"
                )
                plt.plot(
                    [0, 0],
                    [0, np.max(ncorrs)],
                    "k--",
                    label="Zero shift",
                )
                plt.plot(
                    [refined_peak_idx - corr_half, refined_peak_idx - corr_half],
                    [0, np.max(ncorrs)],
                    "g--",
                    label="Curve-fit peak",
                )
                plt.plot(
                    [mean_shift_peak_idx - corr_half, mean_shift_peak_idx - corr_half],
                    [0, np.max(ncorrs)],
                    "c--",
                    label="Mean-shift peak",
                )
                plt.legend()
                plt.savefig(output_dir / f"mean_shift_{xi:04d}.png")
                plt.close()

    if output_dir is not None:
        plt.imshow(heatmap)
        plt.savefig(output_dir / "heatmap.png")
        plt.close()

        plt.figure(figsize=(16, 9), dpi=300)
        plt.plot(good_xs, ys, "b.", label="Raw peaks")
        plt.plot(good_xs, refined_ys, "r.", label="Subpixel refined peaks")
        plt.plot(xs, np.zeros_like(xs), "k--")
        plt.legend()
        plt.savefig(output_dir / "subpixel_peaks.png")
        plt.close()

    weights = np.array(weights)
    weights /= np.max(weights)
    mask = weights > 0.1
    return np.array(good_xs)[mask], np.array(refined_ys)[mask], weights[mask]


def robust_bspline_fit(
    x, y, weights, smoothness, g: Path, num_iter=10, c=4.685, visualize: bool = False
):
    """
    Fit a robust cubic B-spline to (x, y) data with given weights using iterative re-weighting.

    Parameters:
      x          : 1D numpy array of independent variable values.
      y          : 1D numpy array of dependent values.
      weights    : 1D numpy array of initial weights (same shape as x and y).
      smoothness : Smoothing factor (the s parameter in UnivariateSpline) that regularizes
                   against rapid oscillations.
      num_iter   : Maximum number of iterations for the IRLS procedure.
      c          : Tuning constant for Tukey’s biweight (default ~4.685 for normal errors).

    Returns:
      spline     : A UnivariateSpline object representing the robustly fitted cubic B-spline.

    Also plots the data and the final spline fit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or weights.shape != x.shape:
        raise ValueError("x, y, and weights must be one-dimensional and equal length")
    if len(x) < 4:
        raise ValueError("At least four motion samples are required for a cubic spline")

    def fit_irls(fit_x, fit_y, fit_weights):
        current_weights = np.copy(fit_weights)
        spline = None

        for _ in range(num_iter):
            # Use square-root weights as required by UnivariateSpline
            # (least-squares weighting).
            spline = UnivariateSpline(
                fit_x,
                fit_y,
                w=np.sqrt(current_weights),
                k=3,
                s=smoothness,
            )
            residuals = fit_y - spline(fit_x)

            # Robust scale estimate via median absolute deviation (MAD).
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = mad / 0.6745 if mad > 1e-6 else 1.0

            # Compute scaled residuals and Tukey biweight factors.
            u = residuals / (c * scale)
            robust_factor = np.square(1 - u**2)
            robust_factor[np.abs(u) >= 1] = 0
            new_weights = fit_weights * robust_factor

            if np.linalg.norm(new_weights - current_weights) < 1e-3 * np.linalg.norm(
                current_weights
            ):
                current_weights = new_weights
                break
            current_weights = new_weights

        return spline, current_weights

    def crosses_zero(spline, lower, upper):
        dense_x = np.linspace(lower, upper, 500)
        dense_y = spline(dense_x)
        return (np.max(dense_y) > 0) != (np.min(dense_y) > 0)

    def weighted_median(values, sample_weights):
        order = np.argsort(values)
        cumulative = np.cumsum(sample_weights[order])
        midpoint = 0.5 * cumulative[-1]
        return values[order[np.searchsorted(cumulative, midpoint)]]

    initial_weights = np.copy(weights)
    fit_x = x
    fit_y = y
    fit_weights = weights
    spline, current_weights = fit_irls(fit_x, fit_y, fit_weights)
    used_direction_fallback = False
    used_constant_fallback = False

    if crosses_zero(spline, np.min(x), np.max(x)):
        # A single pass can contain weak periodic aliases on both sides of
        # zero. Physical motion cannot reverse within one capture, so retry
        # using the well-supported direction instead of allowing the spline
        # to cross zero and catastrophically stretch the output.
        dominant_motion = weighted_median(y, weights)
        confidence_cutoff = 0.2 * np.max(weights)
        fallback_mask = (y * dominant_motion > 0) & (
            weights >= confidence_cutoff
        )
        if np.count_nonzero(fallback_mask) < 4:
            raise ValueError("Oscillating spline found")

        fit_x = x[fallback_mask]
        fit_y = y[fallback_mask]
        fit_weights = weights[fallback_mask]
        spline, current_weights = fit_irls(fit_x, fit_y, fit_weights)
        if crosses_zero(spline, np.min(fit_x), np.max(fit_x)):
            raise ValueError("Oscillating spline found")

        # Outside the supported interval, hold the nearest fitted speed. A
        # cubic extrapolation can otherwise invent another sign reversal.
        support_min = float(np.min(fit_x))
        support_max = float(np.max(fit_x))
        unclamped_spline = spline

        def spline(values):
            return unclamped_spline(np.clip(values, support_min, support_max))

        used_direction_fallback = True
        print(
            "speed fit: rejected reversal aliases; "
            f"using {len(fit_x)}/{len(x)} high-confidence samples "
            f"from {support_min:g} to {support_max:g}"
        )

    motion_dense = spline(np.linspace(np.min(x), np.max(x), 500))
    motion_magnitude = np.abs(motion_dense)
    dynamic_range = np.max(motion_magnitude) / max(
        np.min(motion_magnitude),
        1e-6,
    )
    if dynamic_range > SPEED_MAX_FIT_DYNAMIC_RANGE:
        # A train cannot plausibly change image speed by this much over one
        # selected pass without producing an obvious aspect-ratio drift. This
        # occurs when a weak capture's local correlations acquire a smooth
        # scene-dependent bias: IRLS cannot reject it because it is coherent.
        # Preserve ordinary acceleration fits, but replace a catastrophic
        # range with the robust constant supported by confident samples.
        dominant_motion = weighted_median(y, weights)
        confidence_cutoff = 0.2 * np.max(weights)
        fallback_mask = (y * dominant_motion > 0) & (
            weights >= confidence_cutoff
        )
        if np.count_nonzero(fallback_mask) < 4:
            raise ValueError("Implausible speed-fit dynamic range")

        fit_x = x[fallback_mask]
        fit_y = y[fallback_mask]
        fit_weights = weights[fallback_mask]
        constant_motion = float(weighted_median(fit_y, fit_weights))

        def spline(values):
            values = np.asarray(values, dtype=np.float64)
            return np.full_like(values, constant_motion, dtype=np.float64)

        current_weights = fit_weights
        used_constant_fallback = True
        print(
            "speed fit: "
            f"{dynamic_range:.2f}x dynamic range exceeds "
            f"{SPEED_MAX_FIT_DYNAMIC_RANGE:.2f}x; using robust constant "
            f"{constant_motion:.6f} from {len(fit_x)}/{len(x)} "
            "high-confidence samples"
        )

    # Plot the original data and the fitted spline with weights below.
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(x, y, "bo", label="Data")
    x_dense = np.linspace(np.min(x), np.max(x), 500)
    y_dense = spline(x_dense)
    ax1.plot(x_dense, y_dense, "r-", lw=2, label="Robust Cubic B-spline")
    if used_direction_fallback or used_constant_fallback:
        ax1.plot(
            fit_x,
            fit_y,
            "go",
            fillstyle="none",
            label=(
                "Constant-speed support"
                if used_constant_fallback
                else "Direction-consistent support"
            ),
        )
    ax1.plot(x, np.zeros_like(x), "k--")
    ax1.set_title(
        "Robust constant-speed fallback"
        if used_constant_fallback
        else "Robust Cubic B-spline Fit"
    )
    ax1.set_ylabel("y")
    ax1.legend()

    ax2.plot(x, initial_weights, "k.", label="Initial weights")
    ax2.plot(fit_x, current_weights, "r.", label="Final weights")
    ax2.set_xlabel("x")
    ax2.set_ylabel("weight")
    ax2.legend()

    plt.savefig(g / "spline.png")
    plt.close(fig)
    if visualize:
        plt.show()

    if (np.max(y_dense) > 0) != (np.min(y_dense) > 0):
        raise ValueError("Oscillating spline found")

    return spline


def raw_channels(data: np.ndarray):
    raw_red = data[0::2, 1::2]
    raw_green_1 = data[1::2, 1::2]
    raw_green_2 = data[0::2, 0::2]
    raw_blue = data[1::2, 0::2]

    return raw_red, raw_green_1, raw_green_2, raw_blue


def sharpen(rgb: np.ndarray) -> np.ndarray:
    rgb_blurred = cv2.GaussianBlur(rgb, (0, 0), 1) + cv2.GaussianBlur(rgb, (0, 0), 2)
    rgb = 1.4 * rgb - 0.2 * rgb_blurred
    return rgb


def patch_denoise(
    data: np.ndarray, neighbour_size: int = 64, similarity_thresh: float = 3
):
    # Patch-based denoising by searching horizontally along rows for similar 3x3 patches.
    denoised = data.copy().astype(np.float32)  # Ensure float32 for speed/memory.
    sqrt_data = np.sqrt(data).astype(np.float32)

    feature_size = 9
    height, width, channels = data.shape
    N = width - 2  # Number of patches per row.

    # Center indices in the feature vector (one per channel).
    center_indices = np.array([0, feature_size, 2 * feature_size])

    time_ab = 0
    time_bc = 0
    time_cd = 0

    for row in tqdm(range(1, height - 1), desc="denoising"):
        time_a = time.time()

        # Extract features (27 x N, float32).
        feature = np.zeros((3 * feature_size, N), dtype=np.float32)
        for channel in range(3):
            offset = channel * feature_size
            feature[offset + 0] = sqrt_data[row, 1:-1, channel]  # Center.
            feature[offset + 1] = sqrt_data[row, :-2, channel]  # Left.
            feature[offset + 2] = sqrt_data[row, 2:, channel]  # Right.
            feature[offset + 3] = sqrt_data[row - 1, 1:-1, channel]  # Top center.
            feature[offset + 4] = sqrt_data[row - 1, :-2, channel]  # Top left.
            feature[offset + 5] = sqrt_data[row - 1, 2:, channel]  # Top right.
            feature[offset + 6] = sqrt_data[row + 1, 1:-1, channel]  # Bottom center.
            feature[offset + 7] = sqrt_data[row + 1, :-2, channel]  # Bottom left.
            feature[offset + 8] = sqrt_data[row + 1, 2:, channel]  # Bottom right.

        time_b = time.time()

        sort_index = np.argsort(np.sum(feature, axis=0))

        feature_sorted = feature[:, sort_index]

        time_c = time.time()

        pad_value = -100.0
        w = 2 * neighbour_size + 1
        padded = np.pad(
            feature_sorted,
            ((0, 0), (neighbour_size, neighbour_size)),
            mode="constant",
            constant_values=pad_value,
        )

        # Sliding window view: (27, N, w).
        windows = sliding_window_view(padded, w, axis=1)

        # Center position in each window.
        center_pos = neighbour_size

        # Compute mask (N, w) by accumulating per-dimension checks (saves memory).
        mask = np.ones((N, w), dtype=bool)
        for dim in range(feature.shape[0]):
            dim_window = windows[dim, :, :]  # (N, w)
            center_dim = dim_window[:, center_pos]  # (N,)
            abs_diff = np.abs(dim_window - center_dim[:, np.newaxis])  # (N, w)
            mask &= abs_diff < similarity_thresh

        # Compute Gaussian weights based on distances for valid patches only
        gaussian_weights = np.zeros((N, w), dtype=np.float32)
        for i in range(N):
            center_feature = feature_sorted[:, i]  # (27,)
            window_features = windows[:, i, :]  # (27, w)
            valid_mask = mask[i, :] & (
                window_features[0, :] != pad_value
            )  # Valid patches (True in mask and not padded)
            valid_window = window_features[:, valid_mask]  # (27, num_valid)
            if valid_window.size > 0:  # Only compute if there are valid patches
                diff = valid_window - center_feature[:, np.newaxis]  # (27, num_valid)
                distances = np.sqrt(np.sum(diff**2, axis=0))  # (num_valid,)
                weights = np.exp(
                    -0.5 * (distances**2) / (similarity_thresh**2)
                )  # (num_valid,)
                weights_sum = np.sum(weights)
                if weights_sum > 0:  # Avoid division by zero
                    weights /= weights_sum  # Normalize to sum to 1
                gaussian_weights[i, valid_mask] = weights  # Assign to valid positions

        # Compute weighted means for each channel using Gaussian weights.
        for ch in range(3):
            center_window = windows[center_indices[ch], :, :]  # (N, w)
            masked = np.ma.masked_array(center_window, mask=~mask)
            # Apply Gaussian weights to the valid (unmasked) data
            weighted = masked * gaussian_weights
            means_sqrt = np.ma.sum(
                weighted, axis=1
            )  # Weighted sum (normalized to mean)
            # Assign to original positions.
            original_cols = sort_index + 1  # Offset for image indexing.
            denoised[row, original_cols, ch] = means_sqrt**2

        time_d = time.time()

        time_ab += time_b - time_a
        time_bc += time_c - time_b
        time_cd += time_d - time_c

    print("times", time_ab, time_bc, time_cd)
    return denoised


def bin2to1(data):
    h = (data.shape[0] // 2) * 2
    w = (data.shape[1] // 2) * 2
    if h == 0 or w == 0:
        return np.empty((0, 0), dtype=data.dtype)
    data = data[:h, :w]
    return (data[::2, ::2] + data[1::2, ::2] + data[::2, 1::2] + data[1::2, 1::2]) / 4


def estimate_white_balance(bin_paths, max_samples: int = 8) -> np.ndarray:
    if not bin_paths:
        return np.ones(3, dtype=np.float32)

    channel_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for bin_path in tqdm(
        bin_paths[:max_samples],
        desc="white balance",
    ):
        dat1 = np.fromfile(bin_path, dtype=np.uint16).astype(np.float32)
        dat1_reshaped = np.reshape(dat1, (-1, 4096)).T
        raw_red = bin2to1(dat1_reshaped[0::2, 1::2])
        raw_green = bin2to1(dat1_reshaped[1::2, 1::2])
        raw_blue = bin2to1(dat1_reshaped[1::2, 0::2])

        channel_sum[0] += raw_red.sum()
        channel_sum[1] += raw_green.sum()
        channel_sum[2] += raw_blue.sum()
        total_pixels += raw_red.size

    means = channel_sum / max(total_pixels, 1)
    mean_of_means = np.mean(means)
    with np.errstate(divide="ignore", invalid="ignore"):
        white_balance = mean_of_means / means
    white_balance[~np.isfinite(white_balance)] = 1.0
    return white_balance.astype(np.float32)


def get_percentile(hist, bin_edges, total, percent):
    if total == 0:
        return 0.0
    cum = np.cumsum(hist)
    thresh = (percent / 100.0) * total
    idx = np.searchsorted(cum, thresh, side="left")
    if idx == 0:
        return bin_edges[0]
    if idx == len(hist):
        return bin_edges[-1]
    prev_cum = cum[idx - 1]
    needed = thresh - prev_cum
    bin_count = hist[idx]
    frac = needed / bin_count if bin_count > 0 else 0
    bin_width = bin_edges[idx + 1] - bin_edges[idx]
    return bin_edges[idx] + frac * bin_width


def autoexposure(selected_bins, white_balance):
    num_bins = 65536
    bin_edges = np.arange(0, num_bins + 1, dtype=np.float64)
    histogram = np.zeros(num_bins, dtype=np.int64)
    total_values = 0

    for i, bin in tqdm(
        enumerate(selected_bins), total=len(selected_bins), desc="autoexposure"
    ):
        dat1 = np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        dat1_reshaped = np.reshape(dat1, (-1, 4096)).T
        raw_red = dat1_reshaped[0::2, 1::2]
        raw_red = bin2to1(raw_red)

        raw_blue = dat1_reshaped[1::2, 0::2]
        raw_blue = bin2to1(raw_blue)

        raw_green = dat1_reshaped[1::2, 1::2]
        raw_green = bin2to1(raw_green)

        stacked = np.stack((raw_red, raw_green, raw_blue), axis=2)
        rgb = apply_color_transform(stacked, white_balance)
        rgb = np.clip(rgb, 0, 65536)

        h, _ = np.histogram(rgb.ravel(), bins=bin_edges)
        histogram += h
        total_values += rgb.size
    p2 = get_percentile(histogram, bin_edges, total_values, 2)
    p98 = get_percentile(histogram, bin_edges, total_values, 98)
    return p2, p98


class ColumnJitterModel:
    """Smooth, channel-specific additive corrections at vertical band centers."""

    def __init__(
        self,
        band_corrections: np.ndarray,
        image_height: int,
        black_levels: np.ndarray,
        white_levels: np.ndarray,
        confidence: np.ndarray,
    ):
        corrections = np.asarray(band_corrections)
        if corrections.ndim != 3 or corrections.shape[0] != 4:
            raise ValueError("band_corrections must have shape (4, bands, columns)")
        if image_height < 1 or corrections.shape[1] < 1:
            raise ValueError("image height and jitter band count must be positive")
        self.band_corrections = corrections
        self.image_height = int(image_height)
        self.black_levels = np.asarray(black_levels, dtype=np.float32)
        self.white_levels = np.asarray(white_levels, dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.num_bands = corrections.shape[1]
        self.width = corrections.shape[2]
        if self.black_levels.shape != (4,) or self.white_levels.shape != (4,):
            raise ValueError("black_levels and white_levels must each contain 4 values")
        if self.confidence.shape != (self.width,):
            raise ValueError("confidence must contain one value per modeled column")

        # Band values live at band centers. Linear interpolation keeps application
        # cheap during motion-corrected sampling while avoiding band boundaries.
        band_position = (
            (np.arange(self.image_height, dtype=np.float32) + 0.5)
            * self.num_bands
            / self.image_height
            - 0.5
        )
        lower = np.floor(band_position).astype(np.int32)
        upper = lower + 1
        weight = band_position - lower
        below = lower < 0
        above = upper >= self.num_bands
        lower = np.clip(lower, 0, self.num_bands - 1)
        upper = np.clip(upper, 0, self.num_bands - 1)
        owner = np.floor(
            (np.arange(self.image_height, dtype=np.float32) + 0.5)
            * self.num_bands
            / self.image_height
        ).astype(np.int32)
        owner = np.clip(owner, 0, self.num_bands - 1)
        weight[below] = 0.0
        weight[above] = 0.0
        self._row_band_lower = lower
        self._row_band_upper = upper
        self._row_band_owner = owner
        self._row_band_weight = weight.astype(np.float32)

    def correct_samples(
        self,
        values: np.ndarray,
        channel: int,
        columns: np.ndarray,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        """Correct a 2-D sample array using either 1-D or per-pixel columns."""
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("values must be two-dimensional")
        if not 0 <= channel < 4:
            raise ValueError("channel must be in the range 0..3")
        if rows is None:
            rows = np.arange(values.shape[0], dtype=np.int32)
        else:
            rows = np.asarray(rows, dtype=np.int32)
        if rows.ndim != 1 or rows.size != values.shape[0]:
            raise ValueError("rows must contain one index per values row")

        columns = np.asarray(columns, dtype=np.int32)
        if columns.ndim == 1:
            if columns.size != values.shape[1]:
                raise ValueError("columns must contain one index per values column")
            column_grid = columns[None, :]
        elif columns.shape == values.shape:
            column_grid = columns
        else:
            raise ValueError("columns must be 1-D or have the same shape as values")
        if (
            np.any(rows < 0)
            or np.any(rows >= self.image_height)
            or np.any(column_grid < 0)
            or np.any(column_grid >= self.width)
        ):
            raise IndexError("jitter model sample is outside the modeled image")

        lower = self._row_band_lower[rows]
        upper = self._row_band_upper[rows]
        weight = self._row_band_weight[rows, None]
        channel_model = self.band_corrections[channel]
        correction_low = channel_model[lower[:, None], column_grid].astype(
            np.float32
        )
        correction_high = channel_model[upper[:, None], column_grid].astype(
            np.float32
        )
        correction = correction_low + (correction_high - correction_low) * weight
        # Interpolation may taper a supported band toward an unsupported
        # neighbor, but it must never carry that neighbor's value across the
        # boundary into a band whose own estimate is zero.
        correction_owner = channel_model[
            self._row_band_owner[rows, None],
            column_grid,
        ]
        correction = np.where(correction_owner != 0.0, correction, 0.0)

        black = float(self.black_levels[channel])
        white = float(self.white_levels[channel])
        span = max(white - black, 1.0)
        highlight_margin = max(
            JITTER_WHITE_MARGIN_MIN,
            JITTER_WHITE_MARGIN_FRACTION * span,
        )
        shadow_weight = np.clip(
            (values - black) / max(JITTER_BLACK_MARGIN, 1.0), 0.0, 1.0
        )
        highlight_weight = np.clip(
            (white - values) / highlight_margin, 0.0, 1.0
        )
        clip_weight = np.minimum(shadow_weight, highlight_weight)
        return values + correction * clip_weight

    def apply_window(
        self, window: np.ndarray, channel: int, start_col: int
    ) -> np.ndarray:
        """Apply the model to a contiguous full-height image window."""
        if window.size == 0:
            return window
        window = np.asarray(window, dtype=np.float32)
        result = window.copy()
        local_start = max(0, -start_col)
        local_stop = min(window.shape[1], self.width - start_col)
        if local_stop <= local_start:
            return result
        columns = np.arange(
            start_col + local_start,
            start_col + local_stop,
            dtype=np.int32,
        )
        result[:, local_start:local_stop] = self.correct_samples(
            window[:, local_start:local_stop],
            channel,
            columns,
        )
        return result


def apply_column_models(
    window: np.ndarray,
    models: ColumnJitterModel | None,
    start_col: int,
    channel: int,
) -> np.ndarray:
    if models is None:
        return window
    return models.apply_window(window, channel, start_col)


def _estimate_jitter_levels(channel_views, max_sample_columns: int = 8192):
    """Estimate clipping endpoints from windows spread across the capture."""
    width = channel_views[0].shape[1]
    sample_width = min(512, width)
    num_windows = max(1, min(16, math.ceil(max_sample_columns / sample_width)))
    max_start = max(0, width - sample_width)
    starts = np.unique(np.linspace(0, max_start, num_windows).astype(np.int64))
    black = np.full(4, np.inf, dtype=np.float32)
    white = np.full(4, -np.inf, dtype=np.float32)
    for channel, view in enumerate(channel_views):
        for start in starts:
            sample = view[:, slice(int(start), int(start) + sample_width)]
            if sample.size:
                black[channel] = min(black[channel], float(np.min(sample)))
                white[channel] = max(white[channel], float(np.max(sample)))
    black[~np.isfinite(black)] = 0.0
    white[~np.isfinite(white)] = 65535.0
    white = np.maximum(white, black + 1.0)
    return black, white


def _jitter_band_statistics(
    data: np.ndarray,
    num_bands: int,
    black: float,
    white: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = data.shape
    if height % num_bands:
        raise ValueError(
            f"image height {height} must be divisible by {num_bands} jitter bands"
        )
    banded = data.reshape(num_bands, height // num_bands, width)
    medians = np.median(banded, axis=1).astype(np.float32, copy=False)
    span = max(white - black, 1.0)
    high_margin = max(
        JITTER_WHITE_MARGIN_MIN,
        JITTER_WHITE_MARGIN_FRACTION * span,
    )
    invalid = (
        np.mean(banded <= black + JITTER_BLACK_MARGIN, axis=1)
        > JITTER_MAX_CLIPPED_FRACTION
    )
    invalid |= (
        np.mean(banded >= white - high_margin, axis=1)
        > JITTER_MAX_CLIPPED_FRACTION
    )
    return medians, ~invalid


def _centered_moving_median(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """Fast centered median with nearest-value padding at both ends."""
    radius = window // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (radius, radius)
    padded = np.pad(values, pad_width, mode="edge")
    trailing = bn.move_median(
        padded,
        window=window,
        min_count=window,
        axis=-1,
    )
    start = window - 1
    return trailing[..., start : start + values.shape[-1]].astype(
        np.float32,
        copy=False,
    )


def _jitter_structure_bands(
    structure_mask: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    num_bands: int,
    scale: int,
) -> np.ndarray:
    """Reduce a downsampled scene-edge mask to jitter band/column support."""
    if scale < 1:
        raise ValueError("structure_mask_scale must be positive")
    if image_height % scale or image_width % scale:
        raise ValueError("jitter image dimensions must be divisible by mask scale")
    expected_shape = (image_height // scale, image_width // scale)
    structure_mask = np.asarray(structure_mask, dtype=bool)
    if structure_mask.shape != expected_shape:
        raise ValueError(
            "structure_mask has shape "
            f"{structure_mask.shape}, expected {expected_shape}"
        )
    if expected_shape[0] % num_bands:
        raise ValueError("downsampled mask height must be divisible by num_bands")
    return structure_mask.reshape(
        num_bands,
        expected_shape[0] // num_bands,
        expected_shape[1],
    ).any(axis=1)


def column_exposure_jitter_correction(
    red_view,
    green1,
    green2,
    blue_view,
    g: Path | None = None,
    *,
    black_levels: np.ndarray | None = None,
    white_levels: np.ndarray | None = None,
    num_bands: int = JITTER_VERTICAL_BANDS,
    fine_temporal_window: int = JITTER_FINE_TEMPORAL_WINDOW,
    temporal_window: int = JITTER_TEMPORAL_WINDOW,
    block_columns: int = JITTER_BLOCK_COLUMNS,
    structure_mask: np.ndarray | None = None,
    structure_mask_scale: int = 1,
    show_progress: bool = True,
) -> ColumnJitterModel:
    """
    Estimate absolute, channel/row-specific exposure corrections.

    A centered temporal median supplies the local unstriped target. Unlike the
    former adjacent-column composition, this has no cumulative state: monotonic
    scene gradients are fixed points and cannot turn into correction drift.
    A short centered median removes narrow line jitter. A longer median handles
    broader exposure cycles, but that broad component is accepted only inside
    a small exposure-relative envelope so a bright scene plateau cannot be
    mistaken for camera error. Strong scene structure is excluded by a
    downsampled edge mask. This lets each Bayer channel and row band retain its
    own response instead of forcing unlike channels to share a sign. Clipped,
    structured, or otherwise unsupported bands receive zero correction;
    estimates are never filled or extrapolated from other parts of the sensor.
    """
    del g  # Retained in the signature for callers that group outputs by capture.
    channel_views = (red_view, green1, green2, blue_view)
    heights = {view.shape[0] for view in channel_views}
    widths = {view.shape[1] for view in channel_views}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("all four jitter channels must have the same dimensions")
    image_height = heights.pop()
    image_width = widths.pop()
    if image_height % num_bands:
        raise ValueError(
            f"image height {image_height} must be divisible by {num_bands}"
        )
    if fine_temporal_window < 3 or fine_temporal_window % 2 != 1:
        raise ValueError(
            "fine_temporal_window must be an odd integer of at least 3"
        )
    if temporal_window < fine_temporal_window or temporal_window % 2 != 1:
        raise ValueError(
            "temporal_window must be an odd integer no smaller than "
            "fine_temporal_window"
        )
    if block_columns < 1:
        raise ValueError("block_columns must be positive")

    if black_levels is None or white_levels is None:
        estimated_black, estimated_white = _estimate_jitter_levels(channel_views)
        if black_levels is None:
            black_levels = estimated_black
        if white_levels is None:
            white_levels = estimated_white
    black_levels = np.asarray(black_levels, dtype=np.float32)
    white_levels = np.asarray(white_levels, dtype=np.float32)
    if black_levels.shape != (4,) or white_levels.shape != (4,):
        raise ValueError("black_levels and white_levels must each contain 4 values")
    white_levels = np.maximum(white_levels, black_levels + 1.0)

    structure_bands = None
    if structure_mask is not None:
        structure_bands = _jitter_structure_bands(
            structure_mask,
            image_height=image_height,
            image_width=image_width,
            num_bands=num_bands,
            scale=structure_mask_scale,
        )

    band_corrections = np.empty(
        (4, num_bands, image_width),
        dtype=np.float16,
    )
    confidence_full = np.zeros(image_width, dtype=np.float32)
    radius = temporal_window // 2
    block_starts = range(0, image_width, block_columns)

    for core_start in tqdm(
        block_starts,
        total=math.ceil(image_width / block_columns),
        desc="column exposure jitter",
        disable=not show_progress,
    ):
        core_stop = min(image_width, core_start + block_columns)
        read_start = max(0, core_start - radius)
        read_stop = min(image_width, core_stop + radius)
        core_slice = slice(core_start - read_start, core_stop - read_start)
        core_width = core_stop - core_start
        fine_offsets = np.empty((4, num_bands, core_width), dtype=np.float32)
        broad_offsets = np.empty((4, num_bands, core_width), dtype=np.float32)
        valid = np.empty((4, num_bands, core_width), dtype=bool)

        for channel, view in enumerate(channel_views):
            data = view[:, slice(read_start, read_stop)]
            statistics, band_valid = _jitter_band_statistics(
                data,
                num_bands,
                float(black_levels[channel]),
                float(white_levels[channel]),
            )
            fine_target = _centered_moving_median(
                statistics,
                fine_temporal_window,
            )
            broad_target = _centered_moving_median(
                statistics,
                temporal_window,
            )
            fine_offsets[channel] = (fine_target - statistics)[:, core_slice]
            broad_offsets[channel] = (
                broad_target - fine_target
            )[:, core_slice]
            valid[channel] = band_valid[:, core_slice]

        if structure_bands is not None:
            lowres_columns = (
                np.arange(core_start, core_stop, dtype=np.int64)
                // structure_mask_scale
            )
            structured = structure_bands[:, lowres_columns]
            valid &= ~structured[None, :, :]

        spans = (white_levels - black_levels)[:, None, None]
        fine_supported = valid & (
            np.abs(fine_offsets)
            <= JITTER_FINE_MAX_CORRECTION_FRACTION * spans
        )
        broad_supported = valid & (
            np.abs(broad_offsets)
            <= JITTER_BROAD_MAX_CORRECTION_FRACTION * spans
        )
        local_corrections = np.where(
            fine_supported,
            fine_offsets,
            0.0,
        ) + np.where(
            broad_supported,
            broad_offsets,
            0.0,
        )
        band_corrections[:, :, core_start:core_stop] = np.clip(
            local_corrections,
            -np.finfo(np.float16).max,
            np.finfo(np.float16).max,
        )
        confidence_full[core_start:core_stop] = np.mean(
            fine_supported | broad_supported,
            axis=(0, 1),
        )

    return ColumnJitterModel(
        band_corrections,
        image_height,
        black_levels,
        white_levels,
        confidence_full,
    )


def interpolate_upsample(arr, off):
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.empty(0)
    res = np.zeros(2 * n)
    if off == 0:
        # Place original at even indices
        res[::2] = arr
        # Interpolate inner odd indices
        if n > 1:
            res[1:-1:2] = (res[:-2:2] + res[2::2]) / 2
        # Repeat at the end for the last odd index
        res[-1] = res[-2]
    elif off == 1:
        # Place original at odd indices
        res[1::2] = arr
        # Interpolate inner even indices (starting from 2)
        if n > 1:
            res[2::2] = (res[1:-1:2] + res[3::2]) / 2
        # Repeat at the start for index 0
        res[0] = res[1]
    return res


def generate_sample_positions(spline, sample_xs, width):
    sample_positions = []
    widths = []
    steps = []
    x = 0
    if spline(sample_xs[0]) < 0:
        x = width - 1
    while 0 <= x < width:
        if x < sample_xs[0]:
            sx = spline(sample_xs[0])
        elif x >= sample_xs[-1]:
            sx = spline(sample_xs[-1])
        else:
            sx = spline(x)
        # enforce minimum step
        if -0.1 < sx < 0.1:
            sx = 0.1 if sx > 0 else -0.1

        sample_positions.append(x)
        widths.append(abs(sx))
        steps.append(sx)
        x += sx

    widths = np.clip(widths, 1, width)
    return sample_positions, widths, steps


def truncate_sample_positions_for_skew(
    sample_positions,
    widths,
    *,
    skew: float,
    step_scales=None,
    row_offsets: np.ndarray,
    width: int,
):
    if not sample_positions:
        return sample_positions, widths
    positions = np.asarray(sample_positions, dtype=np.float32)
    widths_arr = np.asarray(widths, dtype=np.float32)
    if step_scales is None:
        step_scales = np.ones_like(positions)
    step_scales = np.asarray(step_scales, dtype=np.float32)
    skew_offsets = row_offsets[None, :] * (skew * step_scales[:, None])
    min_offset = np.min(skew_offsets, axis=1)
    max_offset = np.max(skew_offsets, axis=1)

    min_center = positions + min_offset - widths_arr
    max_center = positions + max_offset + widths_arr
    valid = (min_center >= 0.0) & (max_center <= (width - 1))
    if not np.any(valid):
        return [], []

    first = int(np.argmax(valid))
    last = int(len(valid) - 1 - np.argmax(valid[::-1]))
    return sample_positions[first : last + 1], widths[first : last + 1]


def sample_motion_corrected_chunks(
    red_view,
    green1,
    green2,
    blue_view,
    sample_positions,
    widths,
    *,
    stride_x: int = 1,
    stride_y: int = 1,
    skew: float = 0.0,
    max_pix: int = 16384,
    jitter_models=None,
    step_scales=None,
):
    indices = list(range(0, len(sample_positions), stride_x))
    if not indices:
        return
    if step_scales is None:
        step_scales = np.ones(len(sample_positions), dtype=np.float32)

    out_height = green1.height * 2
    center_y = (out_height - 1) / 2.0
    row_offsets = (2 * np.arange(green1.height) - center_y).astype(np.float32)

    chunk_i = 0
    for start in range(0, len(indices), max_pix):
        batch_indices = indices[start : start + max_pix]
        pad_left = start > 0
        if pad_left:
            use_indices = [indices[start - 1]] + batch_indices
        else:
            use_indices = list(batch_indices)

        out_width = len(use_indices)
        raw_sampled = np.zeros((2 * green1.height, out_width, 3), dtype=np.float32)

        for j, sample_idx in tqdm(
            enumerate(use_indices),
            total=len(use_indices),
            desc="sampling chunk",
        ):
            xi = sample_positions[sample_idx]
            wi = widths[sample_idx]
            skew_scaled = skew * step_scales[sample_idx]

            centers = xi + skew_scaled * row_offsets
            window_half = int(math.ceil(wi))
            offsets = np.arange(-window_half, window_half + 1, dtype=np.int32)
            base = np.floor(centers).astype(np.int32)
            window_indices = base[:, None] + offsets[None, :]
            valid = (window_indices >= 0) & (window_indices < green1.width)
            indices_clipped = np.clip(window_indices, 0, green1.width - 1)

            start_col = int(indices_clipped.min())
            end_col = int(indices_clipped.max()) + 1
            if end_col <= start_col:
                continue

            r_win = red_view[:, slice(start_col, end_col)]
            g1_win = green1[:, slice(start_col, end_col)]
            g2_win = green2[:, slice(start_col, end_col)]
            b_win = blue_view[:, slice(start_col, end_col)]

            indices_local = indices_clipped - start_col
            indices_local = np.clip(indices_local, 0, r_win.shape[1] - 1)
            sample_rows = np.arange(green1.height, dtype=np.int32)
            row_idx = sample_rows[:, None]

            r_vals = r_win[row_idx, indices_local]
            g1_vals = g1_win[row_idx, indices_local]
            g2_vals = g2_win[row_idx, indices_local]
            b_vals = b_win[row_idx, indices_local]

            if jitter_models is not None:
                r_vals = jitter_models.correct_samples(
                    r_vals,
                    JITTER_RED,
                    indices_clipped,
                    sample_rows,
                )
                g1_vals = jitter_models.correct_samples(
                    g1_vals,
                    JITTER_GREEN1,
                    indices_clipped,
                    sample_rows,
                )
                g2_vals = jitter_models.correct_samples(
                    g2_vals,
                    JITTER_GREEN2,
                    indices_clipped,
                    sample_rows,
                )
                b_vals = jitter_models.correct_samples(
                    b_vals,
                    JITTER_BLUE,
                    indices_clipped,
                    sample_rows,
                )

            dist = window_indices.astype(np.float32) - centers[:, None].astype(
                np.float32
            )
            weights = np.zeros_like(dist, dtype=np.float32)
            mask = (np.abs(dist) < wi) & valid
            weights[mask] = 1 + np.cos(np.pi * dist[mask] / wi)
            sum_weight = np.sum(weights, axis=1)
            sum_weight[sum_weight == 0] = 1.0

            r_weighted = np.sum(r_vals * weights, axis=1) / sum_weight
            g1_weighted = np.sum(g1_vals * weights, axis=1) / sum_weight
            g2_weighted = np.sum(g2_vals * weights, axis=1) / sum_weight
            b_weighted = np.sum(b_vals * weights, axis=1) / sum_weight

            r_interp = interpolate_upsample(r_weighted, 0)
            g1_interp = interpolate_upsample(g1_weighted, 1)
            g2_interp = interpolate_upsample(g2_weighted, 0)
            b_interp = interpolate_upsample(b_weighted, 1)

            if j == 0:
                raw_sampled[:, 0, 0] = r_interp
                raw_sampled[:, 0, 1] += g1_interp * 0.5
            if j < len(use_indices) - 1:
                raw_sampled[:, j + 1, 0] = r_interp
                raw_sampled[:, j + 1, 1] += g1_interp * 0.5
            raw_sampled[:, j, 1] += g2_interp * 0.5
            raw_sampled[:, j, 2] = b_interp

        if pad_left:
            raw_sampled = raw_sampled[:, 1:, :]
        if stride_y > 1:
            raw_sampled = raw_sampled[::stride_y, :, :]

        yield chunk_i, raw_sampled
        chunk_i += 1


def estimate_skew_hough(
    chunk_iter,
    *,
    g: Path,
    stride_x: int,
    stride_y: int,
    skew_min: float = -0.05,
    skew_max: float = 0.05,
    bins: int = 69,
    energy_percentile: float = 98,
):
    skews = np.linspace(skew_min, skew_max, bins, dtype=np.float32)
    scores = np.zeros_like(skews, dtype=np.float64)

    def find_peaks_nms(values: np.ndarray, min_distance: int) -> np.ndarray:
        if values.size < 3:
            return np.empty((0,), dtype=np.int32)
        peaks = np.zeros_like(values, dtype=bool)
        peaks[1:-1] = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
        peak_idxs = np.where(peaks)[0]
        if peak_idxs.size == 0:
            return peak_idxs
        peak_vals = values[peak_idxs]
        order = np.argsort(peak_vals)[::-1]
        selected = []
        for idx in peak_idxs[order]:
            if all(abs(idx - s) >= min_distance for s in selected):
                selected.append(int(idx))
        return np.array(selected, dtype=np.int32)

    for _chunk_i, chunk in tqdm(chunk_iter, desc="estimating skew"):
        if chunk.size == 0:
            continue
        gray = chunk[:, :, 1].astype(np.float32)
        grad_y, grad_x = np.gradient(gray)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 0.1 * np.max(gray)
        energy = np.abs(grad_x) / magnitude

        thresh = np.percentile(energy, energy_percentile)
        mask = energy >= thresh
        if not np.any(mask):
            continue

        energy = np.sqrt(energy)

        weights = energy * mask
        width = chunk.shape[1]
        hist_len = width * 3
        hist_offset = width
        xs = np.arange(width, dtype=np.int32)
        ys = np.arange(chunk.shape[0], dtype=np.int32)
        scale = stride_y / stride_x
        total_weight = float(np.sum(weights)) + 1e-6

        chunk_scores = np.zeros_like(skews, dtype=np.float64)
        histograms = np.zeros((len(skews), hist_len), dtype=np.float64)
        peak_values = [[] for _ in range(len(skews))]
        for i, skew in enumerate(skews):
            skew_ds = skew * scale
            hist = np.zeros(hist_len, dtype=np.float64)
            for row_idx, y in enumerate(ys):
                shift = int(round(skew_ds * y))
                b_idx = xs - shift + hist_offset
                valid = (b_idx >= 0) & (b_idx < hist_len)
                if not np.any(valid):
                    continue
                hist += np.bincount(
                    b_idx[valid],
                    weights=weights[row_idx, valid],
                    minlength=hist_len,
                )
            if hist.size:
                hist_norm = hist / total_weight
                histograms[i, :] = hist_norm
                peak_idxs = find_peaks_nms(hist_norm, min_distance=12)
                if peak_idxs.size:
                    peak_values[i] = sorted(
                        (float(hist_norm[p]) for p in peak_idxs), reverse=True
                    )
                else:
                    peak_values[i] = [float(np.max(hist_norm))]

        peak_max = np.array(
            [vals[0] if vals else 0.0 for vals in peak_values], dtype=np.float64
        )
        top_s = min(5, len(skews))
        top_indices = np.argsort(peak_max)[-top_s:]
        k = None
        for idx in top_indices:
            n_peaks = len(peak_values[idx])
            if n_peaks == 0:
                continue
            k = n_peaks if k is None else min(k, n_peaks)
        if k is None or k == 0:
            k = 1

        for i in range(len(skews)):
            vals = peak_values[i]
            if not vals:
                continue
            chunk_scores[i] = float(np.sum(vals[:k]))

        best_idx = int(np.argmax(chunk_scores))
        best_skew = skews[best_idx]
        skew_ds = best_skew * scale
        best_hist = (
            histograms[best_idx]
            if histograms.size
            else np.zeros(hist_len, dtype=np.float64)
        )
        best_offset = hist_offset

        energy_norm = energy / max(np.max(energy), 1e-6)
        mask_u8 = mask.astype(np.uint8) * 255
        energy_u8 = (np.clip(energy_norm, 0, 1) * 255).astype(np.uint8)

        energy_vis = cv2.cvtColor(energy_u8, cv2.COLOR_GRAY2BGR)
        energy_vis = np.ascontiguousarray(energy_vis[::-1, :, :])
        mask_vis = np.ascontiguousarray(mask_u8[::-1, :])

        peak_thresh = 0.7 * np.max(best_hist) if best_hist.size else 0
        if peak_thresh > 0:
            peaks = np.zeros_like(best_hist, dtype=bool)
            peaks[1:-1] = (best_hist[1:-1] > best_hist[:-2]) & (
                best_hist[1:-1] >= best_hist[2:]
            )
            peaks &= best_hist >= peak_thresh
            peak_idxs = np.where(peaks)[0]
        else:
            peak_idxs = []

        h = energy_vis.shape[0]
        for b in peak_idxs:
            intercept = b - best_offset
            x0 = int(round(intercept + skew_ds * (h - 1)))
            x1 = int(round(intercept))
            y0 = 0
            y1 = h - 1
            if (0 <= x0 < energy_vis.shape[1]) or (0 <= x1 < energy_vis.shape[1]):
                cv2.line(energy_vis, (x0, y0), (x1, y1), (0, 0, 255), 1)

        cv2.imwrite(str(g / f"skew_energy_{_chunk_i:02d}.png"), energy_vis)
        cv2.imwrite(str(g / f"skew_mask_{_chunk_i:02d}.png"), mask_vis)

        if histograms.size:
            fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
            ax.imshow(
                histograms,
                cmap="gray",
                aspect="auto",
                origin="lower",
                extent=[-hist_offset, hist_len - hist_offset, skews[0], skews[-1]],
            )
            ax.axhline(best_skew, color="red", linewidth=1)
            ax.set_ylabel("skew slope")
            ax.set_xlabel("intercept")
            fig.savefig(g / f"skew_histogram{_chunk_i:02d}.png")
            plt.close(fig)

        plt.figure(figsize=(16, 9), dpi=300)
        plt.plot(chunk_scores, "b.", label="Skew response")
        plt.legend()
        plt.savefig(g / f"skew_response_{_chunk_i:04d}.png")
        plt.close()
        scores += chunk_scores

    if np.all(scores == 0):
        return 0.0
    return float(skews[int(np.argmax(scores))])


def process_preview(
    g: Path,
    padding: int = 10,
):
    bins = sorted(list(g.glob("*.bin")))

    downsample_steps = int(math.log2(STRUCTURE_MASK_DOWNSAMPLE))
    if 2**downsample_steps != STRUCTURE_MASK_DOWNSAMPLE:
        raise ValueError("STRUCTURE_MASK_DOWNSAMPLE must be a power of two")

    max_green = 0
    max_green_down = 0
    jitter_black_levels = np.full(4, np.inf, dtype=np.float32)
    jitter_white_levels = np.full(4, -np.inf, dtype=np.float32)
    for i, bin in tqdm(enumerate(bins), total=len(bins)):
        dat1 = np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        dat1_reshaped = np.reshape(dat1, (-1, 4096)).T
        raw_green = dat1_reshaped[1::2, 1::2]
        max_green = max(max_green, np.max(raw_green))
        if ENABLE_COLUMN_JITTER_CORRECTION:
            raw_channels = (
                dat1_reshaped[0::2, 1::2],
                dat1_reshaped[1::2, 1::2],
                dat1_reshaped[0::2, 0::2],
                dat1_reshaped[1::2, 0::2],
            )
            for channel, raw_channel in enumerate(raw_channels):
                jitter_black_levels[channel] = min(
                    jitter_black_levels[channel],
                    float(np.min(raw_channel)),
                )
                jitter_white_levels[channel] = max(
                    jitter_white_levels[channel],
                    float(np.max(raw_channel)),
                )
        raw_green_down = raw_green
        for _ in range(downsample_steps):
            raw_green_down = bin2to1(raw_green_down)
        max_green_down = max(max_green_down, np.max(raw_green_down))

    scores = []
    for i, bin in tqdm(enumerate(bins), total=len(bins), desc="finding moving region"):
        dat1 = np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        dat1_reshaped = np.reshape(dat1, (-1, 4096)).T

        raw_green = dat1_reshaped[1::2, 1::2]
        raw_green = bin2to1(raw_green)

        grad_y, grad_x = np.gradient(raw_green)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 0.1 * max_green
        energy = np.abs(grad_x) / magnitude
        # cv2.imwrite(str(g / f"score_{i:05d}.png"), energy * 255)
        score = np.percentile(energy, 99)
        scores.append(score)

    min_score = min(scores)
    selected = set()
    for i, score in enumerate(scores):
        if score > min_score * 1.3:
            for pi in range(-padding, padding + 1):
                if i + pi >= 0 and i + pi < len(bins):
                    selected.add(i + pi)
    selected = sorted(list(selected))
    if FORCE_CONTIGUOUS_SELECTION and selected:
        selected = list(range(selected[0], selected[-1] + 1))
    print("selected", format_contiguous_segments(selected))

    selected_bins = [bins[s] for s in selected]

    white_balance = estimate_white_balance(selected_bins)
    print("white balance", white_balance)

    p2, p98 = autoexposure(selected_bins, white_balance)

    # lazily load binary data across selected files without full-memory concatenation
    cache = FileDataCache(selected_bins, dtype=np.uint16, rows=4096)
    data = cache

    red_view = ChannelView(data, row_offset=0, col_offset=1)
    blue_view = ChannelView(data, row_offset=1, col_offset=0)
    green1 = ChannelView(data, row_offset=1, col_offset=1)
    green2 = ChannelView(data, row_offset=0, col_offset=0)

    energy_bins = []
    green_bins = []
    for i, _bin_path in tqdm(
        enumerate(selected_bins), total=len(selected_bins), desc="stripe mask"
    ):
        start = cache.cum_cols[i] // green1.col_step
        stop = cache.cum_cols[i + 1] // green1.col_step
        raw_green = green1[:, slice(start, stop)]
        for _ in range(downsample_steps):
            raw_green = bin2to1(raw_green)
        green_bins.append(raw_green)

        grad_y, grad_x = np.gradient(raw_green)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 0.1 * max_green_down
        energy_down = np.abs(grad_x) / magnitude
        energy_bins.append(energy_down)

    if energy_bins:
        energy_full = np.hstack(energy_bins)
        green_full = np.hstack(green_bins)
    else:
        energy_full = np.empty((0, 0), dtype=np.float32)
        green_full = np.empty((0, 0), dtype=np.float32)

    mask_full = energy_full > STRUCTURE_MASK_THRESHOLD
    if mask_full.size > 0:
        preview_mask = mask_full.astype(np.uint8) * 255
        cv2.imwrite(str(g / "structure_mask_undilated.png"), preview_mask)

    if STRUCTURE_MASK_DILATE_PX > 0 and mask_full.size > 0:
        dilate_px = max(1, STRUCTURE_MASK_DILATE_PX // STRUCTURE_MASK_DOWNSAMPLE)
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_full = cv2.dilate(mask_full.astype(np.uint8), kernel) > 0

    if mask_full.size > 0:
        green_norm = green_full / max(max_green_down, 1.0)
        green_u8 = np.clip(green_norm * 255.0, 0, 255).astype(np.uint8)
        base = cv2.cvtColor(green_u8, cv2.COLOR_GRAY2BGR)
        mask_overlay = np.zeros_like(base)
        mask_overlay[:, :, 2] = mask_full.astype(np.uint8) * 255
        preview = cv2.addWeighted(base, 0.9, mask_overlay, 0.1, 0.0)
        cv2.imwrite(str(g / "structure_mask_preview.png"), preview)

    jitter_models = None
    if ENABLE_COLUMN_JITTER_CORRECTION:
        jitter_models = column_exposure_jitter_correction(
            red_view,
            green1,
            green2,
            blue_view,
            g=g,
            black_levels=jitter_black_levels,
            white_levels=jitter_white_levels,
            structure_mask=mask_full,
            structure_mask_scale=STRUCTURE_MASK_DOWNSAMPLE,
        )
    if ENABLE_JITTER_DEBUG_PLOT and jitter_models is not None:
        row_idx = min(JITTER_DEBUG_ROW, green1.height - 1)
        cols = np.arange(green1.width, dtype=np.int32)
        pre_channels = (
            red_view[row_idx, :],
            green1[row_idx, :],
            green2[row_idx, :],
            blue_view[row_idx, :],
        )
        colors = ("r", "g", "k", "b")
        labels = ("red", "green1", "green2", "blue")
        post_channels = tuple(
            jitter_models.correct_samples(
                values[None, :],
                channel,
                cols,
                np.array([row_idx], dtype=np.int32),
            )[0]
            for channel, values in enumerate(pre_channels)
        )

        fig, (ax_pre, ax_post, ax_delta, ax_confidence) = plt.subplots(
            4, 1, figsize=(16, 14), dpi=200, sharex=True
        )
        for values, color, label in zip(pre_channels, colors, labels):
            ax_pre.plot(cols, values, color, linewidth=0.35, label=label)
        ax_pre.set_ylabel("value")
        ax_pre.set_title(f"Jitter correction row {row_idx} (pre)")
        ax_pre.legend()

        for values, color, label in zip(post_channels, colors, labels):
            ax_post.plot(cols, values, color, linewidth=0.35, label=label)
        ax_post.set_ylabel("value")
        ax_post.set_title("post")
        ax_post.legend()

        for pre, post, color, label in zip(
            pre_channels, post_channels, colors, labels
        ):
            ax_delta.plot(
                cols,
                post - pre,
                color,
                linewidth=0.35,
                label=label,
            )
        ax_delta.set_title("applied additive correction")
        ax_delta.set_ylabel("raw units")
        ax_delta.legend()

        ax_confidence.plot(cols, jitter_models.confidence, "m", linewidth=0.5)
        ax_confidence.set_title("cross-channel/row consensus")
        ax_confidence.set_ylabel("confidence")
        ax_confidence.set_xlabel("column")
        ax_confidence.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        fig.savefig(g / "jitter_row_debug.png")
        plt.close(fig)

    sample_xs, sample_ys, _weights = windowed_cross_correlation(
        green1, green2, jitter_models=jitter_models
    )

    spline = robust_bspline_fit(
        x=sample_xs,
        y=sample_ys,
        weights=_weights,
        smoothness=80000,
        visualize=False,
        g=g,
    )

    print(g, len(bins), data.shape)

    sample_positions, widths, step_scales = generate_sample_positions(
        spline, sample_xs, green1.width
    )
    step_scales = np.asarray(step_scales, dtype=np.float32)
    # chunk size for sampling
    skew_chunk_size = 4096
    hough_stride = 3

    skew = estimate_skew_hough(
        sample_motion_corrected_chunks(
            red_view,
            green1,
            green2,
            blue_view,
            sample_positions,
            widths,
            stride_x=hough_stride,
            stride_y=hough_stride,
            skew=0.0,
            max_pix=skew_chunk_size,
            jitter_models=jitter_models,
            step_scales=step_scales,
        ),
        g=g,
        stride_x=hough_stride,
        stride_y=hough_stride,
    )

    print("estimated skew", skew)

    out_height = green1.height * 2
    center_y = (out_height - 1) / 2.0
    row_offsets = (2 * np.arange(green1.height) - center_y).astype(np.float32)
    sample_positions, widths = truncate_sample_positions_for_skew(
        sample_positions,
        widths,
        skew=skew,
        step_scales=step_scales,
        row_offsets=row_offsets,
        width=green1.width,
    )
    step_scales = step_scales[: len(sample_positions)]

    n = len(sample_positions)
    target = 32768
    n_chunks = max(1, int(math.ceil(n / target)))
    chunk_size = max(1, n // n_chunks)
    n_use = chunk_size * n_chunks
    if n_use < n:
        sample_positions = sample_positions[:n_use]
        widths = widths[:n_use]
        step_scales = step_scales[:n_use]

    output_chunk_count = int(math.ceil(len(sample_positions) / chunk_size))
    stacked_preview = []
    for chunk_i, raw_sampled in sample_motion_corrected_chunks(
        red_view,
        green1,
        green2,
        blue_view,
        sample_positions,
        widths,
        stride_x=1,
        stride_y=1,
        skew=skew,
        max_pix=chunk_size,
        jitter_models=jitter_models,
        step_scales=step_scales,
    ):
        output_chunk_i = (
            output_chunk_count - 1 - chunk_i if FLIP_OUTPUT_180 else chunk_i
        )
        out_name = f"rgb_{output_chunk_i:02d}_cfpeak_deskewed_jitter"
        # denoised_sampled = patch_denoise(raw_sampled)
        denoised_sampled = raw_sampled
        dng_sampled = denoised_sampled
        if FLIP_OUTPUT_180:
            dng_sampled = np.ascontiguousarray(dng_sampled[::-1, ::-1, :])
        write_linear_dng(g / f"{out_name}_linear.dng", dng_sampled, white_balance)
        # calibrate, clip, sharpen, tone-map
        rgb = apply_color_transform(denoised_sampled, white_balance)
        rgb = np.clip(rgb, 0, 65536)
        rgb = sharpen(rgb)
        rgb = rgb - p2
        rgb += 0.1
        rgb /= p98
        rgb = np.clip(rgb, 0, 1)
        rgb = rgb[::-1, :, :]
        rgb = np.sqrt(rgb)
        rgb *= 255

        rgb_u8 = rgb[:, :, ::-1].astype(np.uint8)
        if FLIP_OUTPUT_180:
            rgb_u8 = np.ascontiguousarray(rgb_u8[::-1, ::-1, :])
        for ext in (".jpg", ".png"):
            cv2.imwrite(str(g / (out_name + ext)), rgb_u8)

        rgb_float = rgb_u8.astype(np.float32)
        binned = np.stack(
            [bin2to1(rgb_float[:, :, ch]) for ch in range(rgb_float.shape[2])],
            axis=2,
        )
        stacked_preview.append(np.clip(binned, 0, 255).astype(np.uint8))

    if stacked_preview:
        if FLIP_OUTPUT_180:
            stacked_preview = stacked_preview[::-1]
        stacked = np.vstack(stacked_preview)
        cv2.imwrite(str(g / "preview_stack.jpg"), stacked)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Demosaic, dewarp, and render Alkeria line-scan captures."
    )
    parser.add_argument(
        "captures",
        nargs="*",
        help=(
            "capture directories; relative names are resolved under "
            f"{linescans}"
        ),
    )
    parser.add_argument(
        "--jitter",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_COLUMN_JITTER_CORRECTION,
        help="enable the clipping-aware vertical-stripe correction",
    )
    parser.add_argument(
        "--flip-output-180",
        action=argparse.BooleanOptionalAction,
        default=FLIP_OUTPUT_180,
        help="rotate final output by 180 degrees",
    )
    parser.add_argument(
        "--jitter-debug",
        action="store_true",
        default=ENABLE_JITTER_DEBUG_PLOT,
        help="write a row-level before/after jitter diagnostic plot",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    global ENABLE_COLUMN_JITTER_CORRECTION
    global ENABLE_JITTER_DEBUG_PLOT
    global FLIP_OUTPUT_180

    args = _parse_args(argv)
    ENABLE_COLUMN_JITTER_CORRECTION = args.jitter
    ENABLE_JITTER_DEBUG_PLOT = args.jitter_debug
    FLIP_OUTPUT_180 = args.flip_output_180

    if args.captures:
        candidates = []
        for capture_arg in args.captures:
            candidate = Path(capture_arg).expanduser()
            if not candidate.is_absolute() and not candidate.exists():
                candidate = linescans / candidate
            candidates.append(candidate)
    else:
        candidates = sorted(
            linescans.glob("2026-06-16-02-21-07*"),
            reverse=True,
        )

    dirs = []
    for candidate in candidates:
        if candidate.is_dir():
            dirs.append(candidate)
        else:
            print("skipping directory", candidate)

    if len(dirs) == 1:
        process_preview(dirs[0])
        return

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {executor.submit(process_preview, g): g for g in dirs}
        try:
            for future in as_completed(future_map):
                g = future_map[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"oh no! {g} {e}")
        except KeyboardInterrupt:
            for future in future_map:
                future.cancel()


if __name__ == "__main__":
    main()
