#!/usr/bin/env python
import math
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
import itertools
import tifffile

# Color calibration components for RGB conversion
RAW_COLOR_CORRECTION = np.array(
    [
        [0.9, -0.3, -0.3],
        [-0.8, 1.5, -0.3],
        [-0.5, -0.5, 1.7],
    ],
    dtype=np.float32,
)

CAMERA_MODEL_NAME = "Alkeria Necta N4K2-7C"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    deficit = 1.0 - row_sums
    adjusted = matrix.astype(np.float32).copy()
    diag = np.arange(adjusted.shape[0])
    adjusted[diag, diag] += deficit[:, 0]
    return adjusted


COLOR_CORRECTION = _normalize_rows(RAW_COLOR_CORRECTION)


def apply_color_transform(
    raw_rgb: np.ndarray,
    white_balance: np.ndarray,
    color_matrix: np.ndarray = COLOR_CORRECTION,
) -> np.ndarray:
    """Apply per-channel white balance followed by the 3x3 color matrix."""
    wb = np.asarray(white_balance, dtype=np.float32).reshape(1, 1, 3)
    balanced = raw_rgb * wb
    return np.tensordot(balanced, color_matrix.T, axes=([2], [0]))


def write_linear_dng(
    path: Path, raw_rgb: np.ndarray, white_balance: np.ndarray
) -> None:
    if tifffile is None:
        raise RuntimeError(
            "tifffile is required for DNG export. Install it via `uv pip install tifffile`."
        )

    linear = np.clip(raw_rgb[::-1, :, :], 0, 65535).astype(np.uint16)
    wb = np.asarray(white_balance, dtype=np.float64)
    safe_wb = np.clip(wb, 1e-6, None)
    as_shot_neutral = 1.0 / safe_wb
    if as_shot_neutral[1] != 0:
        as_shot_neutral /= as_shot_neutral[1]

    color_matrix = COLOR_CORRECTION.astype(np.float64).ravel()
    extratags = [
        (50706, "B", 4, (1, 4, 0, 0)),  # DNGVersion
        (50707, "B", 4, (1, 2, 0, 0)),  # DNGBackwardVersion
        (
            50708,
            "s",
            len(CAMERA_MODEL_NAME) + 1,
            CAMERA_MODEL_NAME,
        ),  # UniqueCameraModel
        (50721, "d", len(color_matrix), tuple(color_matrix)),  # ColorMatrix1
        (50728, "d", 3, tuple(as_shot_neutral.tolist())),  # AsShotNeutral
        (50730, "H", 1, 21),  # CalibrationIlluminant1 = D65
    ]

    tifffile.imwrite(
        path,
        linear,
        photometric="rgb",
        metadata={"Software": "nectar-preview"},
        extratags=extratags,
    )


# linescans = Path("/home/dllu/pictures/linescan")
linescans = Path("/mnt/dataz/pictures/linescan")


class FileDataCache:
    def __init__(self, bin_paths, dtype, rows):
        self.bin_paths = bin_paths
        self.dtype = dtype
        self.rows = rows
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
        self.cache = {}

    @property
    def shape(self):
        return (self.height, self.total_cols)

    def _load_file(self, idx):
        if idx not in self.cache:
            # load and reshape file idx
            data1d = np.fromfile(self.bin_paths[idx], dtype=self.dtype).astype(
                np.float32
            )
            # reshape to (rows, cols) then transpose to (height, cols)
            data2d = data1d.reshape(-1, self.height).T
            self.cache[idx] = data2d
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
                blocks.append(arr_file[:, lo:hi])
            if not blocks:
                return np.empty((self.height, 0), dtype=np.float32)
            arr = np.concatenate(blocks, axis=1)
            # apply column step
            if step != 1:
                arr = arr[:, ::step]
            # apply row indexing
            return arr[row_idx, :]
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
    x = np.arange(len(y))
    sigma2 = sigma**2
    for iter in range(10):
        weights = np.exp(-((x - peak) ** 2) / sigma2) * y
        peak = np.sum(weights * x) / np.sum(weights)
    return peak


def windowed_cross_correlation(
    green_1: np.ndarray,
    green_2: np.ndarray,
    window_step: int = 256,
    window_size: int = 4096,
    corr_size: int = 15,
    output_dir: Path = None,
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

    for xi, x in tqdm(
        enumerate(xs[1:-1]), desc="train speed estimation", total=len(xs)
    ):
        window_1 = green_1[:, x : x + window_step]

        corrs = np.zeros((corr_size,))
        for corr_ind in range(corr_size):
            corr_x = corr_ind - corr_half
            window_2 = green_2[
                :,
                x + corr_x : x + window_step + corr_x,
            ]

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
            # peak_idx = np.argmax(ncorrs)
            refined_peak_idx = subpixel_peak(ncorrs, 2.0, peak_idx)

            ys.append(peak_idx - corr_half)
            refined_ys.append(refined_peak_idx - corr_half)
            good_xs.append(xi * window_step + window_size // 2)

            if output_dir is not None and x > 20000 and x < 25000:
                plt.figure(figsize=(16, 9), dpi=300)
                plt.plot(ncorrs, "b.", label="Raw peaks")
                plt.plot([peak_idx], [ncorrs[peak_idx]], "r.", label="Naive peak")
                plt.plot(
                    [refined_peak_idx, refined_peak_idx],
                    [0, np.max(ncorrs)],
                    "g--",
                    label="Subpixel refined peaks",
                )
                plt.legend()
                plt.savefig(output_dir / f"mean_shift_{xi:04d}.png")

    if output_dir is not None:
        plt.imshow(heatmap)
        plt.savefig(output_dir / "heatmap.png")

        plt.figure(figsize=(16, 9), dpi=300)
        plt.plot(good_xs, ys, "b.", label="Raw peaks")
        plt.plot(good_xs, refined_ys, "r.", label="Subpixel refined peaks")
        plt.plot(xs, np.zeros_like(xs), "k--")
        plt.legend()
        plt.savefig(output_dir / "subpixel_peaks.png")

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
    current_weights = np.copy(weights)
    spline = None

    for i in range(num_iter):
        # Use square-root weights as required by UnivariateSpline (least-squares weighting)
        spline = UnivariateSpline(x, y, w=np.sqrt(current_weights), k=3, s=smoothness)
        y_fit = spline(x)
        residuals = y - y_fit

        # Robust scale estimate via median absolute deviation (MAD)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = mad / 0.6745 if mad > 1e-6 else 1.0

        # Compute scaled residuals and the Tukey biweight factors:
        u = residuals / (c * scale)
        robust_factor = np.square(1 - u**2)
        robust_factor[np.abs(u) >= 1] = 0

        new_weights = weights * robust_factor

        # Check for convergence in weight updates.
        if np.linalg.norm(new_weights - current_weights) < 1e-3 * np.linalg.norm(
            current_weights
        ):
            current_weights = new_weights
            break
        current_weights = new_weights

    # Plot the original data and the fitted spline on a dense grid.
    plt.figure(figsize=(16, 9), dpi=300)
    plt.plot(x, y, "bo", label="Data")
    x_dense = np.linspace(np.min(x), np.max(x), 500)
    y_dense = spline(x_dense)
    plt.plot(x_dense, y_dense, "r-", lw=2, label="Robust Cubic B-spline")
    plt.plot(x, np.zeros_like(x), "k--")
    plt.title("Robust Cubic B-spline Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(g / "spline.png")
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
    return (
        data[::2, ::2]
        + data[1::2, ::2]
        + data[::2, 1::2]
        + data[1::2, 1::2]
    ) / 4


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


def weighted_linear_regression(y, x, w):
    """
    Perform weighted linear regression: y = a + b*x + c*ind
    where ind = np.arange(len(x))

    Parameters:
    y : 1D numpy array
    x : 1D numpy array (same size as y)
    w : 1D numpy array of non-negative weights (same size as y)

    Returns:
    np.array([a, b, c])
    """
    n = len(x)
    ind = np.arange(n)

    # Design matrix: columns [1, x, ind]
    X = np.column_stack((np.ones_like(x), x, ind))
    wX = w[:, None] * X  # n x 3
    XtWX = X.T @ wX  # 3 x 3

    # Compute X^T W y: X.T @ (w * y)
    XtWy = X.T @ (w * y)  # 3 x 1

    # Solve XtWX @ beta = XtWy for beta
    damping = 0
    beta = np.linalg.solve(XtWX + damping * np.eye(X.shape[1]), XtWy)
    return beta


def apply_column_correction(x, beta):
    ind = np.arange(len(x))
    return beta[0] + x * beta[1] + ind * beta[2]


def compose_model(beta1, beta2):
    return np.array(
        [
            beta1[0] + beta1[1] * beta2[0],
            beta1[1] * beta2[1],
            beta1[2] + beta1[1] * beta2[2],
        ]
    )


def invert_model(beta):
    return np.array([-beta[0] / beta[1], 1 / beta[1], -beta[2] / beta[1]])


def column_exposure_jitter_correction(
    red_view, green1, green2, blue_view, g: Path, sigma=500
):
    n_cols = red_view.width

    off_r = np.zeros(n_cols)
    off_g = np.zeros(n_cols)
    off_b = np.zeros(n_cols)

    locs = [7095, 7390]

    green_currs_to_plot = []

    betas = np.column_stack(
        [np.zeros(n_cols - 1), np.ones(n_cols - 1), np.zeros(n_cols - 1)]
    )

    red_prev = red_view[:, 0]
    green_prev = 0.5 * (green1[:, 0] + green2[:, 0])
    blue_prev = blue_view[:, 0]
    for col in tqdm(range(1, n_cols), desc="column exposure jitter"):
        red_curr = red_view[:, col]
        green_curr = 0.5 * (green1[:, col] + green2[:, col])
        blue_curr = blue_view[:, col]

        red_delta = red_curr - red_prev
        green_delta = green_curr - green_prev
        blue_delta = blue_curr - blue_prev

        beta = np.array([0, 1, 0])
        for iter in range(3):
            red_delta = red_curr - apply_column_correction(red_prev, beta)
            green_delta = green_curr - apply_column_correction(green_prev, beta)
            blue_delta = blue_curr - apply_column_correction(blue_prev, beta)

            iter_sigma2 = 0.5**iter / (sigma * sigma)
            weight = np.exp(
                -(np.square(red_delta) + np.square(green_delta) + np.square(blue_delta))
                * iter_sigma2
            )

            weight[400:1600] = 0  # exclude middle region likely to include subject

            weight = weight * weight

            gray_curr = np.column_stack((red_curr, green_curr, blue_curr)).mean(axis=1)
            gray_prev = np.column_stack((red_prev, green_prev, blue_prev)).mean(axis=1)
            beta = weighted_linear_regression(gray_curr, gray_prev, weight)

        betas[col - 1, :] = beta

        red_prev = red_curr
        green_prev = green_curr
        blue_prev = blue_curr

    beta = np.array([0, 1, 0])
    global_models = np.zeros((n_cols, 3))
    for col in range(n_cols):
        global_models[col] = beta
        if col < n_cols - 1:
            lamb = 0.02
            beta = (1 - lamb) * compose_model(
                invert_model(betas[col]), beta
            ) + lamb * np.array([0, 1, 0])
    return global_models

    # DEBUG!!!
    beta = np.array([0, 1, 0])
    for iter in range(3):
        red_prev = red_view[:, locs[0]]
        green_prev = green1[:, locs[0]] + green2[:, locs[0]]
        blue_prev = blue_view[:, locs[0]]

        red_curr = red_view[:, locs[1]]
        green_curr = green1[:, locs[1]] + green2[:, locs[1]]
        blue_curr = blue_view[:, locs[1]]

        red_delta = red_curr - apply_column_correction(red_prev, beta)
        green_delta = green_curr - apply_column_correction(green_prev, beta)
        blue_delta = blue_curr - apply_column_correction(blue_prev, beta)

        iter_sigma2 = 0.5**iter / (sigma * sigma)
        weight = np.exp(
            -(np.square(red_delta) + np.square(green_delta) + np.square(blue_delta))
            * iter_sigma2
        )

        weight = weight * weight
        beta = weighted_linear_regression(green_curr, green_prev, weight)

    red_prev = red_view[:, locs[0]]
    green_prev = green1[:, locs[0]] + green2[:, locs[0]]
    blue_prev = blue_view[:, locs[0]]

    red_curr = red_view[:, locs[1]]
    green_curr = green1[:, locs[1]] + green2[:, locs[1]]
    blue_curr = blue_view[:, locs[1]]

    fig = plt.figure(figsize=(32, 18), dpi=300)
    gs = fig.add_gridspec(5, 1)

    # Main plot (takes up top 3/4 of the figure)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.plot(green_prev, "", label="prev")
    ax1.plot(green_curr, "", label="curr")
    ax1.plot(apply_column_correction(green_prev, beta), "", label="model(prev)")
    ax1.plot([0, 2047], [0, 0], "k--")
    ax1.set_xlim(0, 2047)
    ax1.set_ylabel("intensity")
    ax1.legend()

    # Weight plot (takes up bottom 1/4 of the figure)
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(weight / np.max(weight), "k", label="weight")
    ax2.set_xlim(0, 2047)
    ax2.set_ylabel("weight")
    ax2.set_title("weight")
    ax2.legend()

    ax3 = fig.add_subplot(gs[4, 0])
    ax3.plot(
        np.square(weight * (green_curr - green_prev)), label="weighted initial error"
    )
    ax3.plot(
        np.square(weight * (green_curr - apply_column_correction(green_prev, beta))),
        label="weighted final error",
    )
    ax3.plot([0, 2047], [0, 0], "k--")
    ax3.set_xlim(0, 2047)
    ax3.set_ylabel("intensity")
    ax3.set_title("squared error")
    ax3.legend()
    plt.savefig(g / "jitter_debug.png")

    plt.figure(figsize=(32, 18), dpi=300)
    plt.plot(green_prev, green_curr, "k.")
    plt.plot(apply_column_correction(green_prev, beta), green_curr, "r.")
    plt.xlabel("prev")
    plt.ylabel("curr")
    plt.savefig(g / "jitter_debug_2.png")

    plt.figure(figsize=(32, 18), dpi=300)
    for label, val in green_currs_to_plot:
        plt.plot(val, label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(g / "green_currs.png")
    return off_r, off_g, off_b


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
        x += sx

    widths = np.clip(widths, 1, width)
    return sample_positions, widths


def truncate_sample_positions_for_skew(
    sample_positions,
    widths,
    *,
    skew: float,
    row_offsets: np.ndarray,
    width: int,
):
    if not sample_positions:
        return sample_positions, widths
    positions = np.asarray(sample_positions, dtype=np.float32)
    widths_arr = np.asarray(widths, dtype=np.float32)
    skew_offsets = skew * row_offsets
    min_offset = float(np.min(skew_offsets))
    max_offset = float(np.max(skew_offsets))

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
):
    indices = list(range(0, len(sample_positions), stride_x))
    if not indices:
        return

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

            centers = xi + skew * row_offsets
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
            row_idx = np.arange(green1.height)[:, None]

            r_vals = r_win[row_idx, indices_local]
            g1_vals = g1_win[row_idx, indices_local]
            g2_vals = g2_win[row_idx, indices_local]
            b_vals = b_win[row_idx, indices_local]

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
    skew_min: float = -0.03,
    skew_max: float = 0.03,
    bins: int = 32,
    energy_percentile: float = 98.0,
):
    skews = np.linspace(skew_min, skew_max, bins, dtype=np.float32)
    scores = np.zeros_like(skews, dtype=np.float64)

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

        if _chunk_i == 0:
            cv2.imwrite(str(g / "skew_energy.png"), energy * 255)
            cv2.imwrite(str(g / "skew_mask.png"), mask * 255)

        weights = energy * mask
        width = chunk.shape[1]
        xs = np.arange(width, dtype=np.int32)
        ys = np.arange(chunk.shape[0], dtype=np.int32)
        scale = stride_y / stride_x

        chunk_scores = np.zeros_like(skews, dtype=np.float64)
        for i, skew in enumerate(skews):
            skew_ds = skew * scale
            hist = np.zeros(width, dtype=np.float64)
            for row_idx, y in enumerate(ys):
                shift = int(round(skew_ds * y))
                b_idx = xs - shift
                valid = (b_idx >= 0) & (b_idx < width)
                if not np.any(valid):
                    continue
                hist += np.bincount(
                    b_idx[valid],
                    weights=weights[row_idx, valid],
                    minlength=width,
                )
            if hist.size:
                chunk_scores[i] += float(np.max(hist))
        print("chunk", _chunk_i, ":", chunk_scores)

        plt.figure(figsize=(16, 9), dpi=300)
        plt.plot(chunk_scores, "b.", label="Skew response")
        plt.legend()
        plt.savefig(g / f"skew_response_{_chunk_i:04d}.png")
        scores += chunk_scores

    if np.all(scores == 0):
        return 0.0
    return float(skews[int(np.argmax(scores))])


def process_preview(
    g: Path,
    padding: int = 3,
    max_chunks: int = 512,
    preview_stride: int = 4,
):
    bins = sorted(list(g.glob("*.bin")))

    max_green = 0
    for i, bin in tqdm(enumerate(bins), total=len(bins)):
        dat1 = np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        dat1_reshaped = np.reshape(dat1, (-1, 4096)).T
        raw_green = dat1_reshaped[1::2, 1::2]
        max_green = max(max_green, np.max(raw_green))

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
        if score > min_score * 1.5:
            for pi in range(-padding, padding + 1):
                if i + pi >= 0 and i + pi < len(bins):
                    selected.add(i + pi)
    selected = sorted(list(selected))
    print("selected", selected)

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

    """
    jitter_models = column_exposure_jitter_correction(
        red_view, green1, green2, blue_view, g=g
    )
    """

    sample_xs, sample_ys, _weights = windowed_cross_correlation(green1, green2)

    spline = robust_bspline_fit(
        x=sample_xs,
        y=sample_ys,
        weights=np.ones_like(sample_xs),
        smoothness=20000,
        visualize=False,
        g=g,
    )

    print(g, len(bins), data.shape)

    sample_positions, widths = generate_sample_positions(
        spline, sample_xs, green1.width
    )
    # chunk size for sampling
    max_pix = 16384

    skew = estimate_skew_hough(
        sample_motion_corrected_chunks(
            red_view,
            green1,
            green2,
            blue_view,
            sample_positions,
            widths,
            stride_x=preview_stride,
            stride_y=preview_stride,
            skew=0.0,
            max_pix=max_pix,
        ),
        g=g,
        stride_x=preview_stride,
        stride_y=preview_stride,
    )
    print("estimated skew", skew)

    out_height = green1.height * 2
    center_y = (out_height - 1) / 2.0
    row_offsets = (2 * np.arange(green1.height) - center_y).astype(np.float32)
    sample_positions, widths = truncate_sample_positions_for_skew(
        sample_positions,
        widths,
        skew=skew,
        row_offsets=row_offsets,
        width=green1.width,
    )

    n = len(sample_positions)
    target = 32768
    n_chunks = max(1, int(math.ceil(n / target)))
    chunk_size = max(1, n // n_chunks)
    n_use = chunk_size * n_chunks
    if n_use < n:
        sample_positions = sample_positions[:n_use]
        widths = widths[:n_use]

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
    ):
        out_name = f"rgb_{chunk_i:02d}_prod_no_denoise_deskewed"
        # denoised_sampled = patch_denoise(raw_sampled)
        denoised_sampled = raw_sampled
        write_linear_dng(
            g / f"{out_name}_linear_deskewed.dng", denoised_sampled, white_balance
        )
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
        for ext in (".jpg", ".png"):
            cv2.imwrite(str(g / (out_name + ext)), rgb_u8)

        rgb_float = rgb_u8.astype(np.float32)
        binned = np.stack(
            [
                bin2to1(bin2to1(rgb_float[:, :, ch]))
                for ch in range(rgb_float.shape[2])
            ],
            axis=2,
        )
        stacked_preview.append(np.clip(binned, 0, 255).astype(np.uint8))

    if stacked_preview:
        stacked = np.vstack(stacked_preview)
        cv2.imwrite(str(g / "preview_stack.jpg"), stacked)


def main():
    # globs = linescans.glob("17-12-03-23-11-30-utc")  # bart
    # globs = linescans.glob("2024-09-10-06-00-05*")  # yamanote
    # globs = linescans.glob("2024-09-19*") # fuxing
    # globs = linescans.glob("2024-09-14-02-13-13*")  # hktram
    # globs = linescans.glob("2024-09*")  # hktram
    # globs = linescans.glob("2024-09-12-01-31-01*")  # hk bus
    # globs = linescans.glob("nyc4")
    # globs = linescans.glob("18*")
    # globs = linescans.glob("18-08-04*")  # amtrak
    # globs = linescans.glob("2023*")
    # globs = linescans.glob("17-10-01-23-46-50-utc")
    # globs = linescans.glob("*")
    # globs = linescans.glob("18*")
    # globs = linescans.glob("18-10-06-13-52-16-utc*")  # pato
    # globs = linescans.glob("2025-11*") # pano
    globs = linescans.glob("2025-09-13-09-18-45*") # picadilly

    globs = sorted(list(globs))[-1:]

    for g in globs:
        if not g.is_dir():
            print("skipping directory", g)
            continue

        print(g)
        if len(globs) == 1:
            process_preview(g)
        else:
            try:
                process_preview(g)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"oh no! {g} {e}")
                continue  # anyway


if __name__ == "__main__":
    main()
