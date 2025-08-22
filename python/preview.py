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

# Color calibration matrix for RGB conversion
COLOR_CALIBRATION = np.array(
    [
        [0.9, -0.3, -0.3],
        [-0.8, 1.6, -0.3],
        [-0.5, -0.5, 2.0],
    ]
)

# linescans = Path("/home/dllu/pictures/linescan")
linescans = Path("/mnt/data14/pictures/linescan")


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
            peak_idx = np.argmax(ncorrs)
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
    return (data[::2, ::2] + data[1::2, ::2] + data[::2, 1::2] + data[1::2, 1::2]) / 4


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


def autoexposure(selected_bins):
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

        rgb = np.tensordot(
            np.stack((raw_red, raw_green, raw_blue), axis=2),
            COLOR_CALIBRATION.T,
            axes=([2], [0]),
        )
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


def process_preview(g: Path, padding: int = 3, max_chunks: int = 512):
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
                selected.add(i + pi)
    selected = sorted(list(selected))
    print("selected", selected)

    selected_bins = [bins[s] for s in selected]

    p2, p98 = autoexposure(selected_bins)

    # lazily load binary data across selected files without full-memory concatenation
    cache = FileDataCache(selected_bins, dtype=np.uint16, rows=4096)
    data = cache

    red_view = ChannelView(data, row_offset=0, col_offset=1)
    blue_view = ChannelView(data, row_offset=1, col_offset=0)
    green1 = ChannelView(data, row_offset=1, col_offset=1)
    green2 = ChannelView(data, row_offset=0, col_offset=0)

    jitter_models = column_exposure_jitter_correction(
        red_view, green1, green2, blue_view, g=g
    )

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

    # Generate sample positions from spline
    sample_positions = []
    widths = []
    x = 0
    # total width
    n = data.total_cols // 2
    if spline(sample_xs[0]) < 0:
        x = n - 1
    while 0 <= x < n:
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

    widths = np.clip(widths, 1, n)
    # chunk size for sampling
    max_pix = 16384

    for chunk_i, batch in enumerate(
        itertools.batched(zip(sample_positions, widths), max_pix)
    ):
        out_name = f"rgb_{chunk_i}_prod_no_denoise"

        raw_sampled = np.zeros((2 * green1.height, len(batch), 3), dtype=np.float32)

        for j, (xi, wi) in tqdm(
            enumerate(batch),
            total=len(batch),
            desc="sampling " + out_name,
        ):
            start = max(int(math.floor(xi - wi)), 0)
            end = min(int(math.ceil(xi + wi) + 1), green1.width - 1)

            r_win = red_view[:, slice(start, end)]
            g1_win = green1[:, slice(start, end)]
            g2_win = green2[:, slice(start, end)]
            b_win = blue_view[:, slice(start, end)]

            for col in range(start, end):
                win_i = col - start
                r_win[:, win_i] = apply_column_correction(
                    r_win[:, win_i], jitter_models[col, :]
                )
                g1_win[:, win_i] = apply_column_correction(
                    g1_win[:, win_i], jitter_models[col, :]
                )
                g2_win[:, win_i] = apply_column_correction(
                    g2_win[:, win_i], jitter_models[col, :]
                )
                b_win[:, win_i] = apply_column_correction(
                    b_win[:, win_i], jitter_models[col, :]
                )

            # hann window
            dist = np.arange(start, end) - xi
            weights = np.zeros_like(dist)
            mask = np.abs(dist) < wi
            weights[mask] = 1 + np.cos(np.pi * dist[mask] / wi)
            sum_weight = np.sum(weights)

            r_weighted = np.sum(r_win * weights, axis=1) / sum_weight
            g1_weighted = np.sum(g1_win * weights, axis=1) / sum_weight
            g2_weighted = np.sum(g2_win * weights, axis=1) / sum_weight
            b_weighted = np.sum(b_win * weights, axis=1) / sum_weight

            r_interp = interpolate_upsample(r_weighted, 0)
            g1_interp = interpolate_upsample(g1_weighted, 1)
            g2_interp = interpolate_upsample(g2_weighted, 0)
            b_interp = interpolate_upsample(b_weighted, 1)

            if j == 0:
                raw_sampled[:, 0, 0] = r_interp
                raw_sampled[:, 0, 1] += g1_interp * 0.5
            if j < len(batch) - 1:
                raw_sampled[:, j + 1, 0] = r_interp
                raw_sampled[:, j + 1, 1] += g1_interp * 0.5
            raw_sampled[:, j, 1] += g2_interp * 0.5
            raw_sampled[:, j, 2] = b_interp
        # denoised_sampled = patch_denoise(raw_sampled)
        denoised_sampled = raw_sampled
        # calibrate, clip, sharpen, tone-map
        rgb = np.tensordot(denoised_sampled, COLOR_CALIBRATION.T, axes=([2], [0]))
        rgb = np.clip(rgb, 0, 65536)
        rgb = sharpen(rgb)
        rgb = rgb - p2
        rgb += 0.1
        rgb /= p98
        rgb = np.clip(rgb, 0, 1)
        rgb = rgb[::-1, :, :]
        rgb = np.sqrt(rgb)
        rgb *= 255

        for ext in (".jpg", ".png"):
            cv2.imwrite(str(g / (out_name + ext)), rgb[:, :, ::-1].astype(np.uint8))


def main():
    # globs = linescans.glob("17-12-03-23-11-30-utc")  # bart
    # globs = linescans.glob("2024-09-10-06-00-05*")  # yamanote
    globs = linescans.glob("2024-09-19*") # fuxing
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
    # globs = linescans.glob("18-10-06-13-52-16-utc*") # pato

    globs = sorted(list(globs))
    for g in globs:
        if not g.is_dir():
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
