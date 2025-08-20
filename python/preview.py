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
    visualize: bool = False,
) -> np.ndarray:
    """
    green_1: (h, w) where w >> h and w >> window_size
    green_2: (h, w)
    """

    xs = np.arange(0, green_1.shape[1], window_step)
    # xs = np.array([0])
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
            refined_peak_idx = subpixel_peak(ncorrs, 2.0, peak_idx)

            ys.append(peak_idx - corr_half)
            refined_ys.append(refined_peak_idx - corr_half)
            good_xs.append(xi * window_step + window_size // 2)

    if visualize:
        plt.imshow(heatmap)
        plt.show()
        plt.plot(good_xs, ys, "b.", label="Raw peaks")
        plt.plot(good_xs, refined_ys, "r.", label="Subpixel refined peaks")
        plt.plot(xs, np.zeros_like(xs), "k--")
        plt.show()

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
    plt.figure(figsize=(8, 5))
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


def calibrate_black_point(
    data: np.ndarray,
    top: int = 512,
    bottom: int = 512,
    similarity_thresh: float = 1000,
    min_px: int = 64,
):
    median_top = np.median(data[:top, :], axis=1)
    median_bottom = np.median(data[-bottom:, :], axis=1)
    height = data.shape[0]
    rows = np.linspace(0, 1, num=height)
    x = np.concatenate((rows[:top], rows[-bottom:]))
    data2 = np.zeros_like(data)
    for col in tqdm(range(data.shape[1]), desc="calibrating black point"):
        y = np.concatenate(
            (
                median_top - data[:top, col],
                median_bottom - data[-bottom:, col],
            )
        )

        mask = np.abs(y) < similarity_thresh
        if np.sum(mask) < min_px:
            continue
        poly = np.polynomial.Polynomial.fit(x[mask], y[mask], 1)
        data2[:, col] = data[:, col] + poly.linspace(height)[1]
    return data2


def patch_denoise_old(
    data: np.ndarray, neighbour_size: int = 128, similarity_thresh: float = 3
):
    # patch based denoising by searching horizontally along rows for similar 3x3 patches
    denoised = data.copy()

    time_ab = 0
    time_bc = 0
    time_cd = 0

    # since the data is poisson distributed, the standard deviation is proportional to sqrt
    # so we sqrt it first so that we just need to compare it to a constant
    sqrt_data = np.sqrt(data)

    feature_size = 9
    for row in tqdm(range(1, data.shape[0] - 1), desc="denoising"):
        # for row in tqdm(range(1000, 1050), desc="denoising"):
        time_a = time.time()
        feature = np.zeros((3 * feature_size, data.shape[1] - 2))
        for channel in range(3):
            feature[0 + channel * feature_size] = sqrt_data[row, 1:-1, channel]
            feature[1 + channel * feature_size] = sqrt_data[row, :-2, channel]
            feature[2 + channel * feature_size] = sqrt_data[row, 2:, channel]
            feature[3 + channel * feature_size] = sqrt_data[row - 1, 1:-1, channel]
            feature[4 + channel * feature_size] = sqrt_data[row - 1, :-2, channel]
            feature[5 + channel * feature_size] = sqrt_data[row - 1, 2:, channel]
            feature[6 + channel * feature_size] = sqrt_data[row + 1, 1:-1, channel]
            feature[7 + channel * feature_size] = sqrt_data[row + 1, :-2, channel]
            feature[8 + channel * feature_size] = sqrt_data[row + 1, 2:, channel]
        feature_mean = np.mean(feature, axis=0)

        time_b = time.time()

        sort_index = np.argsort(feature_mean)
        feature_sorted = feature[:, sort_index]
        index_in_sorted = np.argsort(sort_index)

        # kd = scipy.spatial.KDTree(feature.T)
        # dist, ind = kd.query(feature.T, neighbour_size)

        time_c = time.time()

        similars = 0
        for col in range(data.shape[1] - 2):
            sorted_col = index_in_sorted[col]
            neighbours = feature_sorted[
                :,
                max(0, sorted_col - neighbour_size) : min(
                    data.shape[1] - 2, sorted_col + neighbour_size
                ),
            ]
            # neighbours = feature[:, ind[col, :]]
            neighbour_mask = (
                np.max(np.abs(neighbours - feature[:, col : col + 1]), axis=0)
                < similarity_thresh
            )
            similars += np.sum(neighbour_mask)
            for channel in range(3):
                denoised[row, col + 1, channel] = np.square(
                    np.mean(neighbours[channel * feature_size, neighbour_mask])
                )

        time_d = time.time()
        # print(similars / (data.shape[1] - 2))

        time_ab += time_b - time_a
        time_bc += time_c - time_b
        time_cd += time_d - time_c
    print("times", time_ab, time_bc, time_cd)
    return denoised


def patch_denoise(
    data: np.ndarray, neighbour_size: int = 256, similarity_thresh: float = 3
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

        # Better sorting key: project onto first principal component for max variance direction.
        data_centered = feature.T - np.mean(feature.T, axis=0)
        _, _, Vt = np.linalg.svd(data_centered, full_matrices=False)
        proj = data_centered @ Vt[0, :].T
        sort_index = np.argsort(proj)

        # Sort features.
        feature_sorted = feature[:, sort_index]

        time_c = time.time()

        # Pad sorted features to handle edge windows (use -100 to ensure diffs exceed thresh).
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

        # Total similars (excluding self, but includes for averaging).
        similars = np.sum(mask) - N  # Self always included.

        # Average centers for each channel using masked arrays.
        for ch in range(3):
            center_window = windows[center_indices[ch], :, :]  # (N, w)
            masked = np.ma.masked_array(center_window, mask=~mask)
            means_sqrt = np.ma.mean(masked, axis=1)
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
        # cv2.imwrite(f"/home/dllu/test/{i:05d}.png", energy * 255)
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

    # set up lazy channel views for green channels
    green1 = ChannelView(data, row_offset=1, col_offset=1)
    green2 = ChannelView(data, row_offset=0, col_offset=0)
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

    # Build raw channel views
    red_view = ChannelView(data, row_offset=0, col_offset=1)
    green1 = ChannelView(data, row_offset=1, col_offset=1)
    green2 = ChannelView(data, row_offset=0, col_offset=0)
    blue_view = ChannelView(data, row_offset=1, col_offset=0)
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
        widths.append(abs(sx * 2))
        x += sx

    widths = np.clip(widths, 1, n)
    # chunk size for sampling
    max_pix = 65535

    for chunk_i, batch in enumerate(
        itertools.batched(zip(sample_positions, widths), max_pix)
    ):
        out_name = f"rgb_{chunk_i}_hann.png"
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

            # hann window
            dist = np.arange(start, end) - xi
            weights = np.zeros_like(dist)
            mask = np.abs(dist) < wi
            weights[mask] = 1 + np.cos(np.pi * dist[mask] / wi)
            sum_weight = np.sum(weights)

            r_win = np.sum(r_win * weights, axis=1) / sum_weight
            g1_win = np.sum(g1_win * weights, axis=1) / sum_weight
            g2_win = np.sum(g2_win * weights, axis=1) / sum_weight
            b_win = np.sum(b_win * weights, axis=1) / sum_weight

            r_interp = interpolate_upsample(r_win, 0)
            g1_interp = interpolate_upsample(g1_win, 1)
            g2_interp = interpolate_upsample(g2_win, 0)
            b_interp = interpolate_upsample(b_win, 1)

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
        cv2.imwrite(str(g / out_name), rgb[:, :, ::-1].astype(np.uint8))


def main():
    # for g in sorted(list(linescans.glob("2024-09-10-06-00-05*"))): # yamanote
    # for g in sorted(list(linescans.glob("17-12-03-23-11-30-utc"))):  # bart
    # for g in sorted(list(linescans.glob("2024-09-10-06-00-05*"))):  # yamanote
    # for g in sorted(list(linescans.glob("2024-09-19-03-33-57*"))): # fuxing
    # for g in sorted(list(linescans.glob("2024-09-14-02-13-13*"))):  # hktram
    # for g in sorted(list(linescans.glob("2024-09*"))):  # hktram
    # for g in sorted(list(linescans.glob("2024-09-12-01-31-01*"))):  # hk bus
    # for g in sorted(list(linescans.glob("nyc4"))):
    # for g in sorted(list(linescans.glob("18*"))):
    for g in sorted(list(linescans.glob("18-10-06-13-52-16-utc*"))):  # pato
    # for g in sorted(list(linescans.glob("18-08-04*"))):  # amtrak
    # for g in sorted(list(linescans.glob("2023*"))):
    # for g in sorted(list(linescans.glob("17-10-01-23-46-50-utc"))):
    # for g in sorted(list(linescans.glob("*"))):
    # for g in sorted(list(linescans.glob("18*"))):
        if not g.is_dir():
            continue

        print(g)
        try:
            process_preview(g)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"oh no! {g} {e}")
            continue  # anyway


if __name__ == "__main__":
    main()
