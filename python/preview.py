#!/usr/bin/env python
import cv2
import numpy as np
from pathlib import Path
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy

linescans = Path("/home/dllu/pictures/linescan")
# linescans = Path("/mnt/data14/pictures/linescan")
preview = "preview_raw.png"


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
    window_step: int = 512,
    window_size: int = 2048,
    corr_size: int = 31,
) -> np.ndarray:
    """
    green_1: (h, w) where w >> h and w >> window_size
    green_2: (h, w)
    """

    # green_1 = green_1[1224:, :]
    # green_2 = green_2[1224:, :]
    xs = np.arange(0, green_1.shape[1] - window_size, window_step)
    ys = []
    refined_ys = []
    good_xs = []
    corr_half = corr_size // 2

    heatmap = np.zeros((len(xs), corr_size))
    print(heatmap.shape)

    for xi, x in enumerate(xs):
        window_2 = green_2[:, x + corr_half : x + window_size - corr_half]

        corrs = np.zeros((corr_size,))
        for corr_ind in range(corr_size):
            corr_x = corr_ind - corr_half
            window_1 = green_1[
                :,
                x + corr_half + corr_x : x + window_size - corr_half + corr_x,
            ]

            # corr_value = np.sum(window_1 * window_2)
            error = np.sum(np.abs(window_1 - window_2))
            corr_value = -error
            # corr_value = -np.log(error)

            corrs[corr_ind] = corr_value

        peak_idx = np.argmax(corrs)

        if peak_idx <= 0 or peak_idx >= corr_size - 1:
            continue

        # polyx = np.array([peak_idx - 1, peak_idx, peak_idx + 1])
        # polyy = corrs[polyx]
        # coeffs = np.polyfit(polyx, polyy, 2)

        # The vertex of the parabola (peak of the quadratic fit) is at -b/(2a)
        # refined_peak_idx = -coeffs[1] / (2 * coeffs[0])
        ncorrs = corrs - np.min(corrs)
        ncorrs = ncorrs / np.max(ncorrs)
        refined_peak_idx = subpixel_peak(ncorrs, 1.5, peak_idx)

        # plt.plot(ncorrs, 'b+')
        # plt.plot([refined_peak_idx, refined_peak_idx], [0, 1], 'k--')
        # plt.show()

        # corrs -= np.min(corrs)
        # corrs /= np.max(corrs)
        heatmap[xi, :] = corrs

        ys.append(peak_idx - corr_half)
        refined_ys.append(refined_peak_idx - corr_half)
        good_xs.append(x)
    plt.show()

    plt.imshow(heatmap)
    plt.show()
    plt.plot(good_xs, ys, "b.")
    plt.plot(good_xs, refined_ys, "r.")
    plt.plot(xs, np.zeros_like(xs), "k--")
    plt.show()


def bin_to_rgb(data: np.ndarray) -> np.ndarray:
    """
    data: mosaiced rggb bayer array
    """
    raw_red = data[0::2, 1::2]
    raw_green_1 = data[1::2, 1::2]
    raw_green_2 = data[0::2, 0::2]

    # plt.plot(raw_green_1[:, 10000])
    # plt.plot(raw_green_2[:, 10000])
    # plt.show()

    raw_blue = data[1::2, 0::2]

    windowed_cross_correlation(raw_green_1, raw_green_2)

    raw_green = (raw_green_1 + raw_green_2) / 2

    # raw_red = calibrate_black_point(raw_red)
    # raw_green = calibrate_black_point(raw_green)
    # raw_blue = calibrate_black_point(raw_blue)

    raw_stack = np.stack((raw_red, raw_green, raw_blue), axis=2)

    color_calibration = np.array([
        [0.9, -0.3, -0.3],
        [-0.8, 1.6, -0.3],
        [-0.5, -0.5, 2.0],
    ])

    rgb = np.dot(raw_stack, color_calibration.T)
    rgb = np.clip(rgb, 0, 65536)

    # rgb = patch_denoise(rgb)

    rgb = sharpen(rgb)
    rgb -= np.percentile(rgb, 2)
    rgb += 0.1
    rgb /= np.percentile(rgb, 95) * 0.9
    rgb[rgb < 0] = 0
    rgb[rgb > 1] = 1
    # for c in range(3):
    # rgb[:, :, c] -= np.min(rgb[:, :, c])
    # rgb[:, :, c] /= np.max(rgb[:, :, c])
    rgb = rgb[::-1, :, :]
    rgb = np.sqrt(rgb)
    rgb *= 255

    return rgb


def sharpen(rgb: np.ndarray) -> np.ndarray:
    rgb_blurred = cv2.GaussianBlur(rgb, (0, 0), 1)
    rgb = rgb - 0.2 * rgb_blurred
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
        y = np.concatenate((
            median_top - data[:top, col],
            median_bottom - data[-bottom:, col],
        ))

        mask = np.abs(y) < similarity_thresh
        if np.sum(mask) < min_px:
            continue
        poly = np.polynomial.Polynomial.fit(x[mask], y[mask], 1)
        data2[:, col] = data[:, col] + poly.linspace(height)[1]
    return data2


def patch_denoise(
    data: np.ndarray, neighbour_size: int = 256, similarity_thresh: float = 7
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
        print(similars / (data.shape[1] - 2))

        time_ab += time_b - time_a
        time_bc += time_c - time_b
        time_cd += time_d - time_c
    print("times", time_ab, time_bc, time_cd)
    return denoised


def process_preview(g: Path, padding: int = 5, max_chunks: int = 192):
    bins = sorted(list(g.glob("*.bin")))

    start = np.inf
    finish = 0
    max_green = 0
    for i, bin in tqdm(enumerate(bins), total=len(bins)):
        dat1 = np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        dat1_reshaped = np.reshape(dat1, (-1, 4096)).T
        raw_green = dat1_reshaped[1::2, 1::2]
        max_green = max(max_green, np.max(raw_green))

    scores = []
    for i, bin in tqdm(enumerate(bins), total=len(bins), desc="Finding moving region"):
        dat1 = np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        dat1_reshaped = np.reshape(dat1, (-1, 4096)).T
        raw_green = dat1_reshaped[1::2, 1::2]

        raw_green = (
            raw_green[::2, ::2]
            + raw_green[1::2, ::2]
            + raw_green[::2, 1::2]
            + raw_green[1::2, 1::2]
        ) / 4
        grad_y, grad_x = np.gradient(raw_green)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 0.1 * max_green
        energy = np.abs(grad_x) / magnitude
        # cv2.imwrite(f"/home/dllu/test/{i:05d}.png", energy * 255)
        score = np.percentile(energy, 99)
        scores.append(score)

    min_score = min(scores)
    for i, score in enumerate(scores):
        if score > min_score * 1.5:
            start = min(start, max(i - padding, 0))
            finish = max(finish, min(i + padding, len(bins)))
    print(start, finish)
    if start == np.inf:
        start = 0
    if finish == 0:
        finish = len(bins) - 1

    if finish - start <= max_chunks:
        all_data = [
            np.fromfile(bin, dtype=np.uint16).astype(np.float32)
            for bin in bins[start : finish + 1]
        ]
    else:
        all_data = [
            np.fromfile(bin, dtype=np.uint16).astype(np.float32)
            for bin in bins[start : start + max_chunks // 2]
        ] + [
            np.fromfile(bin, dtype=np.uint16).astype(np.float32)
            for bin in bins[finish - max_chunks // 2 : finish + 1]
        ]
    data = np.concatenate(all_data)
    data = np.reshape(data, (-1, 4096)).T
    data = data.astype(np.float32)

    print(g, len(bins), start, finish, data.shape)

    rgb = bin_to_rgb(data)

    cv2.imwrite(str(g / preview), rgb[:, :, ::-1])


def main():
    for g in sorted(list(linescans.glob("2025-06*"))): # yamanote
    # for g in sorted(list(linescans.glob("2024-09-10-06-00-05*"))): # yamanote
    # for g in sorted(list(linescans.glob("2024-09-19-03-33-57*"))): # fuxing
    # for g in sorted(list(linescans.glob("2024-09-14-02-13-13*"))): # hktram
    # for g in sorted(list(linescans.glob("nyc4"))):
        if not g.is_dir():
            continue

        print(g)
        if (g / preview).exists():
            ...
        try:
            process_preview(g)
        except KeyboardInterrupt:
            break
        """
        except Exception as e:
            print(f"oh no! {g} {e}")
            continue  # anyway
        """


if __name__ == "__main__":
    main()
