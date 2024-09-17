#!/usr/bin/env python
import cv2
import numpy as np
from pathlib import Path
import time

# import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy

# linescans = Path("/home/dllu/pictures/linescan")
linescans = Path("/home/dllu/pictures/linescan")
preview = "preview.png"


def windowed_cross_correlation(
    green_1: np.ndarray,
    green_2: np.ndarray,
    window_size: int = 256,
    corr_size: int = 11,
) -> np.ndarray:
    """
    green_1: (h, w) where w >> h and w >> window_size
    green_2: (h, w)
    """

    xs = np.arange(0, green_1.shape[1], window_size)
    ys = []
    corr_half = corr_size // 2
    for x in xs:
        window_2 = green_2[:, x + corr_half : x + window_size - corr_half]

        corrs = np.zeros((corr_size,))
        for corr_ind in range(corr_size):
            corr_x = corr_ind - corr_half
            window_1 = green_1[
                :,
                x + corr_half + corr_x : x + window_size - corr_half + corr_x,
            ]
            corr_value = np.sum(window_1 * window_2)
            corrs[corr_ind] = corr_value

        peak_idx = np.argmax(corrs)
        x = np.array([peak_idx - 1, peak_idx, peak_idx + 1])
        y = corrs[peak_idx - 1 : peak_idx + 2]
        coeffs = np.polyfit(x, y, 2)

        # The vertex of the parabola (peak of the quadratic fit) is at -b/(2a)
        refined_peak_idx = -coeffs[1] / (2 * coeffs[0])

        ys.append(refined_peak_idx - corr_half)

    # plt.plot(xs, ys)
    # plt.show()


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

    # windowed_cross_correlation(raw_green_1, raw_green_2)

    raw_green = (raw_green_1 + raw_green_2) / 2

    raw_red = calibrate_black_point(raw_red)
    raw_green = calibrate_black_point(raw_green)
    raw_blue = calibrate_black_point(raw_blue)

    blue = 1.7 * (raw_blue - 0.2 * raw_red - 0.2 * raw_green)
    green = raw_green - 0.3 * raw_red - 0.3 * raw_blue
    red = 0.7 * (raw_red - 0.3 * raw_green - 0.3 * raw_blue)

    rgb = np.concatenate(
        (np.expand_dims(red, 2), np.expand_dims(green, 2), np.expand_dims(blue, 2)),
        axis=2,
    )

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
    top: int = 256,
    bottom: int = 256,
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


def process_preview(g: Path, padding: int = 2, max_chunks: int = 96):
    bins = sorted(list(g.glob("*.bin")))

    dat0 = np.fromfile(bins[0], dtype=np.uint16).astype(np.float32)

    start = np.inf
    finish = 0
    for i, bin in enumerate(bins):
        dat1 = np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        score = np.sum(np.abs(dat0 - dat1) / np.max(dat0) > 0.2) / dat0.shape[0]
        if score > 0.1:
            start = min(start, max(i - padding, 0))
            finish = max(finish, min(i + padding, len(bins)))
        if finish - start >= max_chunks:
            break
    if start == np.inf:
        start = 0

    all_data = [
        np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        for bin in bins[start : finish + 1]
    ]
    data = np.concatenate(all_data)
    data = np.reshape(data, (-1, 4096)).T
    data = data.astype(np.float32)

    print(g, len(bins), start, finish, data.shape)

    rgb = bin_to_rgb(data)

    cv2.imwrite(str(g / preview), rgb[:, :, ::-1])


def main():
    # g = sorted(list(linescans.glob("2024-09*")))[-1]
    # process_preview(g)
    # return
    for g in sorted(list(linescans.glob("2024-09-*"))):
        if not g.is_dir():
            continue

        print(g)
        if (g / preview).exists():
            # continue
            ...
        try:
            process_preview(g)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"oh no! {g} {e}")
            continue  # anyway


if __name__ == "__main__":
    main()
