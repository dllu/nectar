#!/usr/bin/env python
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# linescans = Path("/home/dllu/pictures/linescan")
linescans = Path("/mnt/data14/pictures/linescan")
preview = "preview_2.png"


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

    plt.plot(xs, ys)
    plt.show()


def bin_to_rgb(data: np.ndarray) -> np.ndarray:
    """
    data: mosaiced rggb bayer array
    """
    raw_red = data[0::2, 1::2]
    raw_green_1 = data[1::2, 1::2]
    raw_green_2 = data[0::2, 0::2]

    windowed_cross_correlation(raw_green_1, raw_green_2)

    raw_green = (raw_green_1 + raw_green_2) / 2
    raw_blue = data[1::2, 0::2]

    blue = 1.7 * (raw_blue - 0.2 * raw_red - 0.2 * raw_green)
    green = raw_green - 0.3 * raw_red - 0.3 * raw_blue
    red = 0.75 * (raw_red - 0.3 * raw_green - 0.3 * raw_blue)

    rgb = np.concatenate(
        (np.expand_dims(red, 2), np.expand_dims(green, 2), np.expand_dims(blue, 2)),
        axis=2,
    )

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


def process_preview(g: Path):
    bins = sorted(list(g.glob("*.bin")))[:100]
    all_data = [np.fromfile(bin, dtype=np.uint16).astype(np.float32) for bin in bins]

    start = np.inf
    finish = 0
    for i, (dat0, dat1) in enumerate(zip(all_data[:-1], all_data[1:])):
        score = np.max(np.abs(dat0 - dat1)) / np.max(dat0)
        if score > 0.4:
            start = min(start, i)
            finish = max(finish, i + 1)
    data = np.concatenate(all_data[start : finish + 1])
    data = np.reshape(data, (-1, 4096)).T[:, :32768]
    data = data.astype(np.float32)
    print(g, len(bins), data.shape)

    rgb = bin_to_rgb(data)

    cv2.imwrite(str(g / preview), rgb[:, :, ::-1])


def main():
    for g in sorted(list(linescans.glob("2024*"))):
        if not g.is_dir():
            continue

        if (g / preview).exists():
            # continue
            ...
        process_preview(g)
        break


if __name__ == "__main__":
    main()
