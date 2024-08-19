#!/usr/bin/env python
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

linescans = Path("/home/dllu/pictures/linescan")
preview = "preview_3.png"


def calibrate_black_point(data, top: int = 256, bottom: int = 256):
    mean_top = np.mean(data[:top, :], axis=1)
    mean_bottom = np.mean(data[-bottom:, :], axis=1)
    height = data.shape[0]
    rows = np.linspace(0, 1, num=height)
    x = np.concatenate((rows[:top], rows[-bottom:]))
    data2 = np.zeros_like(data)
    for col in tqdm(range(data.shape[1]), desc="calibrating black point"):
        y = np.concatenate(
            (mean_top - data[:top, col], mean_bottom - data[-bottom:, col])
        )
        poly = np.polynomial.Polynomial.fit(x, y, 1)
        data2[:, col] = data[:, col] + poly.linspace(height)[1]
    return data2


def process_preview(g: Path):
    bins = sorted(list(g.glob("*.bin")))

    dat0 = np.fromfile(bins[0], dtype=np.uint16).astype(np.float32)

    start = np.inf
    finish = 0
    for i, bin in enumerate(bins):
        dat1 = np.fromfile(bin, dtype=np.uint16).astype(np.float32)
        score = np.sum(np.abs(dat0 - dat1) / np.max(dat0) > 0.2) / dat0.shape[0]
        if score > 0.1:
            start = min(start, i)
            finish = max(finish, i + 1)
        if finish - start > 50:
            break
    if start == np.inf:
        start = 0

    all_data = [np.fromfile(bin, dtype=np.uint16).astype(np.float32) for bin in bins[start:finish+1]]
    data = np.concatenate(all_data)
    data = np.reshape(data, (-1, 4096)).T[:, :32768]
    data = data.astype(np.float32)

    print(g, len(bins), start, finish, data.shape)

    raw_red = data[0::2, 1::2]
    raw_green = (data[1::2, 1::2] + data[0::2, 0::2]) / 2
    raw_blue = data[1::2, 0::2]

    # raw_red = calibrate_black_point(raw_red)
    # raw_green = calibrate_black_point(raw_green)
    # raw_blue = calibrate_black_point(raw_blue)

    blue = 1.5 * raw_blue
    green = raw_green
    red = 0.8 * raw_red

    rgb = np.concatenate(
        (np.expand_dims(red, 2), np.expand_dims(green, 2), np.expand_dims(blue, 2)),
        axis=2,
    )

    # print(np.percentile(rgb, 99), np.max(rgb), np.percentile(rgb, 0.1), np.min(rgb))
    rgb_blurred = cv2.GaussianBlur(rgb, (0, 0), 1)
    rgb = rgb - 0.2 * rgb_blurred
    rgb -= np.percentile(rgb, 0.1)
    rgb /= np.percentile(rgb, 99) * 0.99
    # print(np.percentile(rgb, 99), np.max(rgb), np.percentile(rgb, 0.1), np.min(rgb))
    rgb[rgb < 0] = 0
    rgb[rgb > 1] = 1
    # for c in range(3):
    # rgb[:, :, c] -= np.min(rgb[:, :, c])
    # rgb[:, :, c] /= np.max(rgb[:, :, c])
    rgb = rgb[::-1, :, :]
    rgb = np.sqrt(rgb)
    rgb *= 255

    cv2.imwrite(str(g / preview), rgb[:, :, ::-1])


def main():
    # process_preview(linescans / "2024-01-03-07:01:32")
    # process_preview (linescans / "2024-08-18-23:16:45")
    for g in sorted(list(linescans.glob("2024-08*"))):
        if not g.is_dir():
            continue

        print(g)
        if (g / preview).exists():
            ...
            # continue
        try:
            process_preview(g)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"oh no! {g} {e}")
            continue


if __name__ == "__main__":
    main()
