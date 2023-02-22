#!/usr/bin/env python
import cv2
import numpy as np
from pathlib import Path

linescans = Path("/home/dllu/pictures/linescan")
preview = "preview.png"


def process_preview(g: Path):
    bins = sorted(list(g.glob("*.bin")))[:100]
    all_data = [np.fromfile(bin, dtype=np.uint16).astype(np.float32) for bin in bins]

    start = np.inf
    finish = 0
    for i, (dat0, dat1) in enumerate(zip(all_data[:-1], all_data[1:])):
        score = np.max(np.abs(dat0 - dat1)) / np.max(dat0)
        if score > 0.3:
            start = min(start, i)
            finish = max(finish, i + 1)
    data = np.concatenate(all_data[start : finish + 1])
    data = np.reshape(data, (-1, 4096)).T[:, :32768]
    data = data.astype(np.float32)
    print(g, len(bins), data.shape)

    red = data[0::2, 1::2]
    green = (data[1::2, 1::2] + data[0::2, 0::2]) / 2
    blue = data[1::2, 0::2]

    rgb = np.concatenate(
        (np.expand_dims(red, 2), np.expand_dims(green, 2), np.expand_dims(blue, 2)),
        axis=2,
    )

    rgb_blurred = cv2.GaussianBlur(rgb, (0, 0), 1)
    rgb = rgb - 0.15 * rgb_blurred
    for c in range(3):
        rgb[:, :, c] -= np.min(rgb[:, :, c])
        rgb[:, :, c] /= np.max(rgb[:, :, c])
    rgb = rgb[::-1, :, :]
    rgb = np.sqrt(rgb)
    rgb *= 255

    cv2.imwrite(str(g / preview), rgb[:, :, ::-1])


def main():
    for g in sorted(list(linescans.glob("*"))):
        if not g.is_dir():
            continue

        if (g / preview).exists():
            continue
        process_preview(g)


if __name__ == "__main__":
    main()
