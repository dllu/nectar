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
        if score > 0.4:
            start = min(start, i)
            finish = max(finish, i + 1)
    data = np.concatenate(all_data[start : finish + 1])
    data = np.reshape(data, (-1, 4096)).T[:, :32768]
    data = data.astype(np.float32)
    print(g, len(bins), data.shape)

    raw_red = data[0::2, 1::2]
    raw_green = (data[1::2, 1::2] + data[0::2, 0::2]) / 2
    raw_blue = data[1::2, 0::2]

    blue = 1.7 * (raw_blue - 0.2 * raw_red - 0.2 * raw_green)
    green = raw_green - 0.3 * raw_red - 0.3 * raw_blue
    red = 0.75 * (raw_red - 0.3 * raw_green - 0.3 * raw_blue)


    rgb = np.concatenate(
        (np.expand_dims(red, 2), np.expand_dims(green, 2), np.expand_dims(blue, 2)),
        axis=2,
    )

    rgb_blurred = cv2.GaussianBlur(rgb, (0, 0), 1)
    rgb = rgb - 0.2 * rgb_blurred
    rgb -= np.percentile(rgb, 2)
    rgb += 0.1
    rgb /= np.percentile(rgb, 95) * 0.9
    rgb[rgb < 0] = 0
    rgb[rgb > 1] = 1
    #for c in range(3):
        #rgb[:, :, c] -= np.min(rgb[:, :, c])
        #rgb[:, :, c] /= np.max(rgb[:, :, c])
    rgb = rgb[::-1, :, :]
    rgb = np.sqrt(rgb)
    rgb *= 255

    cv2.imwrite(str(g / preview), rgb[:, :, ::-1])


def main():
    for g in sorted(list(linescans.glob("*"))):
        if not g.is_dir():
            continue

        if (g / preview).exists():
            #continue
            ...
        process_preview(g)


if __name__ == "__main__":
    main()
