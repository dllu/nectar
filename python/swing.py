#!/usr/bin/env python3
"""One-off polar reconstruction of the 2026-09-03 child-on-a-swing capture.

    uv run python python/swing.py

Time coordinates are Bayer-binned columns, as in speed_constraints.py. Spatial
coordinates (including the pivot) are full sensor pixels, with row zero at the
top. The manually located passages below deliberately exclude the other swing
from the motion fit. A single pivot and one signed angular speed per passage
are fitted from green-plane correlations on the chains and seat, not the legs.

The angular speed is approximately cosine-shaped around each passage; its
integral supplies the polar angle. Only the short, monotonic interval around
each crossing is rendered. --motion constant uses the local constant-speed
approximation instead. No speed is inferred from stationary background.
Separate framing annotations center the child within a shared 0.56-radian
angular span. A small configurable visual pivot adjustment is applied before
refitting the per-passage amplitudes; the original fitted pivot is recorded.

Outputs are RGBA PNG wedges, camera-linear 16-bit RGB DNGs, a contact sheet,
an annotated uncorrected overview, and a JSON/plot record of the fit. The DNGs
are demosaiced linear DNGs, not original Bayer mosaics; white balance and the
display tone curve are not baked into them. Outside the measured annular
sector, PNG alpha is zero and DNG pixels are black. The original raw capture
and the train/panorama processing code are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares

from image_processing import apply_color_transform, auto_tone_map_to_uint8
from preview import (
    bayer_channel_frame_geometry,
    parse_capture_datetime,
    parse_exposure_time,
    raw_channels,
    windowed_cross_correlation,
    write_linear_dng,
)
from raw_io import load_raw_frame


DEFAULT_CAPTURE = Path("/mnt/dataz/pictures/linescan/2026-09-03-00-31-12")
SENSOR_ROWS = 4096
# Visually annotated on the uncorrected, upright, Bayer-binned time strip.
# These identify the foreground child, including where the other swing overlaps.
PASS_CENTERS = (
    1200, 4040, 7030, 10060, 13060, 16070, 19010, 21990,
    24920, 27990, 31020, 34040, 37140, 40140, 43200,
)
# Framing is independent of the speed/phase annotations above. These centers
# were inspected against the child's complete head/seat/feet silhouette, rather
# than the chains or the overlapping background swing. Adjusting a crop must
# not change the motion estimate or pretend its center is the speed maximum.
FRAMING_CENTERS = (
    1190, 3970, 7050, 10000, 13090, 16030, 19110, 21865,
    24950, 27960, 31090, 34060, 37150, 40170, 43220,
)
DEFAULT_ANGLE_WIDTH = 0.56  # radians, identical for all passages
DEFAULT_PIVOT_OFFSET = 200.0  # modest visual correction downward from the fit


class Capture:
    def __init__(self, directory: Path):
        self.directory = directory.expanduser().resolve()
        self.paths = sorted(
            self.directory.glob("*.bin"),
            key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)],
        )
        if not self.paths:
            raise ValueError(f"no raw files in {self.directory}")
        self.starts, widths = bayer_channel_frame_geometry(self.paths)
        if np.any(widths == 0):
            raise ValueError("empty raw chunks are not supported")
        self.width = int(self.starts[-1])

    def read(self, start: int, stop: int) -> np.ndarray:
        """Read only intersecting files; handle windows spanning chunk boundaries."""
        if not 0 <= start < stop <= self.width:
            raise ValueError(f"source interval [{start}, {stop}) exceeds the capture")
        first = int(np.searchsorted(self.starts, start, side="right") - 1)
        last = int(np.searchsorted(self.starts, stop - 1, side="right") - 1)
        result = np.empty((SENSOR_ROWS, 2 * (stop - start)), dtype=np.uint16)
        for i in range(first, last + 1):
            raw = load_raw_frame(self.paths[i], rows=SENSOR_ROWS)
            a, b = max(start, int(self.starts[i])), min(stop, int(self.starts[i + 1]))
            result[:, 2 * (a - start):2 * (b - start)] = raw[
                :, 2 * (a - self.starts[i]):2 * (b - self.starts[i])
            ]
        return result


@dataclass(frozen=True)
class Passage:
    center: float
    omega: float  # signed peak radians / Bayer-binned time column
    period: float
    half_window: float
    motion: str = "sinusoidal"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (
            self.center, self.omega, self.period, self.half_window
        )):
            raise ValueError("passage parameters must be finite")
        if self.omega == 0 or self.period <= 0 or self.half_window <= 0:
            raise ValueError("passage needs nonzero speed and positive period/window")
        if self.motion not in ("sinusoidal", "constant"):
            raise ValueError("unknown motion model")
        if self.half_window >= self.period / 4:
            raise ValueError("passage window must stay inside one monotonic half-swing")
        if abs(float(self.angle(self.center + self.half_window))) >= math.pi / 2:
            raise ValueError("wedge must be narrower than 180 degrees")

    def angle(self, time):
        dt = np.asarray(time) - self.center
        if self.motion == "constant":
            return self.omega * dt
        frequency = 2 * np.pi / self.period
        return self.omega * np.sin(frequency * dt) / frequency

    def time(self, angle):
        if self.motion == "constant":
            return self.center + np.asarray(angle) / self.omega
        frequency = 2 * np.pi / self.period
        # Clipping only makes out-of-sector queries finite. The renderer masks
        # them by angle and radius, so they cannot become repeated edge pixels.
        phase = np.clip(np.asarray(angle) * frequency / self.omega, -1, 1)
        return self.center + np.arcsin(phase) / frequency


def frame_passage(
    center: float, omega: float, period: float, motion: str,
    framing_center: float, angle_width: float,
) -> tuple[Passage, float, tuple[float, float]]:
    """Choose a centered angular crop without shifting the fitted motion phase."""
    if not math.isfinite(angle_width) or not 0 < angle_width < math.pi:
        raise ValueError("angle width must be finite and between zero and pi radians")
    if not math.isfinite(framing_center) or abs(framing_center - center) >= period / 4:
        raise ValueError("framing center must be inside the fitted monotonic passage")
    # Start with a tiny valid window; replace it with the required time extent
    # after inverting the two angular crop boundaries.
    model = Passage(center, omega, period, min(1.0, period / 8), motion)
    offset = float(model.angle(framing_center))
    limits = offset + np.array([-0.5, 0.5]) * angle_width
    if motion == "sinusoidal" and np.max(np.abs(limits)) >= abs(omega) * period / (2 * np.pi):
        raise ValueError("angular crop reaches a swing reversal; reduce --angle-width")
    start, stop = sorted(map(float, model.time(limits)))
    half_window = max(center - start, stop - center)
    passage = Passage(center, omega, period, half_window, motion)
    return passage, offset, (start, stop)


def measure_motion(capture: Capture, centers: np.ndarray) -> np.ndarray:
    """Rows: pass index, full-resolution sensor y, time, signed lag, confidence."""
    measurements = []
    for i, center in enumerate(centers):
        start, stop = int(center) - 384, int(center) + 384
        _, g1, g2, _ = raw_channels(capture.read(start, stop))
        # The top 2560 sensor rows cover rigid chains and seat. The child's legs
        # bend independently, and the lower background contains another child.
        for y in range(0, 1280, 256):
            xs, lags, weights = windowed_cross_correlation(
                g1[y:y + 256], g2[y:y + 256],
                window_step=64, window_size=256, corr_size=15,
                show_progress=False,
            )
            valid = (
                (np.abs(xs + start - center) <= 128)
                & (weights >= 0.5) & (np.abs(lags) >= 0.1) & (np.abs(lags) <= 3)
            )
            # Both greens are vertically registered halfway between their rows
            # inside windowed_cross_correlation, hence the extra half raw pixel.
            sensor_y = 2 * (y + 127.5) + 0.5
            measurements.extend(
                (i, sensor_y, x + start, lag, weight)
                for x, lag, weight in zip(xs[valid], lags[valid], weights[valid])
            )
        print(f"Measured passage {i + 1:02d}/{len(centers)} at column {center:g}", flush=True)
    return np.asarray(measurements, dtype=np.float64).reshape(-1, 5)


def fit_motion(
    measurements: np.ndarray,
    centers: np.ndarray,
    period: float,
    motion: str,
    pivot_y: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit lag = 1 / (omega * (y - pivot_y)), robustly in log magnitude.

    The two green scan planes are separated by one spatial sensor pixel, so
    their temporal lag measures reciprocal tangential speed. Fitting the radius
    dependence jointly supplies one stationary pivot, while each pass can have
    a different amplitude. Overlapping correlation windows are not independent;
    residuals are diagnostics, not formal uncertainty estimates.
    """
    if measurements.ndim != 2 or measurements.shape[1] != 5 or not len(measurements):
        raise ValueError("no usable foreground motion correlations")
    ids = measurements[:, 0].astype(int)
    y, times, lags, weights = measurements[:, 1:].T
    if (not np.all(np.isfinite(measurements)) or np.any(lags == 0)
            or np.any(weights <= 0) or np.any(ids < 0) or np.any(ids >= len(centers))):
        raise ValueError("invalid motion correlations")
    directions = []
    for i in range(len(centers)):
        selected = ids == i
        if selected.sum() < 8 or np.unique(y[selected]).size < 3:
            raise ValueError(f"passage {i + 1} has insufficient evidence to fit the pivot")
        directions.append(1 if np.sum(weights[selected] * np.sign(lags[selected])) > 0 else -1)
    directions = np.asarray(directions)
    if np.any(directions[1:] == directions[:-1]):
        raise ValueError("correlated directions do not alternate; check the passage annotations")
    keep = np.sign(lags) == directions[ids]
    temporal = np.ones(len(times))
    if motion == "sinusoidal":
        temporal = np.cos(2 * np.pi * (times - centers[ids]) / period)
    if np.any(temporal <= 0):
        raise ValueError("correlations extend beyond the monotonic passage")

    def unpack(q):
        if pivot_y is None:
            return -np.exp(q[0]), np.exp(q[1:])
        return pivot_y, np.exp(q)

    def residual(q):
        pivot, rates = unpack(q)
        return np.log(np.abs(lags) * rates[ids] * (y - pivot) * temporal)

    initial = np.log([0.0007] * len(centers))
    if pivot_y is None:
        initial = np.r_[np.log(1600), initial]
    bounds_low = np.full(len(initial), np.log(1e-6))
    bounds_high = np.full(len(initial), np.log(0.1))
    if pivot_y is None:
        bounds_low[0], bounds_high[0] = np.log(100), np.log(20000)
    fitted = least_squares(
        lambda q: residual(q)[keep] * np.sqrt(weights[keep]), initial,
        bounds=(bounds_low, bounds_high), loss="soft_l1", f_scale=0.025,
    )
    if not fitted.success:
        raise ValueError(f"motion fit failed: {fitted.message}")
    pivot, rates = unpack(fitted.x)
    errors = residual(fitted.x)
    for i in range(len(centers)):
        if np.median(np.abs(errors[(ids == i) & keep])) > 0.15:
            raise ValueError(f"passage {i + 1} does not support a rigid swing model")
    return float(pivot), rates * directions, errors


def render_wedge(
    raw: np.ndarray, source_start: int, passage: Passage, pivot_y: float,
    *, scale: float = 1.0, half_angle: float | None = None, center_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inverse-map an annular sector, sampling each Bayer plane separately.

    x = -radius*sin(angle), y = pivot_y + radius*cos(angle). The minus sign
    matches this capture's camera mounting and keeps alternate passes in the
    same physical orientation. A stationary source row becomes a circular arc.
    Each output pixel is one full sensor pixel at scale=1; source data remain
    16-bit until interpolation, and are quantized to uint16 only at the end.
    """
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    if not math.isfinite(pivot_y) or pivot_y >= 0:
        raise ValueError("pivot must be above sensor row zero")
    height = raw.shape[0]
    r_min, r_max = -pivot_y, height - 1 - pivot_y
    if half_angle is None:
        half_angle = abs(float(passage.angle(passage.center + passage.half_window)))
    if not math.isfinite(half_angle) or not 0 < half_angle < math.pi / 2:
        raise ValueError("wedge half-angle must be between zero and pi/2")
    if not math.isfinite(center_angle):
        raise ValueError("framing angle must be finite")
    half_width = math.ceil(r_max * math.sin(half_angle) * scale)
    top = math.floor((pivot_y + r_min * math.cos(half_angle)) * scale) / scale
    out_height = math.ceil((height - 1 - top) * scale) + 1
    out_width = 2 * half_width + 1
    if out_height * out_width > 100_000_000:
        raise ValueError("wedge exceeds 100 megapixels; reduce --half-window or --scale")
    result = np.zeros((out_height, out_width, 3), dtype=np.uint16)
    alpha = np.zeros((out_height, out_width), dtype=np.uint8)
    planes = [np.ascontiguousarray(p, dtype=np.float32) for p in raw_channels(raw)]
    x = (np.arange(out_width, dtype=np.float32) - half_width) / scale
    # At reduced resolution, prefilter each plane to avoid aliasing in previews.
    if scale < 1:
        sigma_y = 0.5 * math.sqrt(scale**-2 - 1) / 2
        speed_fraction = (math.cos(2 * np.pi * passage.half_window / passage.period)
                          if passage.motion == "sinusoidal" else 1.0)
        max_lag = 1 / (abs(passage.omega) * r_min * speed_fraction)
        sigma_x = 0.5 * math.sqrt(scale**-2 - 1) * max_lag
        planes = [cv2.GaussianBlur(p, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y) for p in planes]

    for a in range(0, out_height, 128):
        b = min(a + 128, out_height)
        y = top + np.arange(a, b, dtype=np.float32)[:, None] / scale
        radius = np.hypot(x[None, :], y - pivot_y)
        angle = -np.arctan2(x[None, :], y - pivot_y)
        sensor_y = pivot_y + radius
        source_angle = angle + center_angle
        t0 = passage.time(source_angle) - source_start
        # R and G1 are on the other scan plane. Its angular separation is
        # approximately 1/r radians; invert the motion law at that angle,
        # rather than shifting a finished RGB image by a fixed time column.
        t1 = passage.time(source_angle - 1 / radius) - source_start
        valid = (
            (radius >= r_min) & (radius <= r_max) & (np.abs(angle) <= half_angle)
            & (t0 >= 0) & (t0 <= planes[0].shape[1] - 1)
            & (t1 >= 0) & (t1 <= planes[0].shape[1] - 1)
        )
        rgb = np.empty((b - a, out_width, 3), dtype=np.float32)
        for plane, (row_offset, time, channel) in zip(
            planes, ((0, t1, 0), (1, t1, 1), (0, t0, 1), (1, t0, 2))
        ):
            samples = cv2.remap(
                plane, time.astype(np.float32), ((sensor_y - row_offset) / 2).astype(np.float32),
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            if channel == 1 and row_offset == 0:
                rgb[:, :, 1] = (rgb[:, :, 1] + samples) * 0.5
            else:
                rgb[:, :, channel] = samples
        rgb[~valid] = 0
        result[a:b] = np.rint(np.clip(rgb, 0, 65535)).astype(np.uint16)
        alpha[a:b] = valid.astype(np.uint8) * 255
    geometry = {
        "width": out_width, "height": out_height, "scale": scale,
        "pivot_in_output_pixels": [half_width, (pivot_y - top) * scale],
        "radius_sensor_pixels": [r_min, r_max],
        "angle_width_radians": 2 * half_angle,
        "framing_angle_offset_radians": center_angle,
        "half_angle_degrees": math.degrees(half_angle),
        "source_start_column": source_start,
    }
    return result, alpha, geometry


def display_rgb(raw_rgb: np.ndarray, white_balance: np.ndarray, white: float) -> np.ndarray:
    rgb = apply_color_transform(raw_rgb, white_balance)
    np.clip(rgb / white, 0, 1, out=rgb)
    np.power(rgb, 1 / 2.2, out=rgb)
    return np.rint(rgb * 255).astype(np.uint8)


def write_overview(capture: Capture, centers: np.ndarray, output: Path) -> None:
    """Bayer-cell preview with no speed compensation, like speed_constraints.py."""
    # One global area resize keeps time coordinates accurate for variable chunks.
    small = np.empty((512, capture.width, 3), dtype=np.uint8)
    for i, path in enumerate(capture.paths):
        r, g1, g2, b = raw_channels(load_raw_frame(path, rows=SENSOR_ROWS))
        rgb = np.stack((r, (g1.astype(np.float32) + g2) / 2, b), axis=2)
        small[:, capture.starts[i]:capture.starts[i + 1]] = (
            cv2.resize(rgb, (r.shape[1], 512), interpolation=cv2.INTER_AREA) / 256
        ).astype(np.uint8)
    small = cv2.resize(small, (math.ceil(capture.width / 4), 512), interpolation=cv2.INTER_AREA)
    small = auto_tone_map_to_uint8(small)
    strip_width, strip_height = 2240, 540
    count = math.ceil(small.shape[1] / strip_width)
    sheet = Image.new("RGB", (strip_width, count * strip_height), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i in range(count):
        left = i * strip_width
        sheet.paste(Image.fromarray(small[:, left:left + strip_width]), (0, i * strip_height + 28))
        draw.text((8, i * strip_height + 7), f"Uncorrected time columns {left * 4} to {min((left + strip_width) * 4, capture.width)}", fill="white")
        for j, center in enumerate(centers):
            x = round(center * small.shape[1] / capture.width - left)
            if 0 <= x < strip_width:
                draw.line((x, i * strip_height + 28, x, (i + 1) * strip_height), fill="#ffaa44", width=1)
                draw.text((x + 4, i * strip_height + 35), f"{j + 1:02d}", fill="#ffaa44")
    sheet.save(output / "uncorrected_overview.jpg", quality=94)


def write_fit_plot(measurements, errors, centers, rates, pivot, period, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    y = np.linspace(0, 2560, 200)
    for i, rate in enumerate(rates):
        selected = measurements[:, 0] == i
        m = measurements[selected]
        axes[0].scatter(m[:, 1], 1 / np.abs(m[:, 3]), s=8, alpha=0.4)
        axes[0].plot(y, abs(rate) * (y - pivot), linewidth=1)
    axes[0].set(xlabel="Sensor row (full pixels)", ylabel="Tangential speed (pixels / time column)",
                title=f"One fixed pivot: y = {pivot:.0f} pixels; median lag residual {np.median(np.abs(errors)):.1%}")
    axes[1].plot(centers, rates, "o-")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set(xlabel="Time (Bayer-binned columns)", ylabel="Signed peak angular speed (radians / column)",
                title=f"Separate amplitude for each passage; swing period = {period:g} columns")
    figure.savefig(output / "motion_fit.png", dpi=150)
    plt.close(figure)


def process(args) -> Path:
    capture = Capture(args.capture)
    centers = np.asarray(args.centers or PASS_CENTERS, dtype=float)
    if len(centers) < 3 or not np.all(np.isfinite(centers)) or np.any(np.diff(centers) <= 0):
        raise ValueError("at least three increasing, finite passage centers are required")
    if args.centers is None and capture.width != 44800:
        raise ValueError("default annotations require this 44800-column capture; supply --centers")
    framing_centers = np.asarray(
        args.framing_centers if args.framing_centers is not None
        else (FRAMING_CENTERS if args.centers is None else centers), dtype=float,
    )
    if framing_centers.shape != centers.shape or not np.all(np.isfinite(framing_centers)):
        raise ValueError("provide one finite framing center for each motion center")
    period = args.period if args.period is not None else float(np.median(centers[2:] - centers[:-2]))
    if not math.isfinite(period) or period <= 4 * 384:
        raise ValueError("period must exceed four times the fitting half-window")
    if args.half_window is not None and (
        not math.isfinite(args.half_window) or not 0 < args.half_window < period / 4
    ):
        raise ValueError("half-window must be positive, finite, and below a quarter-period")
    if args.angle_width is not None and (
        not math.isfinite(args.angle_width) or not 0 < args.angle_width < math.pi
    ):
        raise ValueError("angle width must be finite and between zero and pi radians")
    if not math.isfinite(args.scale) or not 0 < args.scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    if args.contact_height < 64:
        raise ValueError("contact height must be at least 64 pixels")
    if not math.isfinite(args.pivot_offset):
        raise ValueError("pivot offset must be finite")
    margin = 400
    if centers[0] - margin < 0 or centers[-1] + margin >= capture.width:
        raise ValueError("passage windows must fit inside the capture")
    if args.pivot_y is not None and (not math.isfinite(args.pivot_y) or args.pivot_y >= 0):
        raise ValueError("pivot-y must be finite and negative (above the image)")
    selected = args.passes or list(range(1, len(centers) + 1))
    if len(set(selected)) != len(selected) or any(i < 1 or i > len(centers) for i in selected):
        raise ValueError("--passes must contain distinct valid 1-based passage numbers")
    output = (args.output_dir or capture.directory / "swing_processed").expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}; choose a new --output-dir")
    output.parent.mkdir(parents=True, exist_ok=True)

    measurements = measure_motion(capture, centers)
    fitted_pivot, rates, errors = fit_motion(measurements, centers, period, args.motion)
    pivot = args.pivot_y if args.pivot_y is not None else fitted_pivot + args.pivot_offset
    if pivot >= 0:
        raise ValueError("adjusted pivot must remain above sensor row zero")
    if pivot != fitted_pivot:
        pivot, rates, errors = fit_motion(measurements, centers, period, args.motion, pivot)
    angle_width = args.angle_width if args.angle_width is not None else DEFAULT_ANGLE_WIDTH
    if args.half_window is not None:
        # Backward-compatible time-window control selects ONE common angle
        # using the median passage amplitude, not a different width per pass.
        reference = Passage(0, float(np.median(np.abs(rates))), period, args.half_window, args.motion)
        angle_width = 2 * float(reference.angle(args.half_window))
    framings = [
        frame_passage(c, rate, period, args.motion, frame_center, angle_width)
        for c, rate, frame_center in zip(centers, rates, framing_centers)
    ]
    print(f"Correlation pivot: y={fitted_pivot:.1f}; applied vertical offset: {pivot - fitted_pivot:+.1f} pixels", flush=True)
    print(f"Fixed pivot: y={pivot:.1f} sensor pixels; period={period:g} Bayer-binned columns", flush=True)
    print(f"Median relative lag residual: {np.median(np.abs(np.expm1(errors))):.2%}", flush=True)
    print(f"Common angular span: {angle_width:.6f} radians ({math.degrees(angle_width):.2f} degrees)", flush=True)

    # A neutral-looking stationary walkway band in this particular capture.
    # One white balance and exposure are shared by all passages; neither uses
    # the black corners of the wedges, nor affects DNG pixel values.
    r, g1, g2, b = raw_channels(capture.read(11000, 11256))
    neutral = np.array([p[860:910].mean() for p in (r, (g1.astype(float) + g2) / 2, b)])
    white_balance = neutral[1] / np.maximum(neutral, 1)
    sample = np.stack((r, (g1.astype(float) + g2) / 2, b), axis=2).astype(np.float32)
    white = max(float(np.percentile(apply_color_transform(sample, white_balance), 99.7)), 1.0)
    del r, g1, g2, b, sample

    # Publish a complete run at once, keeping failed runs and existing exports
    # out of the destination. No existing files or editor sidecars are replaced.
    with tempfile.TemporaryDirectory(prefix=".swing-", dir=output.parent) as tmp:
        staging = Path(tmp) / "result"
        staging.mkdir()
        write_overview(capture, centers, staging)
        write_fit_plot(measurements, errors, centers, rates, pivot, period, staging)
        records, thumbs = [], []
        thumb_width = round(args.contact_height * 500 / 640)
        for number in sorted(selected):
            i = number - 1
            passage, center_angle, (time_start, time_stop) = framings[i]
            # Include both green-plane offsets and bilinear interpolation support.
            speed_fraction = (math.cos(2 * np.pi * passage.half_window / period)
                              if args.motion == "sinusoidal" else 1.0)
            min_speed = abs(rates[i]) * (-pivot) * speed_fraction
            padding = math.ceil(1 / min_speed) + 4
            start = math.floor(time_start) - padding
            stop = math.ceil(time_stop) + padding + 1
            raw_rgb, alpha, geometry = render_wedge(
                capture.read(start, stop), start, passage, pivot, scale=args.scale,
                half_angle=angle_width / 2, center_angle=center_angle,
            )
            name = f"swing_{number:02d}"
            # The shared writer flips vertically for the legacy renderer. Our
            # image is already upright, so counter that flip at its boundary.
            write_linear_dng(
                staging / f"{name}.dng", raw_rgb[::-1], white_balance,
                captured_at=parse_capture_datetime(capture.directory),
                exposure_time=parse_exposure_time(capture.directory),
            )
            rgba = np.dstack((display_rgb(raw_rgb, white_balance, white), alpha))
            image = Image.fromarray(rgba)
            image.save(staging / f"{name}.png")
            image.thumbnail((thumb_width, args.contact_height), Image.Resampling.LANCZOS)
            thumbs.append((number, image.copy()))
            records.append({
                "number": number, "center_column": centers[i],
                "framing_center_column": framing_centers[i],
                "signed_angular_speed": rates[i],
                "source_interval_columns": [time_start, time_stop],
                "png": f"{name}.png", "dng": f"{name}.dng", **geometry,
            })
            print(f"Rendered {name}: {geometry['width']} x {geometry['height']} PNG + DNG", flush=True)
            del raw_rgb, rgba, alpha
        padding = max(2, round(args.contact_height / 32))
        cell_width = thumb_width + padding
        cell_height = args.contact_height + 2 * padding
        sheet = Image.new("RGB", (3 * cell_width, math.ceil(len(thumbs) / 3) * cell_height), "#242424")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=max(12, args.contact_height // 64))
        for j, (number, thumb) in enumerate(thumbs):
            x, y = (j % 3) * cell_width, (j // 3) * cell_height
            sheet.paste(thumb, (x + (cell_width - thumb.width) // 2, y + 3 * padding // 2), thumb)
            draw.text((x + padding // 2, y + padding // 4),
                      f"Passage {number:02d} | column {framing_centers[number - 1]:g}", fill="white", font=font)
        sheet.save(staging / "contact_sheet.jpg", quality=96, subsampling=0)
        report = {
            "capture": str(capture.directory), "motion": args.motion,
            "time_unit": "Bayer-binned column (two stored raw columns); no line rate recorded",
            "spatial_unit": "full sensor pixel", "pivot_sensor_y": pivot,
            "correlation_fitted_pivot_sensor_y": fitted_pivot,
            "applied_pivot_offset_pixels": pivot - fitted_pivot,
            "period_columns": period, "angle_width_radians": angle_width,
            "reference_half_window_columns": args.half_window,
            "annotated_centers": centers.tolist(), "signed_angular_speeds": rates.tolist(),
            "framing_centers": framing_centers.tolist(),
            "contact_sheet_size": list(sheet.size),
            "white_balance": white_balance.tolist(), "display_white_level": white,
            "median_absolute_relative_lag_residual": float(np.median(np.abs(np.expm1(errors)))),
            "correlation_columns": ["pass_index_zero_based", "sensor_y", "time_column", "lag", "weight"],
            "correlations": measurements.tolist(), "log_lag_residuals": errors.tolist(),
            "dng": "16-bit camera-linear RGB; demosaiced/resampled; black outside PNG alpha mask",
            "assumptions": [
                "Fixed projected pivot lies on the scanning line above sensor row zero.",
                "Each annotated passage is near peak speed; period estimated between same-direction passes.",
                "Framing centers are independent visual annotations of the head/seat/feet silhouette.",
                "A shared visual pivot adjustment is applied before refitting passage amplitudes.",
                "Chains and seat approximately rotate rigidly; independent limb motion is not corrected.",
                "The other swing is excluded from fitting as far as possible; overlap remains visible.",
            ],
            "outputs": records,
        }
        (staging / "swing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staging.rename(output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path, nargs="?", default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", type=Path, help="new destination directory (default: capture/swing_processed)")
    parser.add_argument("--centers", type=float, nargs="+", help="override annotated passage centers in Bayer-binned columns")
    parser.add_argument("--framing-centers", type=float, nargs="+", help="one crop center per passage, independent of motion phase")
    parser.add_argument("--period", type=float, help="full swing period in Bayer-binned columns (default: annotation spacing)")
    framing = parser.add_mutually_exclusive_group()
    framing.add_argument("--angle-width", type=float, help="common full wedge angle in radians (default: 0.56)")
    framing.add_argument("--half-window", type=float, help="choose the common angle from this time margin at the median passage speed")
    parser.add_argument("--motion", choices=("sinusoidal", "constant"), default="sinusoidal")
    parser.add_argument("--pivot-y", type=float, help="override fitted pivot, in full sensor pixels; must be negative")
    parser.add_argument("--pivot-offset", type=float, default=DEFAULT_PIVOT_OFFSET,
                        help="downward offset from correlation pivot (default: 200); ignored with --pivot-y")
    parser.add_argument("--scale", type=float, default=1, help="output scale, 0 < scale <= 1; use 0.25 for quick previews")
    parser.add_argument("--contact-height", type=int, default=1920, help="maximum thumbnail height in the contact sheet (default: 1920)")
    parser.add_argument("--passes", type=int, nargs="+", help="render selected 1-based passages (fit still uses every passage)")
    args = parser.parse_args(argv)
    try:
        output = process(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Wrote swing reconstruction to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
