"""Small color and display helpers shared by Nectar's Python tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


# Alkeria NECTA User Guide v2.4, figs. 3.2–3.3. Integrate QE × Bayer
# transmission × wavelength (photons per unit energy) under D65 over the
# published 400–700 nm range. The native neutral is relative to green.
CAMERA_D65_NEUTRAL = np.array([1.16927, 1.0, 0.72297], dtype=np.float64)
# Least-squares fit to BabelColor Average patch reflectances and CIE 1931
# XYZ, at 5 nm intervals. Weight squared errors inversely by patch brightness
# and constrain each row sum to one. This is a nominal visible-light profile.
COLOR_CORRECTION = np.array(
    [
        [1.00645, 0.35265, -0.35910],
        [-0.37271, 1.68434, -0.31163],
        [0.18233, -0.92823, 1.74590],
    ],
    dtype=np.float32,
)
COLOR_TRANSFORM_BLOCK_ROWS = 64


@dataclass(frozen=True)
class NeutralReference:
    """Relative channel offsets and the native response to a neutral surface."""

    neutral: np.ndarray
    lift: np.ndarray
    shadow_chroma_threshold: float = 0.0
    shadow_range: tuple[float, float] | None = None

    @classmethod
    def from_patches(cls, dark, light, *, shadow_chroma_threshold=0.0) -> NeutralReference:
        dark = np.asarray(dark, dtype=np.float64)
        light = np.asarray(light, dtype=np.float64)
        if (
            dark.shape != (3,)
            or light.shape != (3,)
            or not np.all(np.isfinite(dark) & np.isfinite(light))
            or np.any(dark <= 0)
            or np.any(light >= 65535)
            or np.any(light <= dark)
        ):
            raise ValueError("neutral patches need three increasing, unclipped RGB values")
        delta = light - dark
        return cls._from_neutral(dark, delta, shadow_chroma_threshold)

    @classmethod
    def from_shadow(
        cls, dark, as_shot_neutral, *, shadow_fade_end, shadow_chroma_threshold=0.0
    ) -> NeutralReference:
        """Fit a dark neutral while retaining the established highlight white balance."""
        dark = np.asarray(dark, dtype=np.float64)
        if dark.shape != (3,) or not np.all((dark > 0) & (dark < 65535)):
            raise ValueError("dark_rgb needs three positive, unclipped camera counts")
        return cls._from_neutral(
            dark, as_shot_neutral, shadow_chroma_threshold, float(shadow_fade_end)
        )

    @classmethod
    def _from_neutral(cls, dark, neutral, threshold, fade_end=None):
        neutral = np.asarray(neutral, dtype=np.float64)
        if neutral.shape != (3,) or not np.all(np.isfinite(neutral) & (neutral > 0)):
            raise ValueError("neutral response needs three finite positive values")
        neutral = neutral / neutral[1]
        threshold = float(threshold)
        if not 0 <= threshold <= 65535:
            raise ValueError("shadow_chroma_threshold must be between 0 and 65535 counts")
        # Choose the earliest measured channel onset as the common zero.
        # This fixes relative offsets without subtracting surviving shadows.
        anchor = float(np.max(dark / neutral))
        lift = np.maximum(neutral * anchor - dark, 0)
        shadow_range = None
        if fade_end is not None:
            if not anchor < fade_end <= 65535 or threshold > fade_end:
                raise ValueError(
                    "shadow_fade_end must exceed the dark reference and cover the chroma taper"
                )
            shadow_range = (anchor, fade_end)
        return cls(neutral.astype(np.float32), lift.astype(np.float32), threshold, shadow_range)

    def correct(self, rgb: np.ndarray) -> None:
        """Correct in camera coordinates, before either export or white balance."""
        for start in range(0, len(rgb), 32):
            block = rgb[start : start + 32]
            level = np.maximum(np.max(block / self.neutral, axis=-1), 0)
            strength = 1.0
            if self.shadow_range is not None:
                anchor, end = self.shadow_range
                t = np.clip((level - anchor) / (end - anchor), 0, 1)
                strength = 1 - t * t * (3 - 2 * t)
            # Below a channel's measured onset its intensity is censored. Cap
            # the correction by the light recorded elsewhere in that pixel.
            # A fully clipped RGB triplet stays black.
            for c in range(3):
                block[..., c] += np.minimum(self.lift[c], level * self.neutral[c]) * strength
            if self.shadow_chroma_threshold:
                # Near the black offsets the surviving signal supports brightness
                # more reliably than channel ratios. Blend toward the fitted
                # neutral at the same mean response, in native camera coordinates.
                balanced = np.maximum(block / self.neutral, 0)
                signal = balanced.max(axis=-1) / self.shadow_chroma_threshold
                np.minimum(signal, 1, out=signal)
                amount = 1 - signal * signal * (3 - 2 * signal)
                gray = balanced.mean(axis=-1, keepdims=True)
                block += (gray * self.neutral - block) * amount[..., None]
        if self.shadow_range is None:
            # A full-range offset also needs headroom at saturation.
            rgb *= np.float32(65535 / (65535 + self.lift.max()))


GREEN_DETAIL_RADIUS = 2


def green_signal_mean(odd, even):
    """Weight aligned measurements by their signal. A lone live tap keeps its value."""
    # Stripe correction can take an almost black measurement below zero.
    # Only nonnegative signal is meaningful as a weight.
    odd = np.maximum(np.asarray(odd, dtype=np.float32), 0)
    even = np.maximum(np.asarray(even, dtype=np.float32), 0)
    total = odd + even
    difference = odd - even
    ratio = np.divide(difference, total, out=np.zeros_like(total), where=total > 0)
    return 0.5 * (total + difference * ratio), ratio


def fuse_aligned_greens(odd, even, *, phase_weights=None):
    """Recover native-row detail after accounting for local phase response.

    A level-dependent offset between green taps can mimic a one-pixel feature
    at every column along a sensor row. Estimate that imbalance independently
    at each height using adjacent aligned exposures. Only the departure from
    this local response supplies higher-frequency detail. A black-clipped tap has
    no leverage on either the base value or the detail.
    """
    green, ratio = green_signal_mean(odd, even)
    typical = median_filter(
        ratio, size=(1, 2 * GREEN_DETAIL_RADIUS + 1), mode="nearest"
    )
    detail = np.maximum(np.minimum(odd, even), 0) * (ratio - typical)
    if phase_weights is not None:
        detail *= phase_weights
    green[1::2] += detail[1::2]
    green[0::2] -= detail[0::2]
    return np.clip(green, 0, 65535, out=green)


def apply_color_transform(
    raw_rgb: np.ndarray,
    white_balance: np.ndarray,
    color_matrix: np.ndarray = COLOR_CORRECTION,
) -> np.ndarray:
    """Apply per-channel white balance followed by a 3x3 color matrix."""
    raw_rgb = np.asarray(raw_rgb)
    if raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3:
        raise ValueError("raw_rgb must be an HxWx3 array")
    wb = np.asarray(white_balance, dtype=np.float32).reshape(1, 1, 3)
    color_matrix = np.asarray(color_matrix)
    output_dtype = np.result_type(raw_rgb.dtype, wb.dtype, color_matrix.dtype)
    transformed = np.empty(raw_rgb.shape, dtype=output_dtype)
    for row_start in range(0, raw_rgb.shape[0], COLOR_TRANSFORM_BLOCK_ROWS):
        row_stop = min(raw_rgb.shape[0], row_start + COLOR_TRANSFORM_BLOCK_ROWS)
        balanced = raw_rgb[row_start:row_stop] * wb
        transformed[row_start:row_stop] = np.tensordot(
            balanced,
            color_matrix.T,
            axes=([2], [0]),
        )
    return transformed


def auto_tone_map_to_uint8(
    rgb: np.ndarray,
    *,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    gamma: float = 2.2,
    statistics: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a simple display-only percentile stretch and gamma curve.

    ``statistics`` may be a representative RGB subset when scanning all of a
    very large image merely to choose two display points would be wasteful.
    The returned image always contains every pixel from ``rgb``.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("automatic tone mapping expects an HxWx3 uint8 image")
    lut = display_tone_lut(
        rgb if statistics is None else statistics,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        gamma=gamma,
    )
    return lut[rgb]


def display_tone_lut(
    statistics: np.ndarray,
    *,
    low_percentile=0.5,
    high_percentile=99.5,
    gamma=2.2,
) -> np.ndarray:
    """Choose one display curve for all windows of a capture."""
    statistics = np.asarray(statistics)
    if (
        statistics.dtype != np.uint8
        or statistics.ndim != 3
        or statistics.shape[2] != 3
        or statistics.size == 0
    ):
        raise ValueError("tone-map statistics must be a non-empty HxWx3 uint8 image")
    black_point, white_point = np.percentile(
        statistics,
        (low_percentile, high_percentile),
    )
    if white_point <= black_point:
        return np.arange(256, dtype=np.uint8)
    lut = np.arange(256, dtype=np.float32)
    lut -= black_point
    lut /= white_point - black_point
    np.clip(lut, 0.0, 1.0, out=lut)
    np.power(lut, 1.0 / gamma, out=lut)
    return np.rint(lut * 255.0).astype(np.uint8)
