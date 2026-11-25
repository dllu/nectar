"""Measure camera pitch from stationary ground and reconstruct Bayer rows."""

import numpy as np
from scipy.ndimage import gaussian_filter1d


DEFAULT_GROUND_ROWS = 128  # Full-resolution sensor pixels.
MIN_GROUND_ROWS = 32       # At least 16 samples on a single Bayer plane.
_PROFILE_COLUMNS = 4       # Simultaneous Bayer column pairs per measurement.
_BLOCK_COLUMNS = 4096
_BASELINE_SIGMA = 256      # In capture-time columns, independent of train speed.


def _ground_motion(profiles):
    """Register each ground profile to a common reference in this time window."""
    # Bandpass along the sensor to suppress read noise and illumination slopes.
    detail = gaussian_filter1d(profiles, 0.8, axis=1) - gaussian_filter1d(
        profiles, 5, axis=1
    )
    radius = min(12, detail.shape[1] // 4)
    template = np.median(detail, axis=0)[radius:-radius].copy()
    template -= template.mean()
    template_norm = np.linalg.norm(template)
    if template_norm == 0:
        return np.zeros(len(profiles), dtype=np.float32)

    # Use the same template support at every lag. Positive lag means that
    # the ground feature was observed at a larger raw sensor row.
    scores = np.empty((len(profiles), 2 * radius + 1), dtype=np.float32)
    for i in range(scores.shape[1]):
        window = detail[:, i : i + len(template)]
        centered = window - window.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(centered, axis=1) * template_norm
        dot = np.einsum("ij,j->i", centered, template)
        scores[:, i] = dot / np.maximum(norm, 1e-12)

    best = scores.argmax(axis=1)
    peak = best.clip(1, 2 * radius - 1)
    t = np.arange(len(profiles))
    left, middle, right = scores[t, peak - 1], scores[t, peak], scores[t, peak + 1]
    distant = np.abs(np.arange(scores.shape[1]) - peak[:, None]) > 2
    runner_up = np.where(distant, scores, -1).max(axis=1)
    valid = (peak == best) & (middle > 0.7) & (middle - runner_up > 0.06)
    fraction = 0.5 * (right - left) / np.maximum(2 * middle - left - right, 1e-6)
    # The green plane has a two-sensor-pixel row pitch.
    offsets = np.where(valid, 2 * (peak - radius + fraction), 0.0).astype(np.float32)
    confidence = valid.astype(np.float32)

    def smooth(sigma):
        weight = gaussian_filter1d(confidence, sigma)
        fitted = gaussian_filter1d(offsets, sigma) / np.maximum(weight, 1e-6)
        return fitted, weight

    fast, support = smooth(1.25)
    baseline, _ = smooth(_BASELINE_SIGMA / _PROFILE_COLUMNS)
    # Bridge brief dropouts. In prolonged or ambiguous intervals, taper the
    # correction to zero rather than projecting motion from distant columns.
    return (fast - baseline) * np.clip((support - 0.25) / 0.4, 0, 1)


def estimate_pitch_offsets(green, *, ground_at_high_rows: bool, ground_rows: int):
    """Return source-row offsets in sensor pixels, one per capture-time column.

    The input is the even-row green Bayer plane in raw acquisition order.
    Local templates overlap so a change in lighting or a raw file boundary
    does not introduce steps in the fitted trajectory.
    """
    if ground_rows < MIN_GROUND_ROWS:
        raise ValueError(f"the ground strip needs at least {MIN_GROUND_ROWS} sensor rows")
    height, width = green.shape
    count = min(height, (ground_rows + int(not ground_at_high_rows)) // 2)
    if count < MIN_GROUND_ROWS // 2 or width < 2 * _PROFILE_COLUMNS:
        return np.zeros(width, dtype=np.float32)
    rows = slice(height - count, height) if ground_at_high_rows else slice(0, count)
    starts = np.arange(0, width, _PROFILE_COLUMNS)
    sizes = np.minimum(width - starts, _PROFILE_COLUMNS)
    centers = starts + (sizes - 1) / 2
    offsets = np.zeros(len(starts), dtype=np.float32)
    step = _BLOCK_COLUMNS // (2 * _PROFILE_COLUMNS)
    ramp = np.arange(step, dtype=np.float32) / step
    for first in range(0, len(starts), step):
        stop = min(first + 2 * step, len(starts))
        columns = slice(starts[first], min(stop * _PROFILE_COLUMNS, width))
        strip = np.asarray(green[rows, columns], dtype=np.float32)
        sums = np.add.reduceat(strip, starts[first:stop] - starts[first], axis=1)
        profiles = (sums / sizes[None, first:stop].astype(np.float32)).T
        motion = _ground_motion(profiles)
        # Complementary linear ramps sum to one in the overlap.
        if first:
            motion[:step] *= ramp
        if stop < len(starts):
            motion[step:] *= 1 - ramp
        offsets[first:stop] += motion
        if stop == len(starts):
            break
    # A stationary textured profile can leave roundoff after detrending,
    # especially when the last averaged group has fewer input columns.
    offsets[np.abs(offsets) < 1e-5] = 0
    return np.interp(np.arange(width), centers, offsets).astype(np.float32)


def reconstruct_pitch_rows(plane, offsets, row_phase):
    """Linearly sample one Bayer plane directly on the stabilized sensor grid."""
    plane = np.asarray(plane, dtype=np.float32)
    height, width = plane.shape
    # Split even/odd output rows. Within either parity the fractional weight
    # is constant down a column. Clamp at the nearest measured color sample.
    result = np.empty((2 * height, width), dtype=np.float32)
    rows = np.arange(height, dtype=np.int32)[:, None]
    cols = np.arange(width)
    for parity in (0, 1):
        shift = (offsets + (parity - row_phase)) * 0.5
        lower = np.floor(shift).astype(np.int32)
        fraction = (shift - lower).astype(np.float32)
        source = rows + lower
        a = plane[source.clip(0, height - 1), cols]
        b = plane[(source + 1).clip(0, height - 1), cols]
        result[parity::2] = a + fraction * (b - a)
    return result
