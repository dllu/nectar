"""Geometric, sensor-plane alignment, and export checks for the swing one-off."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tifffile

from preview import write_linear_dng
from swing import Capture, Passage, fit_motion, frame_passage, render_wedge


class SwingTest(unittest.TestCase):
    def test_window_spans_variable_size_raw_chunks(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw = np.arange(4096 * 16, dtype=np.uint16).reshape(4096, 16)
            raw[:, :6].T.tofile(folder / "im2.bin")
            raw[:, 6:].T.tofile(folder / "im10.bin")
            capture = Capture(folder)
            self.assertEqual(capture.width, 8)
            np.testing.assert_array_equal(capture.read(2, 7), raw[:, 4:14])
            with self.assertRaises(ValueError):
                capture.read(0, 9)

    def test_signed_sinusoidal_motion_round_trip_and_turning_point_guard(self):
        for sign in (-1, 1):
            passage = Passage(1000, sign * 0.0007, 6000, 400)
            times = np.linspace(600, 1400, 100)
            np.testing.assert_allclose(passage.time(passage.angle(times)), times, atol=1e-10)
            with self.assertRaisesRegex(ValueError, "monotonic"):
                Passage(1000, sign * 0.0007, 6000, 1500)

    def test_joint_fit_recovers_fixed_pivot_with_variable_amplitudes_and_outliers(self):
        rng = np.random.default_rng(42)
        centers = np.array([1000, 4000, 7000, 10000], dtype=float)
        rates = np.array([0.0005, -0.0007, 0.0006, -0.0008])
        pivot, period = -1500, 6000
        measurements = []
        for i, rate in enumerate(rates):
            for y in (256, 768, 1280, 1792, 2304):
                for dt in (-128, -64, 0, 64, 128):
                    lag = 1 / (rate * (y - pivot) * np.cos(2 * np.pi * dt / period))
                    lag *= np.exp(rng.normal(0, 0.003))
                    measurements.append([i, y, centers[i] + dt, lag, 1])
        measurements = np.asarray(measurements)
        measurements[::17, 3] *= 1.8
        estimated_pivot, estimated_rates, _ = fit_motion(
            measurements, centers, period, "sinusoidal"
        )
        self.assertAlmostEqual(estimated_pivot, pivot, delta=35)
        np.testing.assert_allclose(estimated_rates, rates, rtol=0.015)
        measurements[measurements[:, 0] == 1, 3] *= -1
        with self.assertRaisesRegex(ValueError, "alternate"):
            fit_motion(measurements, centers, period, "sinusoidal")

    @staticmethod
    def synthetic_raw(passage, pivot=-80, rows=64, columns=200, angular_gradient=7000):
        # Each sensor plane observes the same analytic grayscale scene. R/G1
        # see a position one spatial pixel beyond G2/B along the scan direction.
        # This must disappear after alignment, for either direction of travel.
        y = np.arange(rows)[:, None]
        t = np.arange(columns)[None, :]
        radius = y - pivot
        angle = passage.angle(t)
        base = 10000 + 100 * radius
        early = base + angular_gradient * angle
        late = base + angular_gradient * (angle + 1 / radius)
        raw = np.empty((rows, columns * 2), dtype=np.uint16)
        raw[:, ::2] = np.rint(early).astype(np.uint16)
        raw[:, 1::2] = np.rint(late).astype(np.uint16)
        return raw

    def test_polar_arcs_plane_alignment_and_opposite_pass_orientation(self):
        rendered = []
        for motion in ("constant", "sinusoidal"):
            for sign in (-1, 1):
                passage = Passage(100, sign * 0.002, 1000, 60, motion)
                raw = self.synthetic_raw(passage)
                rgb, alpha, geometry = render_wedge(raw, 0, passage, -80)
                px, py = geometry["pivot_in_output_pixels"]
                yy, xx = np.indices(alpha.shape)
                radius = np.hypot(xx - px, yy - py)
                angle = -np.arctan2(xx - px, yy - py)
                expected = 10000 + 100 * radius + 7000 * angle
                interior = (alpha > 0) & (radius > 82) & (radius < 141)
                self.assertGreater(interior.sum(), 500)
                # Nonzero angular texture checks scan-plane compensation as
                # well as the radius dependence and temporal inversion.
                self.assertLess(np.max(np.abs(rgb[interior] - expected[interior, None])), 5)
                self.assertLess(np.max(np.ptp(rgb[interior].astype(int), axis=1)), 5)
                self.assertTrue(np.all(rgb[alpha == 0] == 0))
                self.assertLess(py, 0)
                rendered.append(rgb)
        np.testing.assert_allclose(rendered[0], rendered[1], atol=2)
        np.testing.assert_allclose(rendered[2], rendered[3], atol=2)

    def test_stationary_background_rows_become_circular_arcs(self):
        passage = Passage(100, 0.002, 1000, 60, "constant")
        raw = self.synthetic_raw(passage, angular_gradient=0)
        rgb, alpha, geometry = render_wedge(raw, 0, passage, -80)
        yy, xx = np.indices(alpha.shape)
        px, py = geometry["pivot_in_output_pixels"]
        radius = np.hypot(xx - px, yy - py)
        valid = (alpha > 0) & (radius > 82) & (radius < 141)
        np.testing.assert_allclose(rgb[:, :, 0][valid], (10000 + 100 * radius)[valid], atol=4)
        # Brightness varies across an output row, unlike a column-only warp.
        self.assertGreater(np.ptp(rgb[40, :, 0][alpha[40] > 0]), 50)

    def test_shared_angle_and_centering_preserve_motion_phase_in_both_directions(self):
        masks = []
        geometries = []
        for motion in ("constant", "sinusoidal"):
            for omega in (0.002, -0.002, 0.003, -0.003):
                passage, offset, interval = frame_passage(100, omega, 1000, motion, 120, 0.18)
                self.assertEqual(passage.center, 100)  # framing must not move peak speed
                self.assertAlmostEqual(float(passage.time(offset)), 120)
                limits = np.sort(passage.angle(np.asarray(interval)) - offset)
                np.testing.assert_allclose(limits, [-0.09, 0.09], atol=1e-12)
                raw = self.synthetic_raw(passage)
                rgb, alpha, geometry = render_wedge(
                    raw, 0, passage, -80, half_angle=0.09, center_angle=offset,
                )
                masks.append(alpha)
                geometries.append(geometry)
                px, py = geometry["pivot_in_output_pixels"]
                yy, xx = np.indices(alpha.shape)
                radius = np.hypot(xx - px, yy - py)
                # The same output ray samples a different source angle after
                # framing, while the circular background geometry stays fixed.
                source_angle = -np.arctan2(xx - px, yy - py) + offset
                expected = 10000 + 100 * radius + 7000 * source_angle
                valid = (alpha > 0) & (radius > 82) & (radius < 141)
                self.assertLess(np.max(np.abs(rgb[valid] - expected[valid, None])), 5)
        for mask, geometry in zip(masks[1:], geometries[1:]):
            np.testing.assert_array_equal(mask, masks[0])
            self.assertEqual(geometry["angle_width_radians"], 0.18)
            self.assertEqual(geometry["pivot_in_output_pixels"], geometries[0]["pivot_in_output_pixels"])

    def test_shifted_angular_crop_cannot_wrap_through_a_reversal(self):
        with self.assertRaisesRegex(ValueError, "reversal"):
            frame_passage(100, 0.002, 1000, "sinusoidal", 300, 0.2)
        with self.assertRaisesRegex(ValueError, "framing center"):
            frame_passage(100, 0.002, 1000, "sinusoidal", 400, 0.2)

    def test_dng_retains_upright_linear_pixels_and_capture_metadata(self):
        passage = Passage(100, -0.002, 1000, 60)
        raw_rgb, _, _ = render_wedge(self.synthetic_raw(passage), 0, passage, -80)
        self.assertGreater(np.unique(raw_rgb).size, 256)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "swing.dng"
            # Same boundary convention as swing.process: cancel the legacy
            # writer's vertical flip, without changing that shared writer.
            write_linear_dng(path, raw_rgb[::-1], np.array([1.2, 1, 1.4]),
                             captured_at=datetime(2026, 9, 3, 0, 31, 12),
                             exposure_time=(4170, 10_000_000))
            with tifffile.TiffFile(path) as dng:
                page = dng.pages[0]
                self.assertEqual(page.dtype, np.dtype(np.uint16))
                self.assertEqual(page.photometric, tifffile.PHOTOMETRIC.LINEAR_RAW)
                self.assertEqual(page.tags[274].value, 1)
                self.assertEqual(page.tags[36867].value, "2026:09:03 00:31:12")
                self.assertEqual(page.tags[33434].value, (4170, 10_000_000))
                np.testing.assert_array_equal(page.asarray(), raw_rgb)


if __name__ == "__main__":
    unittest.main()
