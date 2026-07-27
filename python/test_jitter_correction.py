import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import preview


class ColumnJitterCorrectionTest(unittest.TestCase):
    HEIGHT = 128
    WIDTH = 512

    def make_clean_channels(self):
        rows = np.arange(self.HEIGHT, dtype=np.float32)[:, None]
        cols = np.arange(self.WIDTH, dtype=np.float32)[None, :]
        scene = (
            8000.0
            + 15.0 * cols
            + 1000.0 * np.sin(np.pi * rows / self.HEIGHT)
            + 600.0
            * np.exp(-np.square((cols - 260.0) / 55.0))
            * (rows > 50)
        )
        return np.stack(
            [scene * (0.8 + 0.1 * channel) + 300.0 * channel for channel in range(4)]
        ).astype(np.float32)

    def estimate(self, channels):
        return preview.column_exposure_jitter_correction(
            *channels,
            black_levels=np.zeros(4, dtype=np.float32),
            white_levels=np.full(4, 65535.0, dtype=np.float32),
            num_bands=16,
            temporal_window=31,
            # Deliberately not a divisor of WIDTH, to exercise block halos.
            block_columns=123,
            show_progress=False,
        )

    def test_monotonic_gradient_is_a_fixed_point(self):
        clean = self.make_clean_channels()
        model = self.estimate(clean)
        self.assertEqual(model.band_corrections.dtype, np.float16)
        self.assertEqual(model.width, self.WIDTH)
        self.assertEqual(
            float(np.max(np.abs(model.band_corrections.astype(np.float32)))),
            0.0,
        )

    def test_channel_and_row_dependent_stripes_are_reduced(self):
        clean = self.make_clean_channels()
        rng = np.random.default_rng(4)
        line_error = rng.normal(0.0, 120.0, self.WIDTH).astype(np.float32)
        line_error += 180.0 * np.sin(
            np.arange(self.WIDTH, dtype=np.float32) * 2.3
        )
        rows = np.arange(self.HEIGHT, dtype=np.float32)
        row_response = 0.7 + 0.3 * np.cos(
            (rows + 0.5) * 2.0 * np.pi / self.HEIGHT
        )
        channel_response = np.array([1.0, 0.82, 1.14, 0.93], dtype=np.float32)
        striped = clean + (
            channel_response[:, None, None]
            * row_response[None, :, None]
            * line_error[None, None, :]
        )

        model = self.estimate(striped)
        corrected = np.stack(
            [
                model.apply_window(striped[channel], channel, 0)
                for channel in range(4)
            ]
        )

        for channel in range(4):
            before = np.sqrt(np.mean(np.square(striped[channel] - clean[channel])))
            after = np.sqrt(np.mean(np.square(corrected[channel] - clean[channel])))
            self.assertLess(after, 0.35 * before)

        single_block_model = preview.column_exposure_jitter_correction(
            *striped,
            black_levels=np.zeros(4, dtype=np.float32),
            white_levels=np.full(4, 65535.0, dtype=np.float32),
            num_bands=16,
            temporal_window=31,
            block_columns=self.WIDTH,
            show_progress=False,
        )
        np.testing.assert_array_equal(
            model.band_corrections,
            single_block_model.band_corrections,
        )

    def test_broad_stripe_cadence_is_reduced(self):
        clean_template = self.make_clean_channels()[:, :, :1]
        clean = np.repeat(clean_template, self.WIDTH, axis=2)
        columns = np.arange(self.WIDTH, dtype=np.float32)
        broad_error = 300.0 * np.sin(2.0 * np.pi * columns / 120.0)
        channel_response = np.array([1.0, 0.75, 1.2, 0.9], dtype=np.float32)
        striped = clean + channel_response[:, None, None] * broad_error[None, None, :]

        model = preview.column_exposure_jitter_correction(
            *striped,
            black_levels=np.zeros(4, dtype=np.float32),
            white_levels=np.full(4, 65535.0, dtype=np.float32),
            num_bands=16,
            fine_temporal_window=31,
            temporal_window=255,
            block_columns=123,
            show_progress=False,
        )
        corrected = np.stack(
            [
                model.apply_window(striped[channel], channel, 0)
                for channel in range(4)
            ]
        )
        before = np.sqrt(np.mean(np.square(striped - clean)))
        after = np.sqrt(np.mean(np.square(corrected - clean)))
        self.assertLess(after, 0.5 * before)

    def test_clipped_scene_transition_does_not_leak_into_other_rows(self):
        clean = self.make_clean_channels()
        clipped_scene = clean.copy()
        # A narrow, overexposed roof-like feature occupies the lower half of
        # the sensor. Its temporal-median residual must not be projected into
        # the unaffected dark/gradient rows above it.
        clipped_scene[:, self.HEIGHT // 2 :, 220:228] = 65535.0

        model = preview.column_exposure_jitter_correction(
            *clipped_scene,
            black_levels=np.zeros(4, dtype=np.float32),
            white_levels=np.full(4, 65535.0, dtype=np.float32),
            num_bands=16,
            fine_temporal_window=31,
            temporal_window=255,
            block_columns=123,
            show_progress=False,
        )
        corrected = np.stack(
            [
                model.apply_window(clipped_scene[channel], channel, 0)
                for channel in range(4)
            ]
        )

        np.testing.assert_array_equal(
            corrected[:, : self.HEIGHT // 2],
            clipped_scene[:, : self.HEIGHT // 2],
        )

    def test_structure_mask_preserves_a_narrow_scene_feature(self):
        clean = self.make_clean_channels()
        scene = clean.copy()
        # Keep the feature inside both correction caps, so the structure mask
        # rather than the amplitude limit is what preserves it.
        scene[:, :, 220:228] += 500.0
        scale = 4
        structure_mask = np.zeros(
            (self.HEIGHT // scale, self.WIDTH // scale),
            dtype=bool,
        )
        structure_mask[:, 220 // scale : 228 // scale] = True

        model = preview.column_exposure_jitter_correction(
            *scene,
            black_levels=np.zeros(4, dtype=np.float32),
            white_levels=np.full(4, 65535.0, dtype=np.float32),
            num_bands=16,
            temporal_window=31,
            block_columns=123,
            structure_mask=structure_mask,
            structure_mask_scale=scale,
            show_progress=False,
        )
        corrected = np.stack(
            [model.apply_window(scene[channel], channel, 0) for channel in range(4)]
        )
        np.testing.assert_array_equal(
            corrected[:, :, 220:228],
            scene[:, :, 220:228],
        )

    def test_unsupported_owner_band_rejects_neighbor_interpolation(self):
        corrections = np.zeros((4, 2, 1), dtype=np.float16)
        corrections[preview.JITTER_RED, 1, 0] = 100.0
        model = preview.ColumnJitterModel(
            corrections,
            image_height=8,
            black_levels=np.zeros(4, dtype=np.float32),
            white_levels=np.full(4, 1000.0, dtype=np.float32),
            confidence=np.ones(1, dtype=np.float32),
        )
        values = np.full((2, 1), 500.0, dtype=np.float32)
        corrected = model.correct_samples(
            values,
            preview.JITTER_RED,
            np.zeros(1, dtype=np.int32),
            rows=np.array([3, 4], dtype=np.int32),
        )

        # Both rows interpolate across the same band boundary. Row 3 belongs
        # to unsupported band 0 and must stay exact; row 4 belongs to the
        # supported neighbor and may receive its tapered correction.
        self.assertEqual(float(corrected[0, 0]), 500.0)
        self.assertGreater(float(corrected[1, 0]), 500.0)

    def test_clipped_samples_are_unchanged(self):
        clean = self.make_clean_channels()
        rng = np.random.default_rng(9)
        line_error = rng.normal(0.0, 150.0, self.WIDTH).astype(np.float32)
        striped = clean + line_error[None, None, :]
        striped[preview.JITTER_GREEN2, :32, 200:240] = 0.0
        striped[preview.JITTER_RED, 96:, 320:350] = 65535.0

        model = self.estimate(striped)
        corrected_green2 = model.apply_window(
            striped[preview.JITTER_GREEN2],
            preview.JITTER_GREEN2,
            0,
        )
        corrected_red = model.apply_window(
            striped[preview.JITTER_RED],
            preview.JITTER_RED,
            0,
        )
        self.assertTrue(np.all(corrected_green2[:32, 200:240] == 0.0))
        self.assertTrue(np.all(corrected_red[96:, 320:350] == 65535.0))

        # The per-pixel column lookup used by skewed resampling must agree with
        # applying a contiguous correction window.
        columns = np.tile(
            np.arange(190, 250, dtype=np.int32),
            (self.HEIGHT, 1),
        )
        rows = np.arange(self.HEIGHT, dtype=np.int32)
        samples = striped[preview.JITTER_GREEN2, rows[:, None], columns]
        corrected_samples = model.correct_samples(
            samples,
            preview.JITTER_GREEN2,
            columns,
            rows,
        )
        np.testing.assert_allclose(
            corrected_samples,
            corrected_green2[rows[:, None], columns],
            rtol=0.0,
            atol=0.0,
        )

    def test_speed_fit_rejects_low_confidence_reversal_aliases(self):
        x = np.arange(40, dtype=np.float64)
        y = np.full(40, -0.3, dtype=np.float64)
        weights = np.ones(40, dtype=np.float64)
        y[:8] = 2.7
        y[-8:] = 2.7
        weights[:8] = 0.1
        weights[-8:] = 0.1

        with TemporaryDirectory() as output_dir:
            spline = preview.robust_bspline_fit(
                x,
                y,
                weights,
                smoothness=80000,
                g=Path(output_dir),
            )

        # The fallback clamps extrapolation to the supported interval, so the
        # rejected opposite-direction aliases cannot reappear at either end.
        fitted = spline(np.linspace(-100.0, 100.0, 1000))
        self.assertTrue(np.all(fitted < 0.0))
        np.testing.assert_allclose(fitted, -0.3, atol=1e-9, rtol=0.0)

    def test_speed_fit_rejects_catastrophic_dynamic_range(self):
        x = np.arange(80, dtype=np.float64)
        y = np.linspace(-0.15, -0.45, len(x))
        weights = np.ones_like(x)

        with TemporaryDirectory() as output_dir:
            spline = preview.robust_bspline_fit(
                x,
                y,
                weights,
                smoothness=80000,
                g=Path(output_dir),
            )

        fitted = spline(np.linspace(-100.0, 200.0, 1000))
        np.testing.assert_allclose(fitted, -0.3, atol=0.002, rtol=0.0)

    def test_speed_fit_preserves_supported_moderate_change(self):
        x = np.arange(80, dtype=np.float64)
        y = np.linspace(-1.0, -1.2, len(x))
        weights = np.ones_like(x)

        with TemporaryDirectory() as output_dir:
            spline = preview.robust_bspline_fit(
                x,
                y,
                weights,
                smoothness=80000,
                g=Path(output_dir),
            )

        fitted = spline(x)
        self.assertLess(fitted[-1], fitted[0] - 0.15)


if __name__ == "__main__":
    unittest.main()
