import unittest

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
            basis_size=8,
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
            basis_size=8,
            temporal_window=31,
            block_columns=self.WIDTH,
            show_progress=False,
        )
        np.testing.assert_array_equal(
            model.band_corrections,
            single_block_model.band_corrections,
        )

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


if __name__ == "__main__":
    unittest.main()
