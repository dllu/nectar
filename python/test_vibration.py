import unittest

import numpy as np

import preview
from vibration import estimate_pitch_offsets, reconstruct_pitch_rows


class CameraVibrationTest(unittest.TestCase):
    def test_recovers_subpixel_motion_from_display_bottom_and_rejects_slow_drift(self):
        # The two sensor ends move in opposite directions. Cross file-sized and
        # template boundaries and finish with a partial four-column profile.
        time = np.arange(8201)
        fast = (
            1.9 * np.sin(2 * np.pi * time / 161)
            + 0.38 * np.cos(2 * np.pi * time / 77)
        )
        slow = 3 * np.sin(2 * np.pi * time / 10000)
        rows = 2 * np.arange(160)[:, None]
        y = rows - np.where(rows < 160, -1, 1) * (fast + slow)
        green = (
            700 + 110 * np.sin(y * 0.22) + 80 * np.cos(y * 0.61)
            + 66 * np.sin(y * 0.38)
        )
        green += np.random.default_rng(16).normal(0, 2, green.shape)

        for high_rows, sign in ((False, -1), (True, 1)):
            with self.subTest(high_rows=high_rows):
                offsets = estimate_pitch_offsets(
                    green, ground_at_high_rows=high_rows, ground_rows=128
                )
                errors = (offsets - sign * fast)[300:-300]
                self.assertLess(np.sqrt(np.mean(errors**2)), 0.08)
                self.assertLess(np.max(np.abs(np.diff(offsets))), 0.13)

    def test_flat_ambiguous_and_stationary_ground_add_no_motion(self):
        y = np.arange(64)[:, None]
        t = np.arange(4099)
        stationary = np.random.default_rng(28).uniform(0, 65535, size=(64, 1))
        for strip in (
            np.full((64, len(t)), 65535),
            np.cos(2 * np.pi * (y - np.sin(t / 20)) / 5),
            np.broadcast_to(stationary, (64, len(t))),
        ):
            offsets = estimate_pitch_offsets(
                strip, ground_at_high_rows=True, ground_rows=128
            )
            np.testing.assert_array_equal(offsets, 0)

    def test_fused_sampling_uses_each_aperture_tap_and_bayer_phase(self):
        height, width = 25, 160
        t = np.arange(width)
        offsets = (2.3 * np.cos(t * 0.87) - 0.43).astype(np.float32)
        y = 2 * np.arange(height)[:, None]
        phases = (0, 1, 0, 1)
        # Sensor samples of four continuous ramps undergoing known pitch.
        # A correct inverse warp reproduces each ramp on every interior row.
        planes = [
            (100 * c + y + phase - offsets).astype(np.float32)
            for c, phase in enumerate(phases)
        ]

        class View:
            def __init__(self, array):
                self.array = array
                self.height, self.width = array.shape

            def __getitem__(self, key):
                return self.array[key]

        positions = np.linspace(6.3, width - 8.1, 83)
        expected = np.arange(5, 2 * height - 5)[:, None, None] + [0, 150, 300]
        for direction in (1, -1):
            chunks = preview.sample_motion_corrected_chunks(
                *map(View, planes),
                positions[::direction],
                np.full(len(positions), 3.2),
                pitch_offsets=offsets,
                skew=0.14,
                max_pix=13,
                step_scales=np.full(len(positions), direction),
                show_progress=False,
            )
            interior = np.concatenate([chunk for _, chunk in chunks], axis=1)[5:-5]
            np.testing.assert_allclose(
                interior, np.broadcast_to(expected, interior.shape), atol=1e-4
            )

    def test_row_reconstruction_replicates_nearest_color_sample_at_boundaries(self):
        plane = np.array([[10, 30], [20, 10]], dtype=np.uint16)
        actual = reconstruct_pitch_rows(plane, np.array([-0.5, 1.5]), row_phase=1)
        np.testing.assert_array_equal(actual, [[10, 25], [10, 15], [12.5, 10], [17.5, 10]])


if __name__ == "__main__":
    unittest.main()
