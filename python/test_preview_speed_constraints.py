import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import preview


class PreviewSpeedConstraintsTest(unittest.TestCase):
    def test_circle_targets_full_resolution_output_aspect_ratio(self):
        circle = preview.CircleSpeedConstraint(
            "wheel",
            x=10.0,
            y=20.0,
            width=180.0,
            height=150.0,
        )

        self.assertAlmostEqual(circle.center_x, 100.0)
        self.assertAlmostEqual(circle.speed_magnitude, 0.6)

    def test_refinement_preserves_direction_and_reduces_manual_residuals(self):
        def correlation_spline(values):
            values = np.asarray(values, dtype=np.float64)
            return np.full_like(values, -0.72)

        circles = (
            preview.CircleSpeedConstraint(
                "left", 130.0, 0.0, 132.0, 100.0
            ),
            preview.CircleSpeedConstraint(
                "right", 720.0, 0.0, 124.0, 100.0
            ),
        )
        group = preview.EqualWidthGroup(
            "windows",
            "Windows",
            (
                preview.EqualWidthSegment("wide-fast", 100.0, 220.0),
                preview.EqualWidthSegment("narrow-slow", 700.0, 800.0),
            ),
        )
        constraints = preview.ManualSpeedConstraints(
            Path("constraints.json"), circles, (group,), ()
        )

        before_circle = np.mean(
            [
                abs(abs(correlation_spline(circle.center_x)) / circle.speed_magnitude - 1)
                for circle in circles
            ]
        )
        before_widths = np.asarray(
            [
                preview.corrected_interval_length(
                    correlation_spline, segment.x1, segment.x2
                )
                for segment in group.segments
            ]
        )
        refined = preview.refine_speed_spline_with_constraints(
            correlation_spline,
            constraints,
            x_min=0.0,
            x_max=1000.0,
            verbose=False,
        )
        after_circle = np.mean(
            [
                abs(abs(refined(circle.center_x)) / circle.speed_magnitude - 1)
                for circle in circles
            ]
        )
        after_widths = np.asarray(
            [
                preview.corrected_interval_length(refined, segment.x1, segment.x2)
                for segment in group.segments
            ]
        )

        self.assertTrue(np.all(refined(np.linspace(0.0, 1000.0, 101)) < 0.0))
        self.assertLess(after_circle, before_circle)
        self.assertLess(
            np.ptp(np.log(after_widths)),
            np.ptp(np.log(before_widths)),
        )

    def test_refinement_is_not_capped_at_twenty_five_percent(self):
        def correlation_spline(values):
            values = np.asarray(values, dtype=np.float64)
            return np.full_like(values, 0.2)

        circles = tuple(
            preview.CircleSpeedConstraint(
                f"circle-{index}",
                x - 50.0,
                0.0,
                100.0,
                100.0,
            )
            for index, x in enumerate((100.0, 350.0, 650.0, 900.0))
        )
        constraints = preview.ManualSpeedConstraints(
            Path("constraints.json"), circles, (), ()
        )

        refined = preview.refine_speed_spline_with_constraints(
            correlation_spline,
            constraints,
            x_min=0.0,
            x_max=1000.0,
            verbose=False,
        )

        fitted = np.abs(refined(np.asarray([circle.center_x for circle in circles])))
        self.assertTrue(np.all(fitted > 0.25))
        self.assertTrue(np.all(fitted < 0.55))

    def test_refinement_accepts_positive_speed_below_old_floor(self):
        def correlation_spline(values):
            values = np.asarray(values, dtype=np.float64)
            return np.full_like(values, -0.01)

        circle = preview.CircleSpeedConstraint(
            "slow", 450.0, 0.0, 4.0, 100.0
        )
        constraints = preview.ManualSpeedConstraints(
            Path("constraints.json"), (circle,), (), ()
        )

        refined = preview.refine_speed_spline_with_constraints(
            correlation_spline,
            constraints,
            x_min=0.0,
            x_max=1000.0,
            verbose=False,
        )

        self.assertLess(float(refined(circle.center_x)), -0.01)

    def test_manual_width_constraints_are_fit_to_sub_tenth_percent(self):
        starts = np.asarray(
            [4000.0, 5500.0, 7000.0, 8500.0, 10000.0, 11500.0]
        )
        raw_widths = np.asarray([760.0, 780.0, 775.0, 790.0, 770.0, 785.0])
        segments = tuple(
            preview.EqualWidthSegment(f"segment-{index}", start, start + width)
            for index, (start, width) in enumerate(zip(starts, raw_widths))
        )
        # These two absolute speed targets agree with a common corrected width
        # of 2500 pixels for the first and last intervals.
        circles = tuple(
            preview.CircleSpeedConstraint(
                f"circle-{index}",
                starts[index] + raw_widths[index] / 2.0 - raw_widths[index] / 25.0,
                0.0,
                2.0 * raw_widths[index] / 25.0,
                100.0,
            )
            for index in (0, len(starts) - 1)
        )
        constraints = preview.ManualSpeedConstraints(
            Path("constraints.json"),
            circles,
            (preview.EqualWidthGroup("cars", "Cars", segments),),
            (),
        )

        def inaccurate_correlation(values):
            values = np.asarray(values, dtype=np.float64)
            magnitude = 0.42 - 0.000014 * values + 0.025 * np.sin(values / 1700.0)
            return -magnitude

        refined = preview.refine_speed_spline_with_constraints(
            inaccurate_correlation,
            constraints,
            x_min=0.0,
            x_max=17000.0,
            verbose=False,
        )
        circle_diagnostics, group_diagnostics = (
            preview._speed_constraint_diagnostics(
                inaccurate_correlation,
                refined,
                constraints,
            )
        )
        circle_residuals = [
            abs(float(entry["refined_relative_residual"]))
            for entry in circle_diagnostics
        ]
        width_residuals = [
            abs(float(entry["refined_relative_residual"]))
            for entry in group_diagnostics[0]["segments"]
        ]

        self.assertLess(max(circle_residuals), 0.001)
        self.assertLess(max(width_residuals), 0.001)

    def test_speed_extrapolation_has_no_endpoint_clamp_kink(self):
        core = preview.CubicSpline(
            [0.0, 1.0, 2.0, 3.0],
            [-0.5, -0.4, -0.3, -0.25],
            bc_type="natural",
        )
        extended = preview._smooth_speed_extrapolator(core, 0.0, 3.0)

        epsilon = 1e-4
        for boundary in (0.0, 3.0):
            slope_from_left = (
                extended(boundary) - extended(boundary - epsilon)
            ) / epsilon
            slope_from_right = (
                extended(boundary + epsilon) - extended(boundary)
            ) / epsilon
            self.assertAlmostEqual(slope_from_left, slope_from_right, places=4)

        self.assertNotEqual(extended(-1.0), extended(0.0))
        self.assertNotEqual(extended(4.0), extended(3.0))
        self.assertTrue(np.all(extended(np.linspace(-2.0, 5.0, 100)) < 0.0))

    def test_manual_cut_bounds_exclude_background_correlation(self):
        correlation_x = np.linspace(0.0, 100.0, 21)

        def quiet_background(values):
            values = np.asarray(values, dtype=np.float64)
            return np.full_like(values, -0.3)

        def extreme_background(values):
            values = np.asarray(values, dtype=np.float64)
            inside = (values >= 25.0) & (values <= 75.0)
            return np.where(inside, -0.3, -3.0)

        for spline in (quiet_background, extreme_background):
            spline.correlation_x = correlation_x
            spline.correlation_weights = np.ones_like(correlation_x)
            spline.motion_direction = -1

        circle = preview.CircleSpeedConstraint(
            "circle", 35.0, 0.0, 30.0, 50.0
        )
        constraints = preview.ManualSpeedConstraints(
            Path("constraints.json"),
            (circle,),
            (),
            (25.0, 75.0),
        )
        quiet_fit = preview.refine_speed_spline_with_constraints(
            quiet_background,
            constraints,
            x_min=0.0,
            x_max=100.0,
            verbose=False,
        )
        extreme_fit = preview.refine_speed_spline_with_constraints(
            extreme_background,
            constraints,
            x_min=0.0,
            x_max=100.0,
            verbose=False,
        )

        inside = np.linspace(25.0, 75.0, 101)
        np.testing.assert_allclose(quiet_fit(inside), extreme_fit(inside))

    def test_cut_ranges_follow_both_correlation_directions(self):
        cuts = [20.0, 60.0, 90.0]

        forward = preview.build_cut_aligned_output_ranges(
            np.arange(0.0, 101.0), cuts
        )
        reverse = preview.build_cut_aligned_output_ranges(
            np.arange(100.0, -1.0, -1.0), cuts
        )

        self.assertEqual(forward, [(20, 60), (60, 90)])
        self.assertEqual(reverse, [(10, 40), (40, 80)])

    def test_equal_output_ranges_cover_roi_without_reaching_target(self):
        ranges = preview.build_equal_output_chunk_ranges(
            (7, 108), chunk_target=32
        )
        widths = [stop - start for start, stop in ranges]

        self.assertEqual(ranges[0][0], 7)
        self.assertEqual(ranges[-1][1], 108)
        self.assertEqual(sum(widths), 101)
        self.assertLessEqual(max(widths) - min(widths), 1)
        self.assertLess(max(widths), 32)
        self.assertTrue(
            all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        )

    def test_equal_output_ranges_split_a_target_sized_roi(self):
        ranges = preview.build_equal_output_chunk_ranges(
            (0, 32), chunk_target=32
        )

        self.assertEqual(ranges, [(0, 16), (16, 32)])

    def test_equal_length_chunk_option_is_off_by_default(self):
        self.assertFalse(preview._parse_args([]).equal_length_chunks)
        self.assertTrue(
            preview._parse_args(["--equal-length-chunks"]).equal_length_chunks
        )

    def test_sampling_uses_full_domain_refined_spline(self):
        def spline(values):
            return 0.2 + 0.01 * np.asarray(values, dtype=np.float64)

        positions, _widths, steps = preview.generate_sample_positions(
            spline,
            np.array([10.0, 20.0]),
            width=2,
        )

        self.assertEqual(positions[0], 0)
        self.assertAlmostEqual(steps[0], 0.2)

    def test_sampling_clamps_steps_that_cannot_make_safe_progress(self):
        def spline(values):
            values = np.asarray(values, dtype=np.float64)
            return np.full_like(values, 0.02)

        positions, _widths, steps = preview.generate_sample_positions(
            spline,
            np.array([0.0, 1.0]),
            width=2,
        )

        self.assertAlmostEqual(positions[1] - positions[0], 0.1)
        self.assertAlmostEqual(steps[0], 0.1)

    def test_cut_roi_adds_processing_padding(self):
        starts = np.array([0, 10, 20, 30, 40], dtype=np.int64)
        widths = np.full(4, 10, dtype=np.int64)

        selected = preview.select_frames_for_cut_region(
            starts,
            widths,
            (12.0, 28.0),
            padding=1,
        )

        self.assertEqual(selected, [0, 1, 2, 3])

    def test_skew_truncation_reports_the_matching_step_scale_slice(self):
        positions = list(np.arange(10.0, 91.0, 10.0))
        widths = [2.0] * len(positions)
        scales = np.arange(len(positions), dtype=np.float32)

        kept_positions, _kept_widths, kept_slice = (
            preview.truncate_sample_positions_for_skew(
                positions,
                widths,
                skew=0.5,
                step_scales=scales,
                row_offsets=np.array([-2.0, 2.0], dtype=np.float32),
                width=100,
                return_slice=True,
            )
        )

        np.testing.assert_array_equal(kept_positions, positions[:-1])
        np.testing.assert_array_equal(scales[kept_slice], scales[:-1])
        self.assertEqual(len(scales[kept_slice]), len(kept_positions))

    def test_stack_preview_centers_variable_width_cut_regions(self):
        narrow = np.full((2, 3, 3), 17, dtype=np.uint8)
        wide = np.full((2, 5, 3), 93, dtype=np.uint8)
        expected = np.zeros((4, 5, 3), dtype=np.uint8)
        expected[0:2, 1:4] = narrow
        expected[2:4] = wide

        with TemporaryDirectory() as directory:
            with preview.DiskBackedStackPreview(
                Path(directory), 2, chunk_widths=[3, 5]
            ) as stack:
                stack.add_chunk(1, wide)
                stack.add_chunk(0, narrow)
                with patch.object(preview, "write_jpeg") as write_jpeg:
                    stack.write(Path(directory) / "stack.jpg", np.empty(0))
                    result = np.array(write_jpeg.call_args.args[1], copy=True)

        np.testing.assert_array_equal(result, expected)

    def test_manual_cuts_skip_the_moving_region_pass(self):
        class ReachedStructureMask(Exception):
            pass

        with TemporaryDirectory() as directory:
            capture = Path(directory) / "2026-07-30-00-00-00"
            output = Path(directory) / "output"
            capture.mkdir()
            raw = np.ones((4096, 16), dtype=np.uint16)
            for index in range(2):
                raw.tofile(capture / f"im{index:06d}.bin")
            payload = {
                "schema_version": 1,
                "coordinate_system": {
                    "name": "bayer_channel_pixels",
                    "bayer_bin": 2,
                },
                "circles": [],
                "equal_width_groups": [],
                "cut_positions": [
                    {"id": "start", "x": 1.0},
                    {"id": "end", "x": 15.0},
                ],
            }
            (capture / preview.SPEED_CONSTRAINTS_JSON).write_text(
                json.dumps(payload), encoding="utf-8"
            )

            with (
                patch.object(
                    preview.np,
                    "percentile",
                    side_effect=AssertionError("moving-region pass should be skipped"),
                ),
                patch.object(
                    preview,
                    "estimate_white_balance",
                    return_value=np.ones(3, dtype=np.float32),
                ),
                patch.object(preview, "autoexposure", return_value=(0.0, 1.0)),
                patch.object(
                    preview,
                    "build_structure_mask",
                    side_effect=ReachedStructureMask,
                ),
            ):
                with self.assertRaises(ReachedStructureMask):
                    preview.process_preview(
                        capture,
                        padding=0,
                        show_progress=False,
                        output_dir=output,
                    )


if __name__ == "__main__":
    unittest.main()
