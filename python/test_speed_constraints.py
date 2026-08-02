import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from image_processing import auto_tone_map_to_uint8
from speed_constraints import (
    AnnotationDocument,
    CaptureSequence,
    CircleConstraint,
    CutPosition,
    DEFAULT_WINDOW_SIZE,
    FrameInfo,
    WidthGroup,
    WidthSegment,
    build_navigator_data,
    capture_defaults_to_rotated_180,
    decode_raw_frame,
    decode_raw_frame_thumbnail,
    navigator_thumbnail_geometry,
)


class SpeedConstraintsTest(unittest.TestCase):
    FRAME_ROWS = 4

    @staticmethod
    def write_frame(path: Path, raw: np.ndarray) -> None:
        np.asarray(raw.T, dtype="<u2").tofile(path)

    def make_raw_frame(self, raw_columns: int = 4) -> np.ndarray:
        return np.zeros((self.FRAME_ROWS, raw_columns), dtype=np.uint16)

    def test_default_window_shows_sixteen_chunks(self):
        self.assertEqual(DEFAULT_WINDOW_SIZE, 16)

    def test_discovers_frames_in_natural_order_and_builds_global_x(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_frame(root / "im10.bin", self.make_raw_frame())
            self.write_frame(root / "im2.bin", self.make_raw_frame())

            capture = CaptureSequence.discover(root, frame_rows=self.FRAME_ROWS)

            self.assertEqual(
                [frame.path.name for frame in capture.frames],
                ["im2.bin", "im10.bin"],
            )
            self.assertEqual([frame.x_start for frame in capture.frames], [0, 2])
            self.assertEqual(capture.image_height, 2)
            self.assertEqual(capture.total_width, 4)

    def test_decodes_the_preview_bayer_layout_without_color_correction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.make_raw_frame()
            red = np.array([[10, 11], [12, 13]], dtype=np.uint16)
            green_1 = np.array([[30, 32], [34, 36]], dtype=np.uint16)
            green_2 = np.array([[50, 52], [54, 56]], dtype=np.uint16)
            blue = np.array([[70, 71], [72, 73]], dtype=np.uint16)
            raw[0::2, 1::2] = red << 8
            raw[1::2, 1::2] = green_1 << 8
            raw[0::2, 0::2] = green_2 << 8
            raw[1::2, 0::2] = blue << 8
            self.write_frame(root / "im000000.bin", raw)
            capture = CaptureSequence.discover(root, frame_rows=self.FRAME_ROWS)

            rgb = decode_raw_frame(capture.frames[0], self.FRAME_ROWS)

            self.assertEqual(rgb.shape, (2, 2, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            np.testing.assert_array_equal(rgb[:, :, 0], red)
            np.testing.assert_array_equal(rgb[:, :, 1], (green_1 + green_2) // 2)
            np.testing.assert_array_equal(rgb[:, :, 2], blue)
            thumbnail = decode_raw_frame_thumbnail(
                capture.frames[0],
                self.FRAME_ROWS,
                target_height=2,
                target_width=2,
            )
            np.testing.assert_array_equal(thumbnail, rgb)

    def test_auto_tone_map_stretches_contrast_and_applies_gamma(self):
        rgb = np.array([[[32, 96, 160]]], dtype=np.uint8)

        corrected = auto_tone_map_to_uint8(
            rgb,
            low_percentile=0,
            high_percentile=100,
            gamma=2.0,
        )

        np.testing.assert_array_equal(
            corrected,
            np.array([[[0, 180, 255]]], dtype=np.uint8),
        )

    def test_json_round_trip_preserves_constraints_and_assigns_cut_roles(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_frame(root / "im000000.bin", self.make_raw_frame())
            capture = CaptureSequence.discover(root, frame_rows=self.FRAME_ROWS)
            document = AnnotationDocument(
                circles=[CircleConstraint("circle_001", 0.25, 0.5, 1.5, 1.25)],
                width_groups=[
                    WidthGroup(
                        "width_group_001",
                        "Train windows",
                        [WidthSegment("segment_001", 0.1, 1.1)],
                    )
                ],
                cut_positions=[
                    CutPosition("position_002", 1.5),
                    CutPosition("position_001", 0.0),
                    CutPosition("position_003", 1.0),
                ],
            )
            path = root / "speed_constraints.json"

            document.save(path, capture)
            loaded = AnnotationDocument.load(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(len(loaded.circles), 1)
            self.assertEqual(loaded.width_groups[0].name, "Train windows")
            self.assertEqual(
                [cut.x for cut in loaded.cut_positions],
                [0.0, 1.0, 1.5],
            )
            self.assertEqual(
                [entry["role"] for entry in payload["cut_positions"]],
                ["start", "cut", "end"],
            )
            self.assertEqual(
                payload["coordinate_system"]["name"], "bayer_channel_pixels"
            )
            self.assertIsNone(loaded.source_warning(capture))
            self.assertEqual(list(root.glob(".speed_constraints.json.*.tmp")), [])

    def test_builds_a_single_bounded_thumbnail_strip_for_all_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_frame(root / "im000000.bin", self.make_raw_frame())
            self.write_frame(root / "im000001.bin", self.make_raw_frame())
            capture = CaptureSequence.discover(root, frame_rows=self.FRAME_ROWS)
            progress = []

            navigator = build_navigator_data(
                capture,
                preferred_height=2,
                max_bytes=1024,
                progress=lambda completed, total: progress.append((completed, total)),
            )

            self.assertIsNotNone(navigator)
            assert navigator is not None
            self.assertEqual(navigator.rgb.shape, (2, 4, 3))
            self.assertEqual(navigator.frame_starts, (0, 2))
            self.assertEqual(navigator.frame_widths, (2, 2))
            self.assertEqual(progress[-1], (2, 2))

    def test_navigator_geometry_reduces_resolution_to_honor_memory_budget(self):
        frame_path = Path("unused.bin")
        capture = CaptureSequence(
            directory=Path("."),
            frame_rows=self.FRAME_ROWS,
            frames=[
                FrameInfo(frame_path, raw_columns=4, x_start=index * 2)
                for index in range(50)
            ],
        )

        height, widths = navigator_thumbnail_geometry(
            capture,
            preferred_height=2,
            max_bytes=300,
        )

        self.assertLessEqual(height * sum(widths) * 3, 300)
        self.assertEqual(height, 1)
        self.assertEqual(set(widths), {1})

    def test_rejects_a_raw_file_with_an_incomplete_sensor_column(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "im000000.bin").write_bytes(b"\0" * 7)
            with self.assertRaisesRegex(ValueError, "expected a multiple"):
                CaptureSequence.discover(root, frame_rows=self.FRAME_ROWS)

    def test_gui_rotation_defaults_to_the_camera_mount_era(self):
        self.assertTrue(
            capture_defaults_to_rotated_180(Path("2025-12-31-23-59-59"))
        )
        self.assertFalse(
            capture_defaults_to_rotated_180(Path("2026-01-01-00-00-00"))
        )
        self.assertFalse(
            capture_defaults_to_rotated_180(Path("2032-03-04-05-06-07-extra"))
        )
        self.assertFalse(capture_defaults_to_rotated_180(Path("capture")))


if __name__ == "__main__":
    unittest.main()
