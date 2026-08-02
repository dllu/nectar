import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import preview


class MemoryEfficientHelpersTest(unittest.TestCase):
    def test_process_preview_passes_show_progress_to_structure_mask(self):
        class ReachedStructureMask(Exception):
            pass

        with TemporaryDirectory() as directory:
            capture = Path(directory) / "2026-07-30-00-00-00"
            output = Path(directory) / "output"
            capture.mkdir()
            raw = np.ones((4096, 16), dtype=np.uint16)
            for index in range(2):
                raw.tofile(capture / f"{index:04d}.bin")

            with (
                patch.object(preview.np, "percentile", side_effect=(1.0, 2.0)),
                patch.object(
                    preview,
                    "estimate_white_balance",
                    return_value=np.ones(3, dtype=np.float32),
                ),
                patch.object(preview, "autoexposure", return_value=(0.0, 1.0)),
                patch.object(
                    preview,
                    "build_structure_mask",
                    autospec=True,
                    side_effect=ReachedStructureMask,
                ) as build_structure_mask,
            ):
                with self.assertRaises(ReachedStructureMask):
                    preview.process_preview(
                        capture,
                        padding=0,
                        show_progress=False,
                        output_dir=output,
                    )

            self.assertFalse(build_structure_mask.call_args.kwargs["show_progress"])
            self.assertNotIn("show", build_structure_mask.call_args.kwargs)

    def test_skew_truncation_matches_materialized_row_offsets(self):
        sample_positions = list(np.linspace(20.0, 180.0, 41, dtype=np.float32))
        widths = np.linspace(1.0, 4.0, 41, dtype=np.float32)
        step_scales = np.linspace(-1.5, 1.5, 41, dtype=np.float32)
        row_offsets = np.linspace(-31.5, 31.5, 64, dtype=np.float32)
        skew = 0.07
        image_width = 200

        materialized = row_offsets[None, :] * (
            np.float32(skew) * step_scales[:, None]
        )
        valid = (
            np.asarray(sample_positions) + np.min(materialized, axis=1) - widths >= 0
        ) & (
            np.asarray(sample_positions) + np.max(materialized, axis=1) + widths
            <= image_width - 1
        )
        first = int(np.argmax(valid))
        last = int(len(valid) - 1 - np.argmax(valid[::-1]))

        positions, result_widths = preview.truncate_sample_positions_for_skew(
            sample_positions,
            widths,
            skew=skew,
            step_scales=step_scales,
            row_offsets=row_offsets,
            width=image_width,
        )

        np.testing.assert_array_equal(positions, sample_positions[first : last + 1])
        np.testing.assert_array_equal(result_widths, widths[first : last + 1])

    def test_uint8_binning_matches_float32_implementation(self):
        rng = np.random.default_rng(14)
        image = rng.integers(0, 256, size=(13, 17, 3), dtype=np.uint8)
        expected = np.stack(
            [
                preview.bin2to1(image[:, :, channel].astype(np.float32))
                for channel in range(3)
            ],
            axis=2,
        ).astype(np.uint8)

        result = preview.bin2to1_rgb_u8(image)

        np.testing.assert_array_equal(result, expected)

    def test_color_transform_matches_explicit_white_balanced_transform(self):
        rng = np.random.default_rng(8)
        raw = rng.random((9, 7, 3), dtype=np.float32) * 65535.0
        white_balance = np.array([1.3, 0.9, 1.1], dtype=np.float32)
        expected = np.tensordot(
            raw * white_balance.reshape(1, 1, 3),
            preview.COLOR_CORRECTION.T,
            axes=([2], [0]),
        )

        result = preview.apply_color_transform(raw, white_balance)

        np.testing.assert_array_equal(result, expected)

    def test_blocked_scene_feather_matches_whole_image_blur(self):
        mask = np.zeros((17, 1001), dtype=bool)
        mask[3:14, 190:811:37] = True
        expected = preview.cv2.GaussianBlur(
            mask.astype(np.float32),
            (0, 0),
            sigmaX=preview.JITTER_SCENE_PROTECTION_FEATHER_COLUMNS,
            sigmaY=preview.JITTER_SCENE_PROTECTION_FEATHER_BANDS,
            borderType=preview.cv2.BORDER_REPLICATE,
        )
        expected = np.clip(expected, 0.0, 1.0).astype(np.float16)

        result = preview._feather_jitter_scene_protection(
            mask,
            block_columns=127,
        )

        np.testing.assert_array_equal(result, expected)

    def test_stack_preview_is_disk_backed_and_preserves_chunk_order(self):
        first = np.full((3, 5, 3), 17, dtype=np.uint8)
        second = np.full((3, 5, 3), 93, dtype=np.uint8)
        expected = np.vstack((first, second))

        with TemporaryDirectory() as directory:
            with preview.DiskBackedStackPreview(Path(directory), 2) as stack:
                stack.add_chunk(1, second)
                stack.add_chunk(0, first)
                self.assertIsInstance(stack.array, np.memmap)

                with patch.object(preview, "write_jpeg") as write_jpeg:
                    self.assertTrue(
                        stack.write(Path(directory) / "stack.jpg", np.empty(0))
                    )
                    written = np.array(write_jpeg.call_args.args[1], copy=True)

        np.testing.assert_array_equal(written, expected)

    def test_disk_backed_stack_is_accepted_by_jpeg_encoder(self):
        chunk = np.full((8, 12, 3), 127, dtype=np.uint8)
        exif = preview.build_jpeg_exif(None, None)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "stack.jpg"
            with preview.DiskBackedStackPreview(Path(directory), 1) as stack:
                stack.add_chunk(0, chunk)
                self.assertTrue(stack.write(path, exif))

            decoded = preview.cv2.imread(str(path), preview.cv2.IMREAD_COLOR)

        self.assertEqual(decoded.shape, chunk.shape)


if __name__ == "__main__":
    unittest.main()
