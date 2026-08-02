import hashlib
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

import preview


class DngMetadataTest(unittest.TestCase):
    def test_capture_datetime_is_parsed_from_directory_prefix(self):
        self.assertEqual(
            preview.parse_capture_datetime(
                Path("/captures/2026-06-16-02-21-07-copy")
            ),
            datetime(2026, 6, 16, 2, 21, 7),
        )
        self.assertIsNone(preview.parse_capture_datetime(Path("/captures/nyc1")))
        self.assertIsNone(
            preview.parse_capture_datetime(Path("/captures/2026-02-31-02-21-07"))
        )

    def test_exposure_time_is_parsed_from_100_ns_shutter_ticks(self):
        with TemporaryDirectory() as root:
            capture_dir = Path(root)
            (capture_dir / "capture_settings.txt").write_text(
                "cds_gain=1\nanalog_gain=10\nshutterspeed=452\n",
                encoding="utf-8",
            )

            self.assertEqual(
                preview.parse_exposure_time(capture_dir),
                (452, 10_000_000),
            )

    def test_dng_contains_capture_metadata_and_hashes_final_bytes(self):
        with TemporaryDirectory() as root:
            capture_dir = Path(root) / "2026-06-16-02-21-07"
            capture_dir.mkdir()
            output_dir = Path(root) / "output"
            (capture_dir / "capture_settings.txt").write_text(
                "shutterspeed=452\n",
                encoding="utf-8",
            )
            raw = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)

            dng_path, digest = preview.write_hashed_linear_dng(
                output_dir,
                "2026-06-16-02-21-07_NECTA0000",
                raw,
                np.ones(3, dtype=np.float32),
                capture_dir=capture_dir,
            )

            self.assertEqual(dng_path.parent, output_dir)
            actual_digest = hashlib.sha1(
                dng_path.read_bytes(),
                usedforsecurity=False,
            ).hexdigest()
            self.assertEqual(digest, actual_digest)
            self.assertTrue(dng_path.stem.endswith(actual_digest))
            with tifffile.TiffFile(dng_path) as dng:
                tags = dng.pages[0].tags
                self.assertEqual(
                    tags[306].value,
                    "2026:06:16 02:21:07",
                )
                self.assertEqual(
                    tags[36867].value,
                    "2026:06:16 02:21:07",
                )
                self.assertEqual(tags[33434].value, (452, 10_000_000))

    def test_unavailable_capture_metadata_is_omitted(self):
        with TemporaryDirectory() as root:
            capture_dir = Path(root) / "legacy-capture"
            capture_dir.mkdir()
            (capture_dir / "capture_settings.txt").write_text(
                "shutterspeed=invalid\n",
                encoding="utf-8",
            )
            raw = np.zeros((4, 6, 3), dtype=np.uint16)

            dng_path, _digest = preview.write_hashed_linear_dng(
                capture_dir,
                "legacy-capture_NECTA0000",
                raw,
                np.ones(3, dtype=np.float32),
            )

            with tifffile.TiffFile(dng_path) as dng:
                tags = dng.pages[0].tags
                self.assertIsNone(tags.get(306))
                self.assertIsNone(tags.get(36867))
                self.assertIsNone(tags.get(33434))


if __name__ == "__main__":
    unittest.main()
