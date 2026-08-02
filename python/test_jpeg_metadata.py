import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import ExifTags, Image

import preview


class JpegMetadataTest(unittest.TestCase):
    def test_frame_and_stack_jpegs_contain_capture_metadata(self):
        captured_at = datetime(2026, 6, 16, 2, 21, 7)
        exposure_time = (452, 10_000_000)
        exif_payload = preview.build_jpeg_exif(captured_at, exposure_time)
        image = np.zeros((16, 24, 3), dtype=np.uint8)

        with TemporaryDirectory() as root:
            paths = [
                Path(root) / "2026-06-16-02-21-07_NECTA0000_hash.jpg",
                Path(root) / "2026-06-16-02-21-07_stack.jpg",
            ]
            for path in paths:
                preview.write_jpeg(path, image, exif_payload)

            for path in paths:
                with self.subTest(path=path), Image.open(path) as jpeg:
                    exif = jpeg.getexif()
                    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                    self.assertEqual(exif[ExifTags.Base.Make], preview.CAMERA_MAKE)
                    self.assertEqual(
                        exif[ExifTags.Base.Model],
                        preview.CAMERA_MODEL_NAME,
                    )
                    self.assertEqual(exif[ExifTags.Base.Orientation], 1)
                    self.assertEqual(
                        exif[ExifTags.Base.Software],
                        "nectar-preview",
                    )
                    self.assertEqual(
                        exif[ExifTags.Base.DateTime],
                        "2026:06:16 02:21:07",
                    )
                    self.assertEqual(
                        exif_ifd[ExifTags.Base.DateTimeOriginal],
                        "2026:06:16 02:21:07",
                    )
                    exposure = exif_ifd[ExifTags.Base.ExposureTime]
                    self.assertEqual(
                        exposure.numerator * exposure_time[1],
                        exposure_time[0] * exposure.denominator,
                    )

    def test_unavailable_capture_fields_are_omitted(self):
        exif_payload = preview.build_jpeg_exif(None, None)
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        with TemporaryDirectory() as root:
            path = Path(root) / "legacy_stack.jpg"
            preview.write_jpeg(path, image, exif_payload)

            with Image.open(path) as jpeg:
                exif = jpeg.getexif()
                exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                self.assertNotIn(ExifTags.Base.DateTime, exif)
                self.assertNotIn(ExifTags.Base.DateTimeOriginal, exif_ifd)
                self.assertNotIn(ExifTags.Base.ExposureTime, exif_ifd)


if __name__ == "__main__":
    unittest.main()
