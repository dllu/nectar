import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from raw_io import load_raw_frame


class RawIoTest(unittest.TestCase):
    def test_loads_column_major_file_as_sensor_rows(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"
            expected = np.arange(24, dtype=np.uint16).reshape(4, 6)
            np.asarray(expected.T, dtype="<u2").tofile(path)

            loaded = load_raw_frame(path, rows=4)

            self.assertEqual(loaded.dtype, np.dtype("<u2"))
            np.testing.assert_array_equal(loaded, expected)

    def test_can_load_an_exact_column_prefix_and_convert_dtype(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"
            expected = np.arange(24, dtype=np.uint16).reshape(4, 6)
            np.asarray(expected.T, dtype="<u2").tofile(path)

            loaded = load_raw_frame(
                path,
                rows=4,
                columns=3,
                output_dtype=np.float32,
            )

            self.assertEqual(loaded.dtype, np.float32)
            np.testing.assert_array_equal(loaded, expected[:, :3])

    def test_rejects_incomplete_or_short_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "incomplete.bin"
            incomplete.write_bytes(b"\0" * 7)
            with self.assertRaisesRegex(ValueError, "expected a multiple"):
                load_raw_frame(incomplete, rows=4)

            short = root / "short.bin"
            np.zeros((2, 4), dtype="<u2").tofile(short)
            with self.assertRaisesRegex(ValueError, "expected 3"):
                load_raw_frame(short, rows=4, columns=3)


if __name__ == "__main__":
    unittest.main()
