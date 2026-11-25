import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from raw_io import RawFrameStore, load_raw_frame


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

    def test_cached_frames_preserve_raw_bits_and_prefix_reads(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"
            raw = np.tile(np.array([0xffff, 0x0001, 0xabcd, 0], dtype="<u2"), 512)
            for budget in (raw.nbytes, raw.nbytes // 4):
                with self.subTest(budget=budget):
                    store = RawFrameStore(budget, raw.nbytes)
                    raw.tofile(path)
                    expected = load_raw_frame(path, rows=4)
                    store.remember(path, expected)
                    path.unlink()
                    np.testing.assert_array_equal(store.read(path, rows=4), expected)
                    prefix = store.read(path, rows=4, columns=5, output_dtype=np.float32)
                    self.assertEqual(prefix.dtype, np.float32)
                    np.testing.assert_array_equal(prefix, expected[:, :5])
                    with self.assertRaisesRegex(ValueError, "expected 513"):
                        store.read(path, rows=4, columns=513)
                    self.assertLessEqual(store.used, budget)
                    store.retain([])
                    self.assertEqual(store.used, 0)

    def test_partial_cache_falls_back_to_disk_within_budget(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.random.default_rng(0).integers(65536, size=(64, 4), dtype="<u2")
            budget = raw.nbytes + 32
            store = RawFrameStore(budget, 2 * raw.nbytes)
            for i in range(2):
                path = root / f"{i}.bin"
                (raw + i).tofile(path)
                store.remember(path, load_raw_frame(path, rows=4))
            self.assertEqual(len(store.frames), 1)
            self.assertLessEqual(store.used, budget)
            np.testing.assert_array_equal(store.read(path, rows=4), (raw + 1).T)


if __name__ == "__main__":
    unittest.main()
