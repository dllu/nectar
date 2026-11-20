import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import preview


CAPTURE = "2026-06-16-02-21-07"


def write_chunk(
    directory: Path,
    chunk: int,
    digest: str,
    *,
    capture: str = CAPTURE,
) -> tuple[Path, Path]:
    stem = f"{capture}_NECTA{chunk:04d}_{digest}"
    dng_path = directory / f"{stem}.dng"
    jpeg_path = directory / f"{stem}.jpg"
    dng_path.write_bytes(f"dng {digest}".encode())
    jpeg_path.write_bytes(f"jpeg {digest}".encode())
    return dng_path, jpeg_path


def write_stack(
    directory: Path,
    contents: bytes = b"stack",
    *,
    capture: str = CAPTURE,
) -> Path:
    path = directory / f"{capture}_stack.jpg"
    path.write_bytes(contents)
    return path


class OutputReplacementTest(unittest.TestCase):
    def test_complete_generation_replaces_all_stale_capture_outputs(self):
        with TemporaryDirectory() as root:
            output_dir = Path(root) / "output"
            staging_dir = Path(root) / "staging"
            output_dir.mkdir()
            staging_dir.mkdir()

            old_digest = "a" * 40
            old_paths = [
                *write_chunk(output_dir, 0, old_digest),
                *write_chunk(output_dir, 1, "b" * 40),
                write_stack(output_dir, b"old stack"),
            ]
            unrelated = output_dir / "another-capture_NECTA0000_notes.txt"
            unrelated.write_text("keep me", encoding="utf-8")
            other_capture = "2026-06-16-02-21-08"
            other_dng, other_jpeg = write_chunk(
                output_dir,
                0,
                "d" * 40,
                capture=other_capture,
            )
            other_stack = write_stack(output_dir, capture=other_capture)

            new_digest = "c" * 40
            new_dng, new_jpeg = write_chunk(staging_dir, 0, new_digest)
            new_stack = write_stack(staging_dir, b"new stack")

            removed, migrated = preview.replace_capture_outputs(
                output_dir,
                staging_dir,
                CAPTURE,
            )

            self.assertEqual(removed, 4)
            self.assertEqual(migrated, 0)
            self.assertTrue((output_dir / new_dng.name).is_file())
            self.assertTrue((output_dir / new_jpeg.name).is_file())
            self.assertEqual((output_dir / new_stack.name).read_bytes(), b"new stack")
            self.assertTrue(unrelated.is_file())
            self.assertTrue(other_dng.is_file())
            self.assertTrue(other_jpeg.is_file())
            self.assertTrue(other_stack.is_file())
            for old_path in old_paths[:-1]:
                self.assertFalse(old_path.exists())

    def test_newest_xmp_for_each_surviving_chunk_is_carried_forward(self):
        with TemporaryDirectory() as root:
            output_dir = Path(root) / "output"
            staging_dir = Path(root) / "staging"
            output_dir.mkdir()
            staging_dir.mkdir()

            older_dng, _older_jpeg = write_chunk(output_dir, 0, "a" * 40)
            newer_dng, _newer_jpeg = write_chunk(output_dir, 0, "b" * 40)
            write_stack(output_dir)
            older_xmp = output_dir / f"{older_dng.name}.xmp"
            newer_xmp = output_dir / f"{newer_dng.name}.xmp"
            older_xmp.write_text(
                f'<rdf xmpMM:DerivedFrom="{older_dng.name}" marker="older"/>',
                encoding="utf-8",
            )
            newer_xmp.write_text(
                f'<rdf xmpMM:DerivedFrom="{newer_dng.name}" marker="newer"/>',
                encoding="utf-8",
            )
            os.utime(older_xmp, ns=(1_000_000_000, 1_000_000_000))
            os.utime(newer_xmp, ns=(2_000_000_000, 2_000_000_000))

            new_dng, _new_jpeg = write_chunk(staging_dir, 0, "c" * 40)
            write_stack(staging_dir)

            _removed, migrated = preview.replace_capture_outputs(
                output_dir,
                staging_dir,
                CAPTURE,
            )

            new_xmp = output_dir / f"{new_dng.name}.xmp"
            self.assertEqual(migrated, 1)
            self.assertTrue(new_xmp.is_file())
            contents = new_xmp.read_text(encoding="utf-8")
            self.assertIn('marker="newer"', contents)
            self.assertIn(f'xmpMM:DerivedFrom="{new_dng.name}"', contents)
            self.assertNotIn("b" * 40, contents)
            self.assertFalse(older_xmp.exists())
            self.assertFalse(newer_xmp.exists())

    def test_incomplete_staging_does_not_touch_existing_outputs(self):
        with TemporaryDirectory() as root:
            output_dir = Path(root) / "output"
            staging_dir = Path(root) / "staging"
            output_dir.mkdir()
            staging_dir.mkdir()

            old_dng, old_jpeg = write_chunk(output_dir, 0, "a" * 40)
            old_stack = write_stack(output_dir, b"old stack")
            incomplete_dng, incomplete_jpeg = write_chunk(
                staging_dir,
                0,
                "b" * 40,
            )
            incomplete_jpeg.unlink()
            write_stack(staging_dir, b"new stack")

            with self.assertRaisesRegex(RuntimeError, "missing JPEGs"):
                preview.replace_capture_outputs(
                    output_dir,
                    staging_dir,
                    CAPTURE,
                )

            self.assertTrue(old_dng.is_file())
            self.assertTrue(old_jpeg.is_file())
            self.assertEqual(old_stack.read_bytes(), b"old stack")
            self.assertTrue(incomplete_dng.is_file())


if __name__ == "__main__":
    unittest.main()
