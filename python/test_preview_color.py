import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

import preview
from image_processing import CAMERA_D65_NEUTRAL, NeutralReference, fuse_aligned_greens


class PreviewColorTest(unittest.TestCase):
    def test_shadow_reference_neutralizes_dark_patch_and_keeps_bright_samples(self):
        dark = np.float32([2200, 1950, 450])
        neutral = np.float32([1.1, 1, 0.75])
        ref = NeutralReference.from_shadow(
            dark, neutral, shadow_fade_end=10000, shadow_chroma_threshold=3000
        )
        anchor, end = ref.shadow_range
        midpoint = np.float32([(anchor + end) / 2, 0, 0]) * neutral
        raw = np.float32([[dark, midpoint, [0, 0, 0], [100, 10000, 220], [65535, 0, 1]]])
        corrected = raw.copy()
        ref.correct(corrected)
        np.testing.assert_allclose(corrected[0, 0] / neutral, anchor, rtol=1e-6)
        np.testing.assert_allclose(corrected[0, 1] - midpoint, 0.5 * ref.lift, atol=0.001)
        np.testing.assert_array_equal(corrected[0, 2:], raw[0, 2:])
        with self.assertRaises(ValueError):
            NeutralReference.from_shadow(dark, neutral, shadow_fade_end=anchor)

    def test_bundled_reference_matches_capture_settings_and_local_file_takes_precedence(self):
        with TemporaryDirectory() as root:
            capture = Path(root) / "2026-06-16-01-59-33"
            capture.mkdir()
            self.assertIsNone(preview.capture_neutral_reference(capture))
            settings = capture / "capture_settings.txt"
            settings.write_text("shutterspeed=452\nanalog_gain=10\ncds_gain=1\n")
            ref = preview.capture_neutral_reference(capture)
            self.assertEqual(ref.shadow_range[1], 10000)
            settings.write_text("shutterspeed=451\nanalog_gain=10\ncds_gain=1\n")
            self.assertIsNone(preview.capture_neutral_reference(capture))
            local = {"dark_rgb": [100, 200, 150], "light_rgb": [1000, 1100, 1050]}
            (capture / "color_reference.json").write_text(json.dumps(local))
            ref = preview.capture_neutral_reference(capture)
            np.testing.assert_array_equal(ref.neutral, np.ones(3))
            self.assertIsNone(ref.shadow_range)

    def test_linear_dng_native_samples_and_both_matrix_paths(self):
        raw = np.arange(90, dtype=np.uint16).reshape(5, 6, 3)
        gains = np.array([0.8, 1.0, 1.6])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "camera.dng"
            preview.write_linear_dng(path, raw, gains)
            with tifffile.TiffFile(path) as f:
                page = f.pages[0]
                self.assertEqual(page.photometric, tifffile.PHOTOMETRIC.LINEAR_RAW)
                self.assertEqual(page.samplesperpixel, 3)
                self.assertFalse(page.extrasamples)
                np.testing.assert_array_equal(page.asarray(), raw[::-1])

                def rationals(key):
                    pairs = np.array(page.tags[key].value).reshape(-1, 2)
                    return pairs[:, 0] / pairs[:, 1]

                neutral = rationals("AsShotNeutral")
                cm = rationals("ColorMatrix1").reshape(3, 3)
                fm = rationals("ForwardMatrix1").reshape(3, 3)

        np.testing.assert_allclose(neutral * gains, gains.min(), atol=1e-6)
        np.testing.assert_allclose(
            cm @ [0.95047, 1, 1.08883], CAMERA_D65_NEUTRAL, atol=2e-6
        )
        rgb = np.array([0.2, 0.5, 0.7])
        camera = cm @ np.linalg.solve(preview.XYZ_D65_TO_LINEAR_SRGB, rgb)
        np.testing.assert_allclose(
            fm @ (camera / CAMERA_D65_NEUTRAL),
            preview.LINEAR_SRGB_TO_XYZ_D50 @ rgb,
            atol=2e-6,
        )

    def test_measured_neutral_axis_keeps_black_and_highlight_headroom(self):
        dark = np.array([1200, 900, 500])
        light = np.array([13200, 10900, 8500])
        ref = NeutralReference.from_patches(dark, light)
        # A third exposure on the line was not used for fitting.
        mid = 0.35 * dark + 0.65 * light
        pixels = np.float32([[dark, mid, light, [0, 0, 0], [65535] * 3, [40, 0, 0]]])
        ref.correct(pixels)
        balanced = pixels[0, :3] / ref.neutral
        np.testing.assert_allclose(
            balanced, np.repeat(balanced[:, 1:2], 3, axis=1), rtol=2e-6
        )
        np.testing.assert_array_equal(pixels[0, 3], 0)
        self.assertLessEqual(pixels.max(), 65535)
        self.assertLess(pixels[0, 5].max(), 41)
        with self.assertRaises(ValueError):
            NeutralReference.from_patches(light, dark)

    def test_shadow_chroma_taper_preserves_measured_bright_colors_and_neutrals(self):
        neutral = np.array([1.25, 1, 0.75], dtype=np.float32)
        dark, light = 1000 * neutral, 10000 * neutral
        ref = NeutralReference.from_patches(dark, light, shadow_chroma_threshold=3000)
        raw = np.float32([[[0, 400, 0], [0, 3500, 0], dark, [0, 0, 0]]])
        corrected = raw.copy()
        ref.correct(corrected)
        before = raw[0, 0] / neutral
        after = corrected[0, 0] / neutral
        self.assertLess(np.ptp(after), 0.06 * np.ptp(before))
        self.assertAlmostEqual(float(after.mean()), float(before.mean()), places=4)
        np.testing.assert_array_equal(corrected[0, 1:], raw[0, 1:])
        with self.assertRaises(ValueError):
            NeutralReference.from_patches(dark, light, shadow_chroma_threshold=-1)

    def test_green_response_can_change_with_height_and_either_phase_can_clip(self):
        odd = np.broadcast_to(
            np.array([0, 320, 433, 1200, 2700])[:, None], (5, 11)
        ).astype("f")
        even = np.broadcast_to(np.array([0, 0, 1900, 411, 0])[:, None], (5, 11)).astype(
            "f"
        )
        expected = (odd**2 + even**2) / np.maximum(odd + even, 1)
        for a, b in ((odd, even), (even, odd)):
            np.testing.assert_allclose(fuse_aligned_greens(a, b), expected)
        below_black = np.full((2, 5), -199.9, dtype=np.float32)
        np.testing.assert_allclose(
            fuse_aligned_greens(below_black, -below_black), 199.9
        )
        # An isolated detail on an odd native row is blurred by interpolation
        # from its two even neighbors. Recover its contrast above the flat field.
        a = np.full((6, 11), 2400, np.float32)
        b = a.copy()
        a[3, 5] += 400
        fused = fuse_aligned_greens(a, b)
        self.assertGreater(fused[3, 5], 2780)
        np.testing.assert_array_equal(fused[:, :5], 2400)

    def test_print_cuts_match_whole_green_in_both_scan_directions(self):
        class Plane:
            def __init__(self, pixels):
                self.pixels = pixels
                self.height, self.width = pixels.shape

            def __getitem__(self, key):
                return self.pixels[key]

        rng = np.random.default_rng(43)
        planes = [Plane(rng.integers(0, 400, (12, 600))) for _ in range(4)]
        steps = np.arange(3, 570, 1.8)
        for positions in (steps, steps[::-1]):

            def render(ranges):
                chunks = preview.sample_motion_corrected_chunks(
                    *planes,
                    positions,
                    np.full(len(steps), 1.8),
                    fuse_green=True,
                    chunk_ranges=ranges,
                    pitch_offsets=np.sin(np.arange(600) * 0.02).astype("f"),
                    show_progress=False,
                )
                return np.concatenate([rgb for _, rgb in chunks], axis=1)

            reference = render(None)
            actual = render([(0, 7), (7, 261), (261, len(steps))])
            np.testing.assert_array_equal(actual, reference)

    def test_preview_applies_srgb_transfer_and_correct_orientation(self):
        levels = np.array([0, 0.003, 0.18, 1], dtype=np.float32)
        # Constant columns within each band avoid the unsharp-mask edges.
        raw = np.broadcast_to(np.repeat(levels, 24)[None, :, None], (135, 96, 3)).copy()
        for flip in (False, True):
            bgr = preview.render_preview_bgr(
                raw, np.ones(3), 0, 1, flip_output_180=flip
            )
            if flip:
                bgr = bgr[:, ::-1]
            for y in (0, 63, 64, 128):
                np.testing.assert_array_equal(bgr[y, 12::24, 0], [0, 10, 118, 255])

    def test_print_jpegs_reproduce_from_dng_in_either_orientation(self):
        rng = np.random.default_rng(7)
        raw = rng.uniform(-20, 62000, size=(27, 32, 3)).astype(np.float32)
        wb = np.array([0.9, 1, 1.3])
        exif = preview.build_jpeg_exif(None, None)
        with TemporaryDirectory() as root:
            directory = Path(root)
            for flip in (False, True):
                prefix = f"frame-{flip}"
                small = preview.render_output_chunk(
                    raw, output_dir=directory, name_prefix=prefix, white_balance=wb,
                    p2=0, p98=50000, capture_dir=directory, jpeg_exif=exif,
                    flip_output_180=flip,
                )
                dng = next(directory.glob(f"{prefix}*.dng"))
                camera = tifffile.imread(dng)
                expected = raw[:, ::-1] if flip else raw[::-1]
                np.testing.assert_array_equal(camera, np.clip(expected, 0, 65535).astype("uint16"))
                bgr = preview.render_preview_bgr(camera[::-1], wb, 0, 50000, flip_output_180=False)
                other = directory / "from-dng.jpg"
                preview.write_jpeg(other, bgr, exif)
                self.assertEqual(dng.with_suffix(".jpg").read_bytes(), other.read_bytes())
                np.testing.assert_array_equal(small, preview.bin2to1_rgb_u8(bgr))


if __name__ == "__main__":
    unittest.main()
