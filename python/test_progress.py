import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import preview


class ProgressTest(unittest.TestCase):
    def test_nested_progress_displays_only_the_outer_bar(self):
        displayed = []

        def fake_tqdm(iterable, **kwargs):
            displayed.append(kwargs["desc"])
            return iterable

        def nested_items():
            yield from preview.progress(range(3), desc="inner")

        with patch.object(preview, "tqdm", side_effect=fake_tqdm):
            self.assertEqual(
                list(preview.progress(nested_items(), desc="outer")),
                [0, 1, 2],
            )

        self.assertEqual(displayed, ["outer"])

    def test_disabled_progress_does_not_create_a_bar(self):
        with patch.object(preview, "tqdm") as tqdm:
            self.assertEqual(
                list(preview.progress(range(2), desc="disabled", show=False)),
                [0, 1],
            )

        tqdm.assert_not_called()

    def test_main_disables_progress_for_multiple_captures(self):
        with TemporaryDirectory() as root:
            captures = [Path(root) / name for name in ("capture-a", "capture-b")]
            output_dir = Path(root) / "output"
            for capture in captures:
                capture.mkdir()

            with patch.object(preview, "process_preview") as process_preview:
                preview.main(
                    [
                        "--output-dir",
                        str(output_dir),
                        str(captures[0]),
                    ]
                )
                process_preview.assert_called_once_with(
                    captures[0],
                    output_dir=output_dir,
                )

            widths = {
                captures[0]: 12345,
                captures[1]: 67890,
            }
            with patch.object(
                preview,
                "process_preview",
                side_effect=lambda capture, **_kwargs: widths[capture],
            ) as process_preview:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    preview.main(
                        [
                            "--output-dir",
                            str(output_dir),
                            *(str(capture) for capture in captures),
                        ]
                    )
                self.assertEqual(process_preview.call_count, 2)
                process_preview.assert_has_calls(
                    [
                        call(
                            captures[0],
                            show_progress=False,
                            output_dir=output_dir,
                        ),
                        call(
                            captures[1],
                            show_progress=False,
                            output_dir=output_dir,
                        ),
                    ],
                    any_order=True,
                )
                self.assertCountEqual(
                    stdout.getvalue().splitlines(),
                    [
                        f"{captures[0]} final_width=12345",
                        f"{captures[1]} final_width=67890",
                    ],
                )

    def test_output_directory_has_the_expected_default(self):
        args = preview._parse_args([])
        self.assertEqual(args.output_dir, preview.DEFAULT_OUTPUT_DIR)

    def test_output_flip_defaults_to_the_camera_mount_era(self):
        self.assertFalse(
            preview.resolve_flip_output_180(Path("2025-12-31-23-59-59"))
        )
        self.assertTrue(
            preview.resolve_flip_output_180(Path("2026-01-01-00-00-00"))
        )
        self.assertTrue(
            preview.resolve_flip_output_180(Path("2030-04-05-06-07-08-extra"))
        )
        self.assertFalse(preview.resolve_flip_output_180(Path("undated-capture")))

    def test_output_flip_cli_flags_override_the_camera_mount_era(self):
        self.assertTrue(
            preview.resolve_flip_output_180(
                Path("2025-12-31-23-59-59"), override=True
            )
        )
        self.assertFalse(
            preview.resolve_flip_output_180(
                Path("2026-01-01-00-00-00"), override=False
            )
        )
        self.assertIsNone(preview._parse_args([]).flip_output_180)
        self.assertTrue(
            preview._parse_args(["--flip-output-180"]).flip_output_180
        )
        self.assertFalse(
            preview._parse_args(["--no-flip-output-180"]).flip_output_180
        )


if __name__ == "__main__":
    unittest.main()
