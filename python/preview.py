#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import bottleneck as bn
import cv2
import matplotlib

# Preview diagnostics are written to disk, including from worker threads when
# multiple captures are requested. A GUI backend is both unnecessary and
# unsafe there (Tk can abort during cross-thread teardown).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from PIL import ExifTags, Image, TiffImagePlugin
from scipy.optimize import least_squares, minimize_scalar
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import itertools
import tifffile
from concurrent.futures import ThreadPoolExecutor, as_completed

from image_processing import (
    COLOR_CORRECTION,
    RAW_COLOR_CORRECTION,
    apply_color_transform,
)
from raw_io import load_raw_frame

CAMERA_MAKE = "Alkeria"
CAMERA_MODEL_NAME = "Alkeria Necta N4K2-7C"
CAPTURE_DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
DNG_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"
SHUTTER_TICKS_PER_SECOND = 10_000_000  # Capture UI stores shutter in 100 ns ticks.
ENABLE_COLUMN_JITTER_CORRECTION = True
# None selects the hardware-era default from the capture timestamp: the camera
# was mounted upright through 2025 and upside down from 2026 onward.
FLIP_OUTPUT_180: bool | None = None
STRUCTURE_MASK_DOWNSAMPLE = 8
STRUCTURE_MASK_THRESHOLD = 0.05
STRUCTURE_MASK_DILATE_PX = 0
PROCESSING_PADDING_COLUMNS = 4096
MOVING_REGION_BLOCK_COLUMNS = 4096
STRUCTURE_MASK_BLOCK_COLUMNS = 4096
WHITE_BALANCE_SAMPLE_COLUMNS = 8192
ENABLE_JITTER_DEBUG_PLOT = False
ENABLE_SKEW_DEBUG_OUTPUT = False
DEFAULT_SKEW_HOUGH_EXTENTS = (-0.05, 0.05)
JITTER_DEBUG_ROW = 2000
JITTER_VERTICAL_BANDS = 256
JITTER_FINE_TEMPORAL_WINDOW = 31
JITTER_TEMPORAL_WINDOW = 1023
JITTER_BLOCK_COLUMNS = 4096
JITTER_MAX_CLIPPED_FRACTION = 0.10
JITTER_BLACK_MARGIN = 64.0
JITTER_WHITE_MARGIN_FRACTION = 0.03
JITTER_WHITE_MARGIN_MIN = 256.0
JITTER_FINE_MAX_CORRECTION_FRACTION = 0.02
JITTER_BROAD_MAX_CORRECTION_FRACTION = 0.01
JITTER_MODEL_MAX_CORRECTION_FRACTION = 0.005
JITTER_MODEL_MIN_VALID_FRACTION = 0.25
JITTER_LOW_RANK_MAX_COMPONENTS = 64
JITTER_LOW_RANK_MAX_SAMPLE_COLUMNS = 32768
JITTER_LOW_RANK_MIN_TEMPORAL_PARTICIPATION = 0.10
JITTER_LOW_RANK_MIN_SPATIAL_PARTICIPATION = 0.05
JITTER_LOW_RANK_MIN_CHANNEL_POWER = 0.05
JITTER_LOW_RANK_MIN_NOISE_RATIO = 5.0
JITTER_LOW_RANK_WINSOR_SIGMA = 4.0
JITTER_LOW_RANK_MIN_ROBUST_SCALE = 1.0e-5
JITTER_RESIDUAL_MAX_CORRECTION_FRACTION = 0.006
JITTER_RESIDUAL_MIN_CHANNEL_FRACTION = 0.0001
JITTER_RESIDUAL_MIN_RELATIVE_CHANNEL_POWER = 0.10
JITTER_MAX_RELATIVE_SIGNAL_CORRECTION = 0.05
JITTER_ALIGNED_EDGE_THRESHOLD = 0.70
JITTER_ALIGNED_STRONG_EDGE_THRESHOLD = 3.0
JITTER_ALIGNED_EDGE_DILATE_COLUMNS = 1
JITTER_ALIGNED_MIN_GRADIENT_SCALE = 5.0
JITTER_SCENE_PROTECTION_DILATE_COLUMNS = 256
JITTER_SCENE_PROTECTION_FEATHER_COLUMNS = 3.0
JITTER_SCENE_PROTECTION_FEATHER_BANDS = 0.75
SPEED_CONSTRAINTS_JSON = "speed_constraints.json"
SPEED_CORRELATION_KNOT_SPACING = 16384.0
SPEED_CORRELATION_LOG_SIGMA = 0.03
SPEED_CORRELATION_CURVATURE_LENGTH = 8192.0
SPEED_CORRELATION_CURVATURE_SAMPLE_SPACING = 512.0
SPEED_CORRELATION_CURVATURE_LOG_SIGMA = 0.05
SPEED_CORRELATION_VERTICAL_HIGHPASS_SIGMA = 4.0
SPEED_CORRELATION_BAND_MIN_PEAK = 0.8
SPEED_CORRELATION_BAND_MIN_CONTRAST = 0.01
SPEED_CORRELATION_BAND_TRIM_FRACTION = 0.15
SPEED_CORRELATION_MIN_VALID_BANDS = 4
SPEED_CORRELATION_ESTIMATOR_AGREEMENT = 0.2
SPEED_CONSTRAINT_KNOT_SPACING = 8192.0
SPEED_CONSTRAINT_PRIOR_LOG_SIGMA = 0.2
SPEED_CONSTRAINT_CURVATURE_LENGTH = 8192.0
SPEED_CONSTRAINT_CURVATURE_SAMPLE_SPACING = 512.0
SPEED_CONSTRAINT_CURVATURE_LOG_SIGMA = 0.05
# Circle diameters and equal-width spans are both dimensions placed manually in
# the Bayer-binned annotation image. Model their uncertainty in those pixels so
# longer dimensions supply proportionally tighter relative constraints.
SPEED_ANNOTATION_DIMENSION_SIGMA_PIXELS = 2.5
OUTPUT_CHUNK_TARGET = 32768
EDGE_PADDING_SAMPLE_COLUMNS = 128
MAX_CONCURRENT_RENDERS = 2
JITTER_RED = 0
JITTER_GREEN1 = 1
JITTER_GREEN2 = 2
JITTER_BLUE = 3


_progress_state = threading.local()
_render_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RENDERS)
_stack_write_lock = threading.Lock()


def progress(iterable, *, show: bool = True, **tqdm_kwargs):
    """Display at most one progress bar per thread."""
    depth = getattr(_progress_state, "depth", 0)
    if not show or depth:
        yield from iterable
        return

    _progress_state.depth = depth + 1
    try:
        yield from tqdm(iterable, **tqdm_kwargs)
    finally:
        _progress_state.depth = depth


# DNG ColorMatrix values map CIE XYZ to camera-native coordinates, whereas
# COLOR_CORRECTION maps white-balanced camera coordinates to linear sRGB.
# These standard matrices bridge the two conventions. The D50 matrix is the
# ICC-adapted linear-sRGB matrix used for DNG's profile connection space.
XYZ_D65_TO_LINEAR_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

LINEAR_SRGB_TO_XYZ_D50 = np.array(
    [
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733],
    ],
    dtype=np.float64,
)

DNG_COLOR_MATRIX_1 = np.linalg.solve(
    COLOR_CORRECTION.astype(np.float64),
    XYZ_D65_TO_LINEAR_SRGB,
)
DNG_FORWARD_MATRIX_1 = LINEAR_SRGB_TO_XYZ_D50 @ COLOR_CORRECTION.astype(np.float64)


def _rational_values(
    values: np.ndarray, *, signed: bool, denominator: int = 1_000_000
) -> tuple[tuple[int, int], ...]:
    values = np.asarray(values, dtype=np.float64).ravel()
    if not signed and np.any(values < 0):
        raise ValueError("unsigned rational values must be non-negative")
    return tuple((int(round(value * denominator)), denominator) for value in values)


def parse_capture_datetime(capture_dir: Path) -> datetime | None:
    candidate = capture_dir.name[:19]
    try:
        captured_at = datetime.strptime(candidate, CAPTURE_DATETIME_FORMAT)
    except ValueError:
        return None
    if captured_at.strftime(CAPTURE_DATETIME_FORMAT) != candidate:
        return None
    return captured_at


def resolve_flip_output_180(
    capture_dir: Path,
    override: bool | None = FLIP_OUTPUT_180,
) -> bool:
    """Resolve the CLI override or the camera-mount era for one capture."""
    if override is not None:
        return bool(override)
    captured_at = parse_capture_datetime(Path(capture_dir))
    return captured_at is not None and captured_at.year >= 2026


def parse_exposure_time(capture_dir: Path) -> tuple[int, int] | None:
    try:
        settings = (capture_dir / "capture_settings.txt").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError):
        return None

    for line in settings.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "shutterspeed":
            try:
                shutter_ticks = int(value.strip())
            except ValueError:
                return None
            if not 0 <= shutter_ticks <= np.iinfo(np.uint32).max:
                return None
            return shutter_ticks, SHUTTER_TICKS_PER_SECOND
    return None


def build_jpeg_exif(
    captured_at: datetime | None,
    exposure_time: tuple[int, int] | None,
) -> np.ndarray:
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = CAMERA_MAKE
    exif[ExifTags.Base.Model] = CAMERA_MODEL_NAME
    exif[ExifTags.Base.Orientation] = 1
    exif[ExifTags.Base.Software] = "nectar-preview"

    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    if captured_at is not None:
        captured_at_text = captured_at.strftime(DNG_DATETIME_FORMAT)
        exif[ExifTags.Base.DateTime] = captured_at_text
        exif_ifd[ExifTags.Base.DateTimeOriginal] = captured_at_text
    if exposure_time is not None:
        exif_ifd[ExifTags.Base.ExposureTime] = TiffImagePlugin.IFDRational(
            *exposure_time
        )

    exif_bytes = exif.tobytes()
    if not exif_bytes.startswith(b"Exif\x00\x00"):
        raise RuntimeError("Pillow produced an invalid EXIF payload")
    return np.frombuffer(exif_bytes, dtype=np.uint8, offset=6)


def write_jpeg(path: Path, bgr: np.ndarray, exif: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwriteWithMetadata(
        str(path),
        bgr,
        [cv2.IMAGE_METADATA_EXIF],
        [exif],
    )
    if not written:
        raise OSError(f"failed to write JPEG: {path}")


class DiskBackedStackPreview:
    """Incrementally assemble a vertical stack without retaining it in RAM."""

    def __init__(
        self,
        directory: Path,
        chunk_count: int,
        *,
        chunk_widths=None,
    ):
        if chunk_count < 1:
            raise ValueError("chunk_count must be positive")
        self.directory = Path(directory)
        self.chunk_count = int(chunk_count)
        if chunk_widths is None:
            self._expected_widths = None
        else:
            expected_widths = np.asarray(chunk_widths, dtype=np.int64)
            if expected_widths.shape != (self.chunk_count,) or np.any(
                expected_widths < 1
            ):
                raise ValueError(
                    "chunk_widths must contain one positive width per chunk"
                )
            self._expected_widths = expected_widths
        self._file = None
        self._array = None
        self._chunk_shape = None
        self._written = np.zeros(self.chunk_count, dtype=bool)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def array(self) -> np.memmap | None:
        return self._array

    def add_chunk(self, index: int, chunk: np.ndarray) -> None:
        if not 0 <= index < self.chunk_count:
            raise IndexError("stack chunk index is out of range")
        chunk = np.asarray(chunk)
        if chunk.ndim != 3 or chunk.shape[2] != 3 or chunk.dtype != np.uint8:
            raise ValueError("stack chunks must be HxWx3 uint8 arrays")
        if self._expected_widths is not None:
            expected_width = int(self._expected_widths[index])
            if chunk.shape[1] != expected_width:
                raise ValueError(
                    f"stack chunk {index} has width {chunk.shape[1]}, "
                    f"expected {expected_width}"
                )

        if self._array is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._chunk_shape = chunk.shape
            # TemporaryFile is unlinked immediately on Unix, so even a killed
            # batch does not leave multi-gigabyte staging files behind.
            self._file = tempfile.TemporaryFile(dir=self.directory)
            stack_width = (
                int(np.max(self._expected_widths))
                if self._expected_widths is not None
                else chunk.shape[1]
            )
            stack_shape = (
                self.chunk_count * chunk.shape[0],
                stack_width,
                chunk.shape[2],
            )
            self._array = np.memmap(
                self._file,
                dtype=np.uint8,
                mode="w+",
                shape=stack_shape,
            )
        elif (
            chunk.shape[0] != self._chunk_shape[0]
            or chunk.shape[2] != self._chunk_shape[2]
            or (
                self._expected_widths is None
                and chunk.shape[1] != self._chunk_shape[1]
            )
        ):
            raise ValueError(
                f"stack chunk has shape {chunk.shape}, expected {self._chunk_shape}"
            )

        row_start = index * chunk.shape[0]
        row_stop = row_start + chunk.shape[0]
        stack_width = self._array.shape[1]
        column_start = (stack_width - chunk.shape[1]) // 2
        column_stop = column_start + chunk.shape[1]
        self._array[row_start:row_stop] = 0
        self._array[row_start:row_stop, column_start:column_stop] = chunk
        self._written[index] = True
        # Bound dirty page-cache growth while several captures render in
        # parallel. The clean file-backed pages remain reclaimable.
        self._array.flush()
        try:
            self._array._mmap.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError, ValueError):
            # madvise is an optional optimization on non-Linux platforms.
            pass

    def write(self, path: Path, exif: np.ndarray) -> bool:
        if self._array is None:
            return False
        if not np.all(self._written):
            missing = np.flatnonzero(~self._written)
            raise RuntimeError(f"stack preview is missing chunks {missing.tolist()}")
        self._array.flush()
        # OpenCV's encoder can allocate its own working buffers. Serialize only
        # this final encode while allowing capture analysis and chunk rendering
        # to remain parallel.
        with _stack_write_lock:
            write_jpeg(path, self._array, exif)
        return True

    def close(self) -> None:
        array = self._array
        self._array = None
        if array is not None:
            array.flush()
            mapping = array._mmap
            del array
            mapping.close()
        if self._file is not None:
            self._file.close()
            self._file = None


def write_linear_dng(
    path: Path,
    raw_rgb: np.ndarray,
    white_balance: np.ndarray,
    *,
    captured_at: datetime | None = None,
    exposure_time: tuple[int, int] | None = None,
) -> None:
    if tifffile is None:
        raise RuntimeError(
            "tifffile is required for DNG export. Install it via `uv pip install tifffile`."
        )

    flipped = raw_rgb[::-1, :, :]
    if flipped.dtype == np.uint16:
        linear = np.ascontiguousarray(flipped)
    else:
        # Writing directly into uint16 avoids a full-size clipped float32
        # temporary alongside the source and destination images.
        linear = np.empty(flipped.shape, dtype=np.uint16)
        np.clip(flipped, 0, 65535, out=linear, casting="unsafe")
    wb = np.asarray(white_balance, dtype=np.float64)
    safe_wb = np.clip(wb, 1e-6, None)
    as_shot_neutral = 1.0 / safe_wb
    if as_shot_neutral[1] != 0:
        as_shot_neutral /= as_shot_neutral[1]

    extratags = [
        (271, "s", len(CAMERA_MAKE) + 1, CAMERA_MAKE),  # Make
        (272, "s", len(CAMERA_MODEL_NAME) + 1, CAMERA_MODEL_NAME),  # Model
        (274, "H", 1, 1),  # Orientation = top-left
        (50706, "B", 4, (1, 4, 0, 0)),  # DNGVersion
        (50707, "B", 4, (1, 2, 0, 0)),  # DNGBackwardVersion
        (
            50708,
            "s",
            len(CAMERA_MODEL_NAME) + 1,
            CAMERA_MODEL_NAME,
        ),  # UniqueCameraModel
        (
            50721,
            "2i",
            9,
            _rational_values(DNG_COLOR_MATRIX_1, signed=True),
        ),  # ColorMatrix1: XYZ D65 -> camera
        (
            50728,
            "2I",
            3,
            _rational_values(as_shot_neutral, signed=False),
        ),  # AsShotNeutral
        (50778, "H", 1, 21),  # CalibrationIlluminant1 = D65
        (
            50964,
            "2i",
            9,
            _rational_values(DNG_FORWARD_MATRIX_1, signed=True),
        ),  # ForwardMatrix1: white-balanced camera -> XYZ D50
    ]
    captured_at_text = None
    if captured_at is not None:
        captured_at_text = captured_at.strftime(DNG_DATETIME_FORMAT)
        extratags.append(
            (
                36867,
                "s",
                len(captured_at_text) + 1,
                captured_at_text,
            )
        )  # DateTimeOriginal
    if exposure_time is not None:
        extratags.append((33434, "2I", 1, (exposure_time,)))  # ExposureTime

    tifffile.imwrite(
        path,
        linear,
        photometric=tifffile.PHOTOMETRIC.LINEAR_RAW,
        planarconfig="contig",
        metadata=None,
        software="nectar-preview",
        datetime=captured_at_text,
        extratags=extratags,
    )


def sha1sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_hashed_linear_dng(
    output_dir: Path,
    name_prefix: str,
    raw_rgb: np.ndarray,
    white_balance: np.ndarray,
    *,
    capture_dir: Path | None = None,
) -> tuple[Path, str]:
    """Write a DNG and append the SHA-1 of its finalized bytes to its name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = capture_dir if capture_dir is not None else output_dir
    with tempfile.NamedTemporaryFile(
        dir=output_dir,
        prefix=f".{name_prefix}_",
        suffix=".dng",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        write_linear_dng(
            temporary_path,
            raw_rgb,
            white_balance,
            captured_at=parse_capture_datetime(metadata_dir),
            exposure_time=parse_exposure_time(metadata_dir),
        )
        dng_sha1 = sha1sum(temporary_path)
        dng_path = output_dir / f"{name_prefix}_{dng_sha1}.dng"
        temporary_path.replace(dng_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return dng_path, dng_sha1


def _capture_output_match(
    path: Path,
    capture_timestamp: str,
) -> re.Match[str] | None:
    return re.fullmatch(
        rf"{re.escape(capture_timestamp)}_NECTA(?P<chunk>\d{{4,}})_"
        r"(?P<digest>[0-9a-f]{40})"
        r"(?P<suffix>\.dng(?:\.xmp)?|\.jpg)",
        path.name,
        flags=re.IGNORECASE,
    )


def _capture_output_files(output_dir: Path, capture_timestamp: str) -> list[Path]:
    if not output_dir.is_dir():
        return []
    stack_name = f"{capture_timestamp}_stack.jpg".casefold()
    return [
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and (
            path.name.casefold() == stack_name
            or _capture_output_match(path, capture_timestamp) is not None
        )
    ]


def _validated_staged_capture_outputs(
    staging_dir: Path,
    capture_timestamp: str,
) -> dict[int, tuple[Path, Path, str]]:
    dngs: dict[int, tuple[Path, str]] = {}
    jpegs: dict[int, tuple[Path, str]] = {}
    for path in staging_dir.iterdir():
        match = _capture_output_match(path, capture_timestamp)
        if match is None:
            continue
        chunk = int(match.group("chunk"))
        digest = match.group("digest").lower()
        suffix = match.group("suffix").lower()
        if suffix == ".dng":
            collection = dngs
        elif suffix == ".jpg":
            collection = jpegs
        else:
            continue
        if chunk in collection:
            raise RuntimeError(f"multiple staged outputs for NECTA{chunk:04d}")
        collection[chunk] = (path, digest)

    if not dngs:
        raise RuntimeError("render produced no DNG files")
    if dngs.keys() != jpegs.keys():
        missing_jpegs = sorted(dngs.keys() - jpegs.keys())
        missing_dngs = sorted(jpegs.keys() - dngs.keys())
        details = []
        if missing_jpegs:
            details.append(f"missing JPEGs for chunks {missing_jpegs}")
        if missing_dngs:
            details.append(f"missing DNGs for chunks {missing_dngs}")
        raise RuntimeError("incomplete rendered output: " + "; ".join(details))

    pairs: dict[int, tuple[Path, Path, str]] = {}
    for chunk, (dng_path, dng_digest) in dngs.items():
        jpeg_path, jpeg_digest = jpegs[chunk]
        if jpeg_digest != dng_digest:
            raise RuntimeError(
                f"DNG/JPEG hash mismatch for NECTA{chunk:04d}: "
                f"{dng_digest} != {jpeg_digest}"
            )
        pairs[chunk] = (dng_path, jpeg_path, dng_digest)

    stack_path = staging_dir / f"{capture_timestamp}_stack.jpg"
    if not stack_path.is_file():
        raise RuntimeError("render produced no stack JPEG")
    return pairs


def _stage_existing_xmp_sidecars(
    output_dir: Path,
    staging_dir: Path,
    capture_timestamp: str,
    new_outputs: dict[int, tuple[Path, Path, str]],
) -> int:
    candidates: dict[int, list[tuple[int, str, Path, str]]] = {}
    for path in _capture_output_files(output_dir, capture_timestamp):
        match = _capture_output_match(path, capture_timestamp)
        if match is None or match.group("suffix").lower() != ".dng.xmp":
            continue
        chunk = int(match.group("chunk"))
        candidates.setdefault(chunk, []).append(
            (
                path.stat().st_mtime_ns,
                path.name,
                path,
                match.group("digest").lower(),
            )
        )

    migrated = 0
    for chunk, sidecars in candidates.items():
        if chunk not in new_outputs:
            continue
        _mtime, _name, source_path, old_digest = max(sidecars)
        if len(sidecars) > 1:
            print(
                f"multiple XMP sidecars for NECTA{chunk:04d}; "
                f"carrying forward newest {source_path.name}"
            )
        new_dng_path, _jpeg_path, new_digest = new_outputs[chunk]
        destination_path = staging_dir / f"{new_dng_path.name}.xmp"
        shutil.copy2(source_path, destination_path)

        contents = destination_path.read_bytes()
        old_dng_name = source_path.name[: -len(".xmp")]
        updated = contents.replace(
            old_dng_name.encode("utf-8"),
            new_dng_path.name.encode("utf-8"),
        )
        if updated == contents and old_digest != new_digest:
            updated = contents.replace(
                old_digest.encode("ascii"),
                new_digest.encode("ascii"),
            )
        if updated != contents:
            destination_path.write_bytes(updated)
        migrated += 1
    return migrated


def replace_capture_outputs(
    output_dir: Path,
    staging_dir: Path,
    capture_timestamp: str,
) -> tuple[int, int]:
    """Publish one complete render and discard the prior output generation."""
    output_dir = Path(output_dir)
    staging_dir = Path(staging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    new_outputs = _validated_staged_capture_outputs(
        staging_dir,
        capture_timestamp,
    )
    old_paths = set(_capture_output_files(output_dir, capture_timestamp))
    migrated_xmps = _stage_existing_xmp_sidecars(
        output_dir,
        staging_dir,
        capture_timestamp,
        new_outputs,
    )

    stack_name = f"{capture_timestamp}_stack.jpg"
    staged_paths = [
        path
        for path in staging_dir.iterdir()
        if path.is_file()
        and (
            path.name == stack_name
            or _capture_output_match(path, capture_timestamp) is not None
        )
    ]
    # Publish the stack last so its replacement is the final visible indication
    # that the complete output generation has changed.
    staged_paths.sort(key=lambda path: (path.name == stack_name, path.name))
    published_names = {path.name for path in staged_paths}
    for staged_path in staged_paths:
        staged_path.replace(output_dir / staged_path.name)

    stale_paths = [path for path in old_paths if path.name not in published_names]
    for stale_path in stale_paths:
        stale_path.unlink()
    return len(stale_paths), migrated_xmps


# linescans = Path("/home/dllu/pictures/linescan")
linescans = Path("/mnt/dataz/pictures/linescan")
DEFAULT_OUTPUT_DIR = linescans / "raw"


@dataclass(frozen=True)
class CircleSpeedConstraint:
    identifier: str
    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def speed_magnitude(self) -> float:
        # Annotation coordinates are binned 2x2 Bayer-cell pixels.  Horizontal
        # motion correction emits one full-resolution output column per spline
        # step, while the annotated vertical diameter becomes twice as many
        # full-resolution output rows.  A corrected circle therefore requires
        # width / |speed| == 2 * height.
        return self.width / (2.0 * self.height)

    @property
    def log_speed_sigma(self) -> float:
        """Propagate pixel-scale diameter uncertainty into the speed ratio."""
        dimension_sigma = SPEED_ANNOTATION_DIMENSION_SIGMA_PIXELS
        return math.hypot(
            dimension_sigma / self.width,
            dimension_sigma / self.height,
        )


@dataclass(frozen=True)
class EqualWidthSegment:
    identifier: str
    x1: float
    x2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def log_width_sigma(self) -> float:
        """Propagate fixed annotation-pixel uncertainty into relative width."""
        return SPEED_ANNOTATION_DIMENSION_SIGMA_PIXELS / self.width


@dataclass(frozen=True)
class EqualWidthGroup:
    identifier: str
    name: str
    segments: tuple[EqualWidthSegment, ...]


@dataclass(frozen=True)
class ManualSpeedConstraints:
    path: Path
    circles: tuple[CircleSpeedConstraint, ...]
    width_groups: tuple[EqualWidthGroup, ...]
    cut_positions: tuple[float, ...]

    @property
    def has_speed_constraints(self) -> bool:
        return bool(self.circles) or any(
            len(group.segments) >= 2 for group in self.width_groups
        )

    @property
    def has_cut_regions(self) -> bool:
        return len(self.cut_positions) >= 2

    def translated(self, x_offset: float, local_width: int) -> ManualSpeedConstraints:
        """Translate global annotation x coordinates into the selected cache."""
        circles = tuple(
            CircleSpeedConstraint(
                circle.identifier,
                circle.x - x_offset,
                circle.y,
                circle.width,
                circle.height,
            )
            for circle in self.circles
            if 0.0 <= circle.center_x - x_offset <= local_width - 1
        )
        width_groups = []
        for group in self.width_groups:
            segments = tuple(
                EqualWidthSegment(
                    segment.identifier,
                    segment.x1 - x_offset,
                    segment.x2 - x_offset,
                )
                for segment in group.segments
                if 0.0 <= segment.x1 - x_offset
                and segment.x2 - x_offset <= local_width - 1
            )
            width_groups.append(
                EqualWidthGroup(group.identifier, group.name, segments)
            )

        cuts = tuple(x - x_offset for x in self.cut_positions)
        if cuts and (cuts[0] < 0.0 or cuts[-1] > local_width):
            raise ValueError(
                "annotated cut positions fall outside the selected processing region"
            )
        return ManualSpeedConstraints(
            self.path,
            circles,
            tuple(width_groups),
            cuts,
        )


def _finite_constraint_number(value, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{description} must be finite")
    return number


def bayer_channel_frame_geometry(
    bins: list[Path],
    *,
    frame_rows: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global starts and widths in the annotation coordinate system."""
    widths = []
    bytes_per_sample = np.dtype(np.uint16).itemsize
    bytes_per_raw_column = frame_rows * bytes_per_sample
    for bin_path in bins:
        size = bin_path.stat().st_size
        if size % bytes_per_raw_column:
            raise ValueError(
                f"{bin_path} size is not a multiple of one {frame_rows}-sample column"
            )
        raw_columns = size // bytes_per_raw_column
        if raw_columns % 2:
            raise ValueError(f"{bin_path} has an odd number of Bayer columns")
        widths.append(raw_columns // 2)
    widths_array = np.asarray(widths, dtype=np.int64)
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(widths_array, dtype=np.int64))
    )
    return starts, widths_array


def load_manual_speed_constraints(
    capture_dir: Path,
    bins: list[Path],
) -> ManualSpeedConstraints | None:
    """Load and validate the GUI's speed-constraint document, when present."""
    path = Path(capture_dir) / SPEED_CONSTRAINTS_JSON
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported speed-constraint schema {payload.get('schema_version')!r}"
        )
    coordinate_system = payload.get("coordinate_system")
    if not isinstance(coordinate_system, dict) or (
        coordinate_system.get("name") != "bayer_channel_pixels"
        or coordinate_system.get("bayer_bin") != 2
    ):
        raise ValueError(
            f"{path} does not use the expected 2x-binned Bayer-channel coordinates"
        )

    circles = []
    identifiers = set()
    for index, entry in enumerate(payload.get("circles", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("bounds"), dict):
            raise TypeError(f"circles[{index}] must contain a bounds object")
        identifier = str(entry.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError(f"circles[{index}] has a missing or duplicate id")
        identifiers.add(identifier)
        bounds = entry["bounds"]
        x = _finite_constraint_number(bounds.get("x"), f"circles[{index}].x")
        y = _finite_constraint_number(bounds.get("y"), f"circles[{index}].y")
        width = _finite_constraint_number(
            bounds.get("width"), f"circles[{index}].width"
        )
        height = _finite_constraint_number(
            bounds.get("height"), f"circles[{index}].height"
        )
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"circles[{index}] width and height must be positive")
        circles.append(CircleSpeedConstraint(identifier, x, y, width, height))

    width_groups = []
    group_ids = set()
    segment_ids = set()
    for group_index, entry in enumerate(payload.get("equal_width_groups", [])):
        if not isinstance(entry, dict):
            raise TypeError(f"equal_width_groups[{group_index}] must be an object")
        identifier = str(entry.get("id", ""))
        if not identifier or identifier in group_ids:
            raise ValueError(
                f"equal_width_groups[{group_index}] has a missing or duplicate id"
            )
        group_ids.add(identifier)
        name = str(entry.get("name", "")).strip() or f"Width group {group_index + 1}"
        segments = []
        for segment_index, segment in enumerate(entry.get("segments", [])):
            description = (
                f"equal_width_groups[{group_index}].segments[{segment_index}]"
            )
            if not isinstance(segment, dict):
                raise TypeError(f"{description} must be an object")
            segment_id = str(segment.get("id", ""))
            if not segment_id or segment_id in segment_ids:
                raise ValueError(f"{description} has a missing or duplicate id")
            segment_ids.add(segment_id)
            x1 = _finite_constraint_number(segment.get("x1"), f"{description}.x1")
            x2 = _finite_constraint_number(segment.get("x2"), f"{description}.x2")
            if x1 == x2:
                raise ValueError(f"{description} must have non-zero width")
            segments.append(
                EqualWidthSegment(segment_id, min(x1, x2), max(x1, x2))
            )
        width_groups.append(EqualWidthGroup(identifier, name, tuple(segments)))

    cut_positions = []
    cut_ids = set()
    for index, entry in enumerate(payload.get("cut_positions", [])):
        if not isinstance(entry, dict):
            raise TypeError(f"cut_positions[{index}] must be an object")
        identifier = str(entry.get("id", ""))
        if not identifier or identifier in cut_ids:
            raise ValueError(f"cut_positions[{index}] has a missing or duplicate id")
        cut_ids.add(identifier)
        cut_positions.append(
            _finite_constraint_number(entry.get("x"), f"cut_positions[{index}].x")
        )
    cut_positions.sort()
    if len(cut_positions) >= 2 and np.any(np.diff(cut_positions) <= 0.0):
        raise ValueError("annotated cut positions must be strictly increasing")

    frame_starts, frame_widths = bayer_channel_frame_geometry(bins)
    capture_width = int(frame_starts[-1])
    source = payload.get("source")
    if isinstance(source, dict) and isinstance(source.get("frames"), list):
        source_frames = source["frames"]
        expected = [
            (
                bin_path.name,
                int(width * 2),
                int(start),
                int(start + width),
            )
            for bin_path, start, width in zip(bins, frame_starts, frame_widths)
        ]
        recorded = [
            (
                str(entry.get("file", "")),
                entry.get("raw_columns"),
                entry.get("x_start"),
                entry.get("x_end"),
            )
            for entry in source_frames
            if isinstance(entry, dict)
        ]
        if source.get("frame_rows") != 4096 or recorded != expected:
            raise ValueError(
                f"{path} was annotated against a different set of raw frames"
            )

    for circle in circles:
        if not 0.0 <= circle.center_x < capture_width:
            raise ValueError(f"circle {circle.identifier} lies outside the capture")
        if not 0.0 <= circle.y or circle.y + circle.height > 2048:
            raise ValueError(f"circle {circle.identifier} lies outside the sensor rows")
    for group in width_groups:
        for segment in group.segments:
            if segment.x1 < 0.0 or segment.x2 > capture_width:
                raise ValueError(
                    f"width segment {segment.identifier} lies outside the capture"
                )
    if cut_positions and (
        cut_positions[0] < 0.0 or cut_positions[-1] > capture_width
    ):
        raise ValueError("annotated cut positions lie outside the capture")

    return ManualSpeedConstraints(
        path,
        tuple(circles),
        tuple(width_groups),
        tuple(cut_positions),
    )


def select_frames_for_cut_region(
    frame_starts: np.ndarray,
    frame_widths: np.ndarray,
    cut_positions: tuple[float, ...],
    *,
    padding_columns: int,
) -> list[int]:
    """Select frames intersecting the manual ROI plus processing-only padding."""
    if len(cut_positions) < 2:
        raise ValueError("at least a start and end cut position are required")
    if padding_columns < 0:
        raise ValueError("padding_columns must be non-negative")
    padded_start, padded_end = aligned_processing_column_bounds(
        int(frame_starts[-1]),
        cut_positions[0] - padding_columns,
        cut_positions[-1] + padding_columns,
    )
    frame_ends = frame_starts[:-1] + frame_widths
    intersecting = np.flatnonzero(
        (frame_ends > padded_start) & (frame_starts[:-1] < padded_end)
    )
    if not len(intersecting):
        raise ValueError("annotated cut region does not intersect any raw frames")
    return list(range(int(intersecting[0]), int(intersecting[-1]) + 1))


def aligned_processing_column_bounds(
    capture_width: int,
    start: float,
    stop: float,
    *,
    alignment: int = STRUCTURE_MASK_DOWNSAMPLE,
) -> tuple[int, int]:
    """Clamp and outward-align an analysis interval independently of raw files."""
    if capture_width < alignment or alignment < 1:
        raise ValueError("capture width and alignment must be positive")
    aligned_capture_width = capture_width - capture_width % alignment
    aligned_start = max(0, math.floor(start / alignment) * alignment)
    aligned_stop = min(
        aligned_capture_width,
        math.ceil(stop / alignment) * alignment,
    )
    if aligned_stop <= aligned_start:
        raise ValueError("processing column interval is empty")
    return aligned_start, aligned_stop


def format_contiguous_segments(indices):
    if not indices:
        return "none"

    segments = []
    start = indices[0]
    prev = start
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev))
        start = idx
        prev = idx
    segments.append((start, prev))

    return ", ".join(
        str(start) if start == end else f"{start}-{end}" for start, end in segments
    )


class FileDataCache:
    def __init__(
        self,
        bin_paths,
        dtype,
        rows,
        max_cache_bytes: int = 512 * 1024 * 1024,
    ):
        self.bin_paths = bin_paths
        self.dtype = dtype
        self.rows = rows
        self.max_cache_bytes = max(1, max_cache_bytes)
        self.cached_bytes = 0
        # determine columns per file
        self.file_cols = []
        for p in bin_paths:
            size = p.stat().st_size
            n_elems = size // np.dtype(dtype).itemsize
            if n_elems % rows != 0:
                raise ValueError(f"File {p} size not multiple of rows")
            cols = n_elems // rows
            self.file_cols.append(cols)
        # cumulative column offsets
        self.cum_cols = np.concatenate(([0], np.cumsum(self.file_cols)))
        self.total_cols = int(self.cum_cols[-1])
        self.height = rows
        self.cache = OrderedDict()

    @property
    def shape(self):
        return (self.height, self.total_cols)

    def _load_file(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
        else:
            data2d = load_raw_frame(
                self.bin_paths[idx],
                rows=self.height,
                dtype=self.dtype,
                output_dtype=np.float32,
            )
            self.cache[idx] = data2d
            self.cached_bytes += data2d.nbytes
            while self.cached_bytes > self.max_cache_bytes and len(self.cache) > 1:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= evicted.nbytes
        return self.cache[idx]

    def __getitem__(self, key):
        # Expect key as (rows, cols)
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Invalid index for FileDataCache")
        row_idx, col_idx = key
        # handle integer column indexing
        if isinstance(col_idx, int):
            ci = col_idx if col_idx >= 0 else self.total_cols + col_idx
            # find file containing this column
            file_idx = np.searchsorted(self.cum_cols, ci, side="right") - 1
            local = ci - self.cum_cols[file_idx]
            arr_file = self._load_file(file_idx)
            return arr_file[row_idx, local]
        # handle slice of columns
        if isinstance(col_idx, slice):
            start = (
                0
                if col_idx.start is None
                else (
                    col_idx.start
                    if col_idx.start >= 0
                    else self.total_cols + col_idx.start
                )
            )
            stop = (
                self.total_cols
                if col_idx.stop is None
                else (
                    col_idx.stop
                    if col_idx.stop >= 0
                    else self.total_cols + col_idx.stop
                )
            )
            step = 1 if col_idx.step is None else col_idx.step
            # collect blocks from each file
            blocks = []
            for i, cols_i in enumerate(self.file_cols):
                fstart = self.cum_cols[i]
                fend = self.cum_cols[i + 1]
                if fend <= start or fstart >= stop:
                    continue
                lo = max(start, fstart) - fstart
                hi = min(stop, fend) - fstart
                arr_file = self._load_file(i)
                blocks.append(arr_file[row_idx, lo:hi])
            if not blocks:
                empty = np.empty((self.height, 0), dtype=np.float32)
                return empty[row_idx, :]
            arr = np.concatenate(blocks, axis=-1)
            # apply column step
            if step != 1:
                arr = arr[..., ::step]
            return arr
        # unsupported indexing
        raise IndexError("Unsupported index type for FileDataCache")


class ChannelView:
    """
    Lazy view of a single Bayer channel from FileDataCache.
    """

    def __init__(self, cache, row_offset, col_offset, row_step=2, col_step=2):
        self.cache = cache
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.row_step = row_step
        self.col_step = col_step
        # compute channel dimensions
        total_cols = cache.total_cols
        self.height = (cache.height - row_offset + row_step - 1) // row_step
        self.width = (total_cols - col_offset + col_step - 1) // col_step

    @property
    def shape(self):
        """Shape of this channel view as (rows, cols)."""
        return (self.height, self.width)

    def __getitem__(self, key):
        # key: (row_idx, col_idx)
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("ChannelView requires 2D indexing")
        row_key, col_key = key
        # map row_key to parent
        if isinstance(row_key, slice):
            start = 0 if row_key.start is None else row_key.start
            stop = self.height if row_key.stop is None else row_key.stop
            step = 1 if row_key.step is None else row_key.step
            abs_row = slice(
                self.row_offset + start * self.row_step,
                self.row_offset + stop * self.row_step,
                self.row_step * step,
            )
        else:
            # single index
            idx = row_key if row_key >= 0 else self.height + row_key
            abs_row = self.row_offset + idx * self.row_step
        # map col_key similarly
        if isinstance(col_key, slice):
            start = 0 if col_key.start is None else col_key.start
            stop = self.width if col_key.stop is None else col_key.stop
            step = 1 if col_key.step is None else col_key.step
            abs_col = slice(
                self.col_offset + start * self.col_step,
                self.col_offset + stop * self.col_step,
                self.col_step * step,
            )
        else:
            idx = col_key if col_key >= 0 else self.width + col_key
            abs_col = self.col_offset + idx * self.col_step
        # fetch from parent cache
        return self.cache[abs_row, abs_col]


class ColumnSliceView:
    """A zero-copy coordinate view over an exact contiguous column interval."""

    def __init__(self, source, start: int, stop: int):
        if not 0 <= start < stop <= source.width:
            raise ValueError("column slice lies outside its source")
        self.source = source
        self.start = int(start)
        self.stop = int(stop)
        self.height = source.height
        self.width = self.stop - self.start

    @property
    def shape(self):
        return (self.height, self.width)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("ColumnSliceView requires 2D indexing")
        row_key, col_key = key
        if isinstance(col_key, slice):
            start, stop, step = col_key.indices(self.width)
            if step < 1:
                raise IndexError("negative column strides are not supported")
            source_columns = slice(
                self.start + start,
                self.start + stop,
                step,
            )
        else:
            index = col_key if col_key >= 0 else self.width + col_key
            if not 0 <= index < self.width:
                raise IndexError("column index is out of range")
            source_columns = self.start + index
        return self.source[row_key, source_columns]


def windowed_cross_correlation(
    green_1: np.ndarray,
    green_2: np.ndarray,
    window_step: int = 256,
    window_size: int = 4096,
    corr_size: int = 15,
    output_dir: Path | None = None,
    jitter_models: ColumnJitterModel | None = None,
    structure_mask: np.ndarray | None = None,
    structure_mask_scale: int = 1,
    valid_x_range: tuple[float, float] | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate green-plane displacement only where scene evidence exists.

    A fixed line-scan background is constant along capture x.  Its displacement
    is therefore undefined, rather than zero or whichever random lag happens to
    win.  Each result must contain at least one full-height-equivalent vertical
    edge and an absolute, distinct normalized-correlation peak. When a scene
    mask is available, normalized matches from its vertical bands are combined
    robustly so one energetic patch cannot overrule motion seen across the
    vehicle.
    """
    if green_1.shape != green_2.shape:
        raise ValueError("green channels must have the same shape")
    if window_step < 1 or window_size < window_step:
        raise ValueError("correlation window dimensions must be positive")
    if window_size % window_step:
        raise ValueError("correlation window size must be divisible by its step")
    if corr_size < 3 or corr_size % 2 == 0:
        raise ValueError("correlation size must be an odd integer of at least 3")
    if structure_mask_scale < 1:
        raise ValueError("structure mask scale must be positive")
    if structure_mask is not None:
        structure_mask = np.asarray(structure_mask, dtype=bool)
        if structure_mask.ndim != 2:
            raise ValueError("structure mask must be two-dimensional")
    if valid_x_range is not None:
        valid_x_min, valid_x_max = map(float, valid_x_range)
        if (
            not math.isfinite(valid_x_min)
            or not math.isfinite(valid_x_max)
            or valid_x_max <= valid_x_min
        ):
            raise ValueError("valid correlation x range must be finite and non-empty")
    else:
        valid_x_min = -math.inf
        valid_x_max = math.inf

    height, width = green_1.shape
    corr_half = corr_size // 2
    # Keep a one-column derivative margin outside the complete lag range.
    margin = corr_half + 1
    first_block_start = int(math.ceil(margin / window_step)) * window_step
    block_starts = np.arange(
        first_block_start,
        width - window_step - margin + 1,
        window_step,
        dtype=np.int64,
    )
    blocks_per_window = window_size // window_step
    if block_starts.size < blocks_per_window:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    l1_profiles = np.empty((block_starts.size, corr_size), dtype=np.float64)
    ncc_numerators = np.empty_like(l1_profiles)
    ncc_green2_energy = np.empty_like(l1_profiles)
    green1_energy = np.empty(block_starts.size, dtype=np.float64)
    internal_structure = np.empty(block_starts.size, dtype=np.float64)
    # After the two greens are aligned below, averaging small vertical bands
    # suppresses sensor noise and lets fractional registration use distributed
    # scene evidence instead of one particular edge. At production height this
    # is 128 x width (about 82 MiB per green for a 160k-column capture).
    compact_band_count = min(128, height)
    compact_rows = (height // compact_band_count) * compact_band_count
    compact_file = tempfile.TemporaryFile()
    compact_features = np.memmap(
        compact_file,
        dtype=np.float32,
        mode="w+",
        shape=(2, compact_band_count, width),
    )
    compact_features[:] = 0.0
    compact_green1 = compact_features[0]
    compact_green2 = compact_features[1]
    compact_structure = None
    if structure_mask is not None and structure_mask.size:
        # Preserve the mask's inexpensive horizontal downsampling, but put its
        # rows on the same vertical bands as the compact registration features.
        # Area resampling gives each band the fraction covered by real scene
        # edges; it does not spread those edges into neighboring x positions.
        vertical_interpolation = (
            cv2.INTER_AREA
            if structure_mask.shape[0] >= compact_band_count
            else cv2.INTER_NEAREST
        )
        compact_structure = cv2.resize(
            structure_mask.astype(np.float32),
            (structure_mask.shape[1], compact_band_count),
            interpolation=vertical_interpolation,
        )

    def compact_vertical(block: np.ndarray) -> np.ndarray:
        compact = block[:compact_rows].reshape(
            compact_band_count,
            compact_rows // compact_band_count,
            block.shape[1],
        ).mean(axis=1)
        # A fixed line-scan background and residual line-wide exposure changes
        # are vertically broad. They carry no vehicle-motion information but
        # otherwise add a strong zero-lag component to fractional registration.
        compact -= gaussian_filter1d(
            compact,
            sigma=SPEED_CORRELATION_VERTICAL_HIGHPASS_SIGMA,
            axis=0,
            mode="nearest",
        )
        return compact

    def align_green_rows(block: np.ndarray, channel: int) -> np.ndarray:
        """Put both Bayer greens on the plane halfway between their rows."""
        # G1 is sampled at raw rows 1, 3, ... and G2 at rows 0, 2, .... Row
        # averaging alone does not remove that one-raw-row parallax: a slanted
        # edge then masquerades as horizontal motion. Interpolating both
        # channels by half a raw row gives them matched vertical blur as well
        # as a common sampling plane.
        aligned = np.asarray(block, dtype=np.float32).copy()
        if len(aligned) <= 1:
            return aligned
        if channel == JITTER_GREEN1:
            aligned[1:] = 0.75 * block[1:] + 0.25 * block[:-1]
        elif channel == JITTER_GREEN2:
            aligned[:-1] = 0.75 * block[:-1] + 0.25 * block[1:]
        else:
            raise ValueError("green row alignment requires a green channel")
        return aligned

    def vertical_energy(block: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(block)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 0.1 * np.max(block)
        return grad_x / magnitude

    for block_index, x in progress(
        enumerate(block_starts),
        desc="train speed estimation",
        total=len(block_starts),
        show=show_progress,
    ):
        green1_start = int(x) - 1
        window_1 = green_1[:, green1_start : int(x) + window_step + 1]
        if jitter_models is not None:
            window_1 = apply_column_models(
                window_1,
                jitter_models,
                green1_start,
                JITTER_GREEN1,
                protect_scene=False,
            )
        window_1 = align_green_rows(window_1, JITTER_GREEN1)
        compact_green1[:, int(x) : int(x) + window_step] = compact_vertical(
            window_1[:, 1 : window_step + 1]
        )
        window_1 = vertical_energy(window_1)[:, 1 : window_step + 1]

        green2_start = int(x) - margin
        window_2_base = green_2[
            :, green2_start : int(x) + window_step + margin
        ]
        if jitter_models is not None:
            window_2_base = apply_column_models(
                window_2_base,
                jitter_models,
                green2_start,
                JITTER_GREEN2,
                protect_scene=False,
            )
        window_2_base = align_green_rows(window_2_base, JITTER_GREEN2)
        compact_green2[
            :, green2_start : int(x) + window_step + margin
        ] = compact_vertical(window_2_base)
        window_2_base = vertical_energy(window_2_base)

        green1_energy[block_index] = np.sum(np.square(window_1))
        central_green2 = window_2_base[
            :, margin : margin + window_step
        ]
        internal_structure[block_index] = 0.5 * (
            np.count_nonzero(np.abs(window_1) > STRUCTURE_MASK_THRESHOLD)
            + np.count_nonzero(
                np.abs(central_green2) > STRUCTURE_MASK_THRESHOLD
            )
        )
        for corr_ind in range(corr_size):
            corr_x = corr_ind - corr_half
            start = corr_x + margin
            window_2 = window_2_base[:, start : start + window_step]
            l1_profiles[block_index, corr_ind] = -np.sum(
                np.abs(window_1 - window_2)
            )
            ncc_numerators[block_index, corr_ind] = np.sum(
                window_1 * window_2
            )
            ncc_green2_energy[block_index, corr_ind] = np.sum(
                np.square(window_2)
            )

    # A Gaussian derivative is a band-limited, continuously interpolable
    # horizontal feature. Directly registering it removes the large
    # texture-dependent bias of fitting a bell curve through integer L1 lags.
    gaussian_filter1d(
        compact_green1,
        sigma=1.5,
        order=1,
        axis=1,
        mode="nearest",
        output=compact_green1,
    )
    gaussian_filter1d(
        compact_green2,
        sigma=1.5,
        order=1,
        axis=1,
        mode="nearest",
        output=compact_green2,
    )

    good_xs = []
    raw_ys = []
    refined_ys = []
    absolute_weights = []
    diagnostic_profiles = []
    band_consensus_count = 0
    for end_index in range(blocks_per_window - 1, block_starts.size):
        start_index = end_index - blocks_per_window + 1
        window_start = int(block_starts[start_index])
        window_stop = int(block_starts[end_index]) + window_step
        if window_start < valid_x_min or window_stop > valid_x_max:
            continue

        if structure_mask is None:
            edge_equivalents = float(
                np.sum(internal_structure[start_index : end_index + 1]) / height
            )
        else:
            mask_start = max(0, window_start // structure_mask_scale)
            mask_stop = min(
                structure_mask.shape[1],
                int(math.ceil(window_stop / structure_mask_scale)),
            )
            if mask_stop <= mask_start or structure_mask.shape[0] == 0:
                edge_equivalents = 0.0
            else:
                edge_equivalents = float(
                    np.count_nonzero(structure_mask[:, mask_start:mask_stop])
                    / structure_mask.shape[0]
                )
        # One full-height edge (or the equivalent amount of distributed scene
        # structure) is the minimum meaningful motion evidence in 4096 columns.
        if edge_equivalents < 1.0:
            continue

        block_slice = slice(start_index, end_index + 1)
        l1_profile = np.sum(l1_profiles[block_slice], axis=0)
        ncc_numerator = np.sum(ncc_numerators[block_slice], axis=0)
        ncc_denominator = np.sqrt(
            np.sum(green1_energy[block_slice])
            * np.sum(ncc_green2_energy[block_slice], axis=0)
        )
        ncc_profile = np.divide(
            ncc_numerator,
            ncc_denominator,
            out=np.zeros_like(ncc_numerator),
            where=ncc_denominator > 0.0,
        )
        feature_1 = compact_green1[:, window_start:window_stop].copy()
        feature_2_base = compact_green2[
            :, window_start - margin : window_stop + margin
        ]
        diagnostic_profile = ncc_profile
        legacy_registration = None
        used_band_consensus = compact_structure is not None

        def interpolated_feature_2(displacement: float) -> np.ndarray | None:
            integer = math.floor(displacement)
            fraction = displacement - integer
            index = margin + integer
            p0 = feature_2_base[:, index - 1 : index - 1 + window_size]
            p1 = feature_2_base[:, index : index + window_size]
            p2 = feature_2_base[:, index + 1 : index + 1 + window_size]
            p3 = feature_2_base[:, index + 2 : index + 2 + window_size]
            if min(p0.shape[1], p1.shape[1], p2.shape[1], p3.shape[1]) < window_size:
                return None
            # Catmull-Rom interpolation preserves the sampled values at each
            # integer lag while providing a smooth one-dimensional objective.
            return 0.5 * (
                2.0 * p1
                + (-p0 + p2) * fraction
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * fraction**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * fraction**3
            )

        if compact_structure is None:
            l1_peak_idx = int(np.argmax(l1_profile))
            ncc_peak_idx = int(np.argmax(ncc_profile))
            if l1_peak_idx <= 0 or l1_peak_idx >= corr_size - 1:
                continue
            if abs(l1_peak_idx - ncc_peak_idx) > 1:
                continue

            peak_idx = l1_peak_idx
            coarse_displacement = peak_idx - corr_half
            peak_correlation = float(ncc_profile[peak_idx])
            peak_contrast = peak_correlation - float(np.median(ncc_profile))
            # These are absolute correlation-quality tests, not capture-relative
            # percentiles. Random background noise is not promoted merely because
            # it is the least-bad window in an otherwise empty capture.
            if peak_correlation < 0.02 or peak_contrast < 0.01:
                continue

            feature_1 -= np.mean(feature_1, axis=1, keepdims=True)
            feature_1_energy = float(np.sum(np.square(feature_1)))
            if feature_1_energy <= 0.0:
                continue

            def fractional_registration_objective(displacement: float) -> float:
                feature_2 = interpolated_feature_2(displacement)
                if feature_2 is None:
                    return 1.0
                feature_2 -= np.mean(feature_2, axis=1, keepdims=True)
                denominator = math.sqrt(
                    feature_1_energy * float(np.sum(np.square(feature_2)))
                )
                if denominator <= 0.0:
                    return 1.0
                return -float(np.sum(feature_1 * feature_2)) / denominator

        else:
            mask_columns = (
                np.arange(window_start, window_stop, dtype=np.int64)
                // structure_mask_scale
            )
            if mask_columns[-1] >= compact_structure.shape[1]:
                continue
            band_weights = compact_structure[:, mask_columns]
            band_support = np.sum(band_weights, axis=1, dtype=np.float64)
            min_band_support = max(4.0, window_size / 128.0)
            supported_bands = band_support >= min_band_support
            if np.count_nonzero(supported_bands) < SPEED_CORRELATION_MIN_VALID_BANDS:
                continue

            safe_support = np.maximum(band_support, 1.0)
            feature_1_mean = np.sum(
                band_weights * feature_1, axis=1, dtype=np.float64
            ) / safe_support
            feature_1 -= feature_1_mean[:, None]
            weighted_feature_1 = band_weights * feature_1
            feature_1_band_energy = np.sum(
                weighted_feature_1 * feature_1, axis=1, dtype=np.float64
            )
            supported_bands &= feature_1_band_energy > 0.0

            def band_correlations(feature_2: np.ndarray) -> np.ndarray:
                feature_2 = np.asarray(feature_2, dtype=np.float32).copy()
                feature_2_mean = np.sum(
                    band_weights * feature_2, axis=1, dtype=np.float64
                ) / safe_support
                feature_2 -= feature_2_mean[:, None]
                feature_2_band_energy = np.sum(
                    band_weights * np.square(feature_2),
                    axis=1,
                    dtype=np.float64,
                )
                denominator = np.sqrt(
                    feature_1_band_energy * feature_2_band_energy
                )
                numerator = np.sum(
                    weighted_feature_1 * feature_2,
                    axis=1,
                    dtype=np.float64,
                )
                return np.divide(
                    numerator,
                    denominator,
                    out=np.full(compact_band_count, np.nan, dtype=np.float64),
                    where=denominator > 0.0,
                )

            candidate_displacements = np.arange(
                -corr_half + 1,
                corr_half,
                dtype=np.int64,
            )
            band_profiles = np.column_stack(
                [
                    band_correlations(
                        feature_2_base[
                            :,
                            margin + displacement : margin
                            + displacement
                            + window_size,
                        ]
                    )
                    for displacement in candidate_displacements
                ]
            )
            quality_profiles = np.nan_to_num(band_profiles, nan=-1.0)
            band_peak = np.max(quality_profiles, axis=1)
            band_contrast = band_peak - np.median(quality_profiles, axis=1)
            valid_bands = (
                supported_bands
                & (band_peak >= SPEED_CORRELATION_BAND_MIN_PEAK)
                & (band_contrast >= SPEED_CORRELATION_BAND_MIN_CONTRAST)
            )
            if np.count_nonzero(valid_bands) < SPEED_CORRELATION_MIN_VALID_BANDS:
                continue

            def trimmed_band_mean(correlations: np.ndarray) -> float:
                values = correlations[valid_bands]
                values = values[np.isfinite(values)]
                if values.size < SPEED_CORRELATION_MIN_VALID_BANDS:
                    return -1.0
                values.sort()
                trim = int(values.size * SPEED_CORRELATION_BAND_TRIM_FRACTION)
                if trim:
                    values = values[trim:-trim]
                return float(np.mean(values))

            consensus_profile = np.asarray(
                [
                    trimmed_band_mean(band_profiles[:, index].copy())
                    for index in range(band_profiles.shape[1])
                ],
                dtype=np.float64,
            )
            consensus_peak_idx = int(np.argmax(consensus_profile))
            if consensus_peak_idx in (0, len(consensus_profile) - 1):
                continue
            coarse_displacement = int(
                candidate_displacements[consensus_peak_idx]
            )
            peak_correlation = float(consensus_profile[consensus_peak_idx])
            peak_contrast = peak_correlation - float(
                np.median(consensus_profile)
            )
            if peak_correlation < 0.02 or peak_contrast < 0.01:
                continue

            diagnostic_profile = np.full(corr_size, np.nan, dtype=np.float64)
            diagnostic_profile[1:-1] = consensus_profile

            def fractional_registration_objective(displacement: float) -> float:
                feature_2 = interpolated_feature_2(displacement)
                if feature_2 is None:
                    return 1.0
                return -trimmed_band_mean(band_correlations(feature_2))

            # Retain the established whole-window estimate whenever its
            # subpixel optimum is genuinely inside the interval around the
            # integer peak. The band consensus is specifically the escape
            # route for rejected samples, optima pinned to that artificial
            # +/-0.75 search boundary, and material disagreement with the
            # structure-masked consensus. It should not perturb already sound
            # measurements merely because it is available.
            legacy_l1_peak_idx = int(np.argmax(l1_profile))
            legacy_ncc_peak_idx = int(np.argmax(ncc_profile))
            if (
                0 < legacy_l1_peak_idx < corr_size - 1
                and abs(legacy_l1_peak_idx - legacy_ncc_peak_idx) <= 1
            ):
                legacy_coarse_displacement = legacy_l1_peak_idx - corr_half
                legacy_peak_correlation = float(
                    ncc_profile[legacy_l1_peak_idx]
                )
                legacy_peak_contrast = legacy_peak_correlation - float(
                    np.median(ncc_profile)
                )
                global_feature_1 = feature_1 - np.mean(
                    feature_1,
                    axis=1,
                    keepdims=True,
                )
                global_feature_1_energy = float(
                    np.sum(np.square(global_feature_1))
                )

                def legacy_fractional_objective(displacement: float) -> float:
                    feature_2 = interpolated_feature_2(displacement)
                    if feature_2 is None:
                        return 1.0
                    feature_2 -= np.mean(feature_2, axis=1, keepdims=True)
                    denominator = math.sqrt(
                        global_feature_1_energy
                        * float(np.sum(np.square(feature_2)))
                    )
                    if denominator <= 0.0:
                        return 1.0
                    return (
                        -float(np.sum(global_feature_1 * feature_2))
                        / denominator
                    )

                if (
                    legacy_peak_correlation >= 0.02
                    and legacy_peak_contrast >= 0.01
                    and global_feature_1_energy > 0.0
                ):
                    legacy_lower = max(
                        -corr_half + 1.0e-4,
                        legacy_coarse_displacement - 0.75,
                    )
                    legacy_upper = min(
                        corr_half - 1.0e-4,
                        legacy_coarse_displacement + 0.75,
                    )
                    candidate = minimize_scalar(
                        legacy_fractional_objective,
                        bounds=(legacy_lower, legacy_upper),
                        method="bounded",
                        options={"xatol": 1.0e-4, "maxiter": 32},
                    )
                    if candidate.success and math.isfinite(candidate.x):
                        legacy_registration = candidate

        lower = max(-corr_half + 1.0e-4, coarse_displacement - 0.75)
        upper = min(corr_half - 1.0e-4, coarse_displacement + 0.75)
        registration = minimize_scalar(
            fractional_registration_objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1.0e-4, "maxiter": 32},
        )
        if not registration.success or not math.isfinite(registration.x):
            continue
        if legacy_registration is not None:
            boundary_clearance = min(
                float(legacy_registration.x) - legacy_lower,
                legacy_upper - float(legacy_registration.x),
            )
            estimator_disagreement = abs(
                float(legacy_registration.x) - float(registration.x)
            )
            if (
                boundary_clearance >= 0.01
                and estimator_disagreement < SPEED_CORRELATION_ESTIMATOR_AGREEMENT
            ):
                registration = legacy_registration
                coarse_displacement = legacy_coarse_displacement
                peak_correlation = legacy_peak_correlation
                peak_contrast = legacy_peak_contrast
                diagnostic_profile = ncc_profile
                used_band_consensus = False
        refined_peak = float(registration.x)
        fractional_correlation = -float(registration.fun)
        if not math.isfinite(fractional_correlation) or fractional_correlation <= 0.0:
            continue
        center_x = 0.5 * (window_start + window_stop)
        good_xs.append(center_x)
        raw_ys.append(coarse_displacement)
        refined_ys.append(refined_peak)
        absolute_weights.append(
            math.sqrt(edge_equivalents)
            * peak_correlation
            * peak_contrast
            * fractional_correlation
        )
        diagnostic_profiles.append(diagnostic_profile.copy())
        band_consensus_count += int(used_band_consensus)

    good_xs = np.asarray(good_xs, dtype=np.float64)
    raw_ys = np.asarray(raw_ys, dtype=np.float64)
    refined_ys = np.asarray(refined_ys, dtype=np.float64)
    weights = np.asarray(absolute_weights, dtype=np.float64)
    if weights.size:
        weights /= np.max(weights)
    if show_progress and compact_structure is not None and good_xs.size:
        print(
            "speed correlation: used vertical-band consensus for "
            f"{band_consensus_count}/{good_xs.size} windows"
        )

    compact_features.flush()
    compact_mapping = compact_features._mmap
    del compact_green1, compact_green2, compact_features
    compact_mapping.close()
    compact_file.close()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if diagnostic_profiles:
            plt.imshow(np.asarray(diagnostic_profiles), aspect="auto")
            plt.savefig(output_dir / "heatmap.png")
            plt.close()
        plt.figure(figsize=(16, 9), dpi=300)
        plt.plot(good_xs, raw_ys, "b.", label="Integer peaks")
        plt.plot(good_xs, refined_ys, "r.", label="Subpixel peaks")
        plt.axhline(0.0, color="k", linestyle="--")
        plt.legend()
        plt.savefig(output_dir / "subpixel_peaks.png")
        plt.close()

    return good_xs, refined_ys, weights


def _smooth_speed_extrapolator(spline, support_min: float, support_max: float):
    """Extend a one-direction speed spline without flat endpoint clamps.

    Outside the correlation support, continue the endpoint value and first
    derivative in log-magnitude space.  The resulting curve stays in the
    detected motion direction and is C1 at both support boundaries.
    """
    support_min = float(support_min)
    support_max = float(support_max)
    if not math.isfinite(support_min) or not math.isfinite(support_max):
        raise ValueError("speed-spline support must be finite")
    if support_max <= support_min:
        raise ValueError("speed-spline support must be non-empty")

    derivative = spline.derivative()
    left_value = float(spline(support_min))
    right_value = float(spline(support_max))
    left_derivative = float(derivative(support_min))
    right_derivative = float(derivative(support_max))
    endpoint_values = np.asarray((left_value, right_value), dtype=np.float64)
    endpoint_derivatives = np.asarray(
        (left_derivative, right_derivative), dtype=np.float64
    )
    if not np.all(np.isfinite(endpoint_values)) or not np.all(
        np.isfinite(endpoint_derivatives)
    ):
        raise ValueError("speed spline has invalid endpoint values")
    if left_value == 0.0 or right_value == 0.0 or left_value * right_value < 0.0:
        raise ValueError("speed spline endpoints must have one non-zero direction")

    left_log_slope = left_derivative / left_value
    right_log_slope = right_derivative / right_value

    def extrapolated(values):
        values_array = np.asarray(values, dtype=np.float64)
        flat_values = values_array.reshape(-1)
        result = np.empty_like(flat_values)
        left = flat_values < support_min
        right = flat_values > support_max
        inside = ~(left | right)
        result[inside] = spline(flat_values[inside])
        result[left] = left_value * np.exp(
            left_log_slope * (flat_values[left] - support_min)
        )
        result[right] = right_value * np.exp(
            right_log_slope * (flat_values[right] - support_max)
        )
        result = result.reshape(values_array.shape)
        if values_array.ndim == 0:
            return float(result)
        return result

    extrapolated.support = (support_min, support_max)
    return extrapolated


def robust_bspline_fit(
    x,
    y,
    weights,
    smoothness,
    g: Path,
    c=2.5,
    visualize: bool = False,
    verbose: bool = True,
):
    """Fit a stiff, direction-preserving log-speed spline.

    Parameters:
      x          : 1D numpy array of independent variable values.
      y          : 1D numpy array of dependent values.
      weights    : 1D numpy array of initial weights (same shape as x and y).
      smoothness : Retained for command/API compatibility; the former
                   scale-dependent smoothing factor is deliberately unused.
      c          : Huber transition in robust standard deviations.

    Returns:
      spline     : A callable representing the robustly fitted cubic spline.

    Also plots the data and the final spline fit.
    """
    del smoothness
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or weights.shape != x.shape:
        raise ValueError("x, y, and weights must be one-dimensional and equal length")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
    finite &= (weights > 0.0) & (y != 0.0)
    x = x[finite]
    y = y[finite]
    weights = weights[finite]
    if len(x) < 4:
        raise ValueError("At least four motion samples are required for a cubic spline")

    def weighted_median(values, sample_weights):
        order = np.argsort(values)
        cumulative = np.cumsum(sample_weights[order])
        midpoint = 0.5 * cumulative[-1]
        return values[order[np.searchsorted(cumulative, midpoint)]]

    initial_weights = np.copy(weights)
    dominant_motion = weighted_median(y, weights)
    direction_mask = y * dominant_motion > 0.0
    if np.count_nonzero(direction_mask) < 4:
        raise ValueError("Fewer than four samples agree on the motion direction")
    fit_x = x[direction_mask]
    fit_y = y[direction_mask]
    fit_weights = weights[direction_mask]
    used_direction_fallback = np.count_nonzero(~direction_mask) > 0
    if used_direction_fallback and verbose:
        print(
            "speed fit: rejected reversal aliases; "
            f"using {len(fit_x)}/{len(x)} direction-consistent samples "
            f"from {np.min(fit_x):g} to {np.max(fit_x):g}"
        )

    support_min = float(np.min(fit_x))
    support_max = float(np.max(fit_x))
    span = support_max - support_min
    knot_count = max(
        4,
        int(math.ceil(span / SPEED_CORRELATION_KNOT_SPACING)) + 1,
    )
    knots = np.linspace(support_min, support_max, knot_count, dtype=np.float64)
    log_magnitude = np.log(np.abs(fit_y))
    normalized_weights = fit_weights / np.max(fit_weights)
    initial_coefficients = np.full(
        knot_count,
        math.log(abs(weighted_median(fit_y, fit_weights))),
        dtype=np.float64,
    )
    curvature_count = max(
        17,
        int(math.ceil(span / SPEED_CORRELATION_CURVATURE_SAMPLE_SPACING)) + 1,
    )
    curvature_x = np.linspace(
        support_min,
        support_max,
        curvature_count,
        dtype=np.float64,
    )

    def model(coefficients):
        return CubicSpline(knots, coefficients, bc_type="natural")

    def huber_residual(values):
        """Encode Huber cost while leaving curvature strictly quadratic."""
        values = np.asarray(values, dtype=np.float64)
        magnitude = np.abs(values)
        outside = magnitude > c
        result = values.copy()
        result[outside] = np.sign(values[outside]) * np.sqrt(
            2.0 * c * magnitude[outside] - c**2
        )
        return result

    def residuals(coefficients):
        speed_model = model(coefficients)
        data_residual = (
            (speed_model(fit_x) - log_magnitude)
            * np.sqrt(normalized_weights)
            / SPEED_CORRELATION_LOG_SIGMA
        )
        curvature_residual = (
            speed_model(curvature_x, 2)
            * SPEED_CORRELATION_CURVATURE_LENGTH**2
            / SPEED_CORRELATION_CURVATURE_LOG_SIGMA
        )
        return np.concatenate(
            (huber_residual(data_residual), curvature_residual)
        )

    solution = least_squares(
        residuals,
        initial_coefficients,
        method="trf",
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=1000,
    )
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise RuntimeError(f"speed correlation fit failed: {solution.message}")
    log_spline = model(solution.x)
    direction = math.copysign(1.0, dominant_motion)

    def spline(values):
        return direction * np.exp(log_spline(values))

    spline.derivative = lambda: (
        lambda values: spline(values) * log_spline(values, 1)
    )
    standardized_residual = (
        (log_spline(fit_x) - log_magnitude)
        * np.sqrt(normalized_weights)
        / SPEED_CORRELATION_LOG_SIGMA
    )
    robust_factor = np.minimum(
        1.0,
        c / np.maximum(np.abs(standardized_residual), 1.0e-12),
    )
    current_weights = fit_weights * robust_factor

    extrapolated_spline = _smooth_speed_extrapolator(
        spline,
        support_min,
        support_max,
    )

    # Plot the original data and the fitted spline with weights below.
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(x, y, "bo", label="Data")
    x_dense = np.linspace(np.min(x), np.max(x), 500)
    y_dense = extrapolated_spline(x_dense)
    ax1.plot(x_dense, y_dense, "r-", lw=2, label="Stiff log-speed spline")
    if used_direction_fallback:
        ax1.plot(
            fit_x,
            fit_y,
            "go",
            fillstyle="none",
            label="Direction-consistent support",
        )
    ax1.plot(x, np.zeros_like(x), "k--")
    ax1.set_title("Stiff Correlation Speed Fit")
    ax1.set_ylabel("y")
    ax1.legend()

    ax2.plot(x, initial_weights, "k.", label="Initial weights")
    ax2.plot(fit_x, current_weights, "r.", label="Final weights")
    ax2.set_xlabel("x")
    ax2.set_ylabel("weight")
    ax2.legend()

    plt.savefig(g / "spline.png")
    plt.close(fig)
    if visualize:
        plt.show()

    if (np.max(y_dense) > 0) != (np.min(y_dense) > 0):
        raise ValueError("Oscillating spline found")

    extrapolated_spline.motion_direction = int(
        np.sign(weighted_median(fit_y, fit_weights))
    )
    extrapolated_spline.correlation_x = fit_x.copy()
    extrapolated_spline.correlation_weights = current_weights.copy()
    return extrapolated_spline


def corrected_interval_length(
    spline,
    x1: float,
    x2: float,
    *,
    max_step: float = 64.0,
) -> float:
    """Integrate output pixels produced by a raw-coordinate interval."""
    lower, upper = sorted((float(x1), float(x2)))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower == upper:
        raise ValueError("interval endpoints must be finite and distinct")
    sample_count = max(33, int(math.ceil((upper - lower) / max_step)) + 1)
    xs = np.linspace(lower, upper, sample_count, dtype=np.float64)
    speed = np.abs(np.asarray(spline(xs), dtype=np.float64))
    if not np.all(np.isfinite(speed)) or np.any(speed <= 0.0):
        raise ValueError("speed spline is invalid inside an equal-width interval")
    return float(np.trapezoid(1.0 / speed, xs))


def _best_fit_equal_width(
    lengths: np.ndarray,
    segments: tuple[EqualWidthSegment, ...],
) -> float:
    """Return the fixed-spline optimum for a group's latent output width."""
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.shape != (len(segments),):
        raise ValueError("equal-width lengths and segments must match")
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("equal-width lengths must be finite and positive")
    log_sigmas = np.asarray(
        [segment.log_width_sigma for segment in segments],
        dtype=np.float64,
    )
    weights = 1.0 / np.square(log_sigmas)
    return float(np.exp(np.average(np.log(lengths), weights=weights)))


def _speed_constraint_diagnostics(
    base_spline,
    refined_spline,
    constraints: ManualSpeedConstraints,
    *,
    refined_group_widths: dict[str, float] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if refined_group_widths is None:
        refined_group_widths = getattr(
            refined_spline,
            "equal_width_group_widths",
            {},
        )
    circles = []
    for circle in constraints.circles:
        base_speed = float(abs(base_spline(circle.center_x)))
        refined_speed = float(abs(refined_spline(circle.center_x)))
        target = circle.speed_magnitude
        circles.append(
            {
                "id": circle.identifier,
                "x": circle.center_x,
                "target_speed_magnitude": target,
                "correlation_speed_magnitude": base_speed,
                "refined_speed_magnitude": refined_speed,
                "correlation_relative_residual": base_speed / target - 1.0,
                "refined_relative_residual": refined_speed / target - 1.0,
            }
        )

    groups = []
    for group in constraints.width_groups:
        if len(group.segments) < 2:
            continue
        base_lengths = np.asarray(
            [
                corrected_interval_length(base_spline, segment.x1, segment.x2)
                for segment in group.segments
            ],
            dtype=np.float64,
        )
        refined_lengths = np.asarray(
            [
                corrected_interval_length(refined_spline, segment.x1, segment.x2)
                for segment in group.segments
            ],
            dtype=np.float64,
        )
        base_target = _best_fit_equal_width(base_lengths, group.segments)
        refined_target = refined_group_widths.get(group.identifier)
        if refined_target is None:
            refined_target = _best_fit_equal_width(refined_lengths, group.segments)
        if not math.isfinite(refined_target) or refined_target <= 0.0:
            raise ValueError(
                f"equal-width group {group.identifier} has an invalid fitted width"
            )
        segments = []
        for segment, base_length, refined_length in zip(
            group.segments, base_lengths, refined_lengths
        ):
            segments.append(
                {
                    "id": segment.identifier,
                    "x1": segment.x1,
                    "x2": segment.x2,
                    "annotation_width_pixels": segment.width,
                    "annotation_sigma_pixels": (
                        SPEED_ANNOTATION_DIMENSION_SIGMA_PIXELS
                    ),
                    "correlation_output_length": float(base_length),
                    "refined_output_length": float(refined_length),
                    "correlation_relative_residual": float(
                        base_length / base_target - 1.0
                    ),
                    "refined_relative_residual": float(
                        refined_length / refined_target - 1.0
                    ),
                    "correlation_annotation_residual_pixels": float(
                        math.log(base_length / base_target) * segment.width
                    ),
                    "refined_annotation_residual_pixels": float(
                        math.log(refined_length / refined_target) * segment.width
                    ),
                }
            )
        groups.append(
            {
                "id": group.identifier,
                "name": group.name,
                "correlation_group_width": base_target,
                "refined_group_width": refined_target,
                "segments": segments,
            }
        )
    return circles, groups


def refine_speed_spline_with_constraints(
    base_spline,
    constraints: ManualSpeedConstraints,
    *,
    x_min: float,
    x_max: float,
    diagnostic_dir: Path | None = None,
    diagnostic_prefix: str = "speed_constraint_fit",
    global_x_offset: float = 0.0,
    verbose: bool = True,
):
    """Fit a smooth speed magnitude while preserving correlation direction.

    Correlation supplies the otherwise-ambiguous motion direction and a weak
    prior where its windows contain evidence.  When start/end cuts exist,
    background correlation outside them is excluded entirely.  The optimized
    log-speed spline itself is physically stiff, so endpoint behavior or local
    errors in the correlation curve cannot be baked into the manual
    refinement. Circle annotations set absolute scale. Each equal-width group
    gets a jointly fitted common output width, and its annotated intervals
    constrain their integrals of reciprocal speed to that width.
    """
    active_groups = tuple(
        group for group in constraints.width_groups if len(group.segments) >= 2
    )
    if not constraints.circles and not active_groups:
        return base_spline
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_max <= x_min:
        raise ValueError("constraint fit requires a non-empty finite x domain")

    if constraints.has_cut_regions:
        fit_x_min = max(x_min, float(constraints.cut_positions[0]))
        fit_x_max = min(x_max, float(constraints.cut_positions[-1]))
    else:
        fit_x_min = x_min
        fit_x_max = x_max
    if fit_x_max <= fit_x_min:
        raise ValueError("annotated start/end do not overlap the speed-fit domain")

    probe_x = np.linspace(fit_x_min, fit_x_max, 257, dtype=np.float64)
    base_probe = np.asarray(base_spline(probe_x), dtype=np.float64)
    if not np.all(np.isfinite(base_probe)) or np.any(base_probe == 0.0):
        raise ValueError("correlation spline is invalid in the constraint-fit domain")
    direction = float(
        getattr(base_spline, "motion_direction", np.sign(np.median(base_probe)))
    )
    if direction == 0.0 or np.any(np.sign(base_probe) != direction):
        raise ValueError("correlation spline must have one direction before refinement")

    circle_data = []
    for circle in constraints.circles:
        if not fit_x_min <= circle.center_x <= fit_x_max:
            continue
        target = circle.speed_magnitude
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError(f"circle {circle.identifier} implies invalid speed")
        circle_data.append((circle, target))

    group_grids = []
    for group in active_groups:
        segment_grids = []
        for segment in group.segments:
            if segment.x1 < fit_x_min or segment.x2 > fit_x_max:
                continue
            count = max(
                33,
                int(math.ceil((segment.x2 - segment.x1) / 64.0)) + 1,
            )
            segment_grids.append(
                (
                    segment,
                    np.linspace(segment.x1, segment.x2, count, dtype=np.float64),
                )
            )
        if len(segment_grids) >= 2:
            group_grids.append((group, tuple(segment_grids)))

    if not circle_data and not group_grids:
        return base_spline

    span = fit_x_max - fit_x_min
    coarse_knot_count = max(
        4,
        int(math.ceil(span / SPEED_CONSTRAINT_KNOT_SPACING)) + 1,
    )
    # Keep the representation independent of annotation locations. Adding a
    # knot at every annotation gives the fit a local degree of freedom that can
    # bend toward a constraint and immediately return to the correlation prior.
    # A fixed coarse basis makes every constraint adjust the shared smooth
    # speed field instead.
    knots = np.linspace(
        fit_x_min,
        fit_x_max,
        coarse_knot_count,
        dtype=np.float64,
    )
    initial_magnitude = np.abs(np.asarray(base_spline(knots), dtype=np.float64))
    if not np.all(np.isfinite(initial_magnitude)) or np.any(
        initial_magnitude <= 0.0
    ):
        raise ValueError("correlation spline cannot initialize the speed fit")
    initial_speed_coefficients = np.log(initial_magnitude)
    speed_coefficient_count = len(initial_speed_coefficients)
    initial_group_log_widths = []
    for _group, segment_grids in group_grids:
        segments = tuple(segment for segment, _xs in segment_grids)
        base_lengths = np.asarray(
            [
                corrected_interval_length(base_spline, segment.x1, segment.x2)
                for segment in segments
            ],
            dtype=np.float64,
        )
        initial_group_log_widths.append(
            math.log(_best_fit_equal_width(base_lengths, segments))
        )
    initial = np.concatenate(
        (
            initial_speed_coefficients,
            np.asarray(initial_group_log_widths, dtype=np.float64),
        )
    )

    correlation_x = np.asarray(
        getattr(base_spline, "correlation_x", knots), dtype=np.float64
    )
    correlation_weights = np.asarray(
        getattr(base_spline, "correlation_weights", np.ones_like(correlation_x)),
        dtype=np.float64,
    )
    if correlation_x.ndim != 1 or correlation_weights.shape != correlation_x.shape:
        raise ValueError("correlation prior coordinates and weights must match")
    correlation_mask = (
        np.isfinite(correlation_x)
        & np.isfinite(correlation_weights)
        & (correlation_weights > 0.0)
        & (correlation_x >= fit_x_min)
        & (correlation_x <= fit_x_max)
    )
    correlation_x = correlation_x[correlation_mask]
    correlation_weights = correlation_weights[correlation_mask]
    if correlation_x.size:
        correlation_weights = np.sqrt(
            correlation_weights / np.max(correlation_weights)
        )
        correlation_log_magnitude = np.log(
            np.abs(np.asarray(base_spline(correlation_x), dtype=np.float64))
        )
    else:
        correlation_log_magnitude = np.empty(0, dtype=np.float64)

    curvature_sample_count = max(
        17,
        int(math.ceil(span / SPEED_CONSTRAINT_CURVATURE_SAMPLE_SPACING)) + 1,
    )
    curvature_x = np.linspace(
        fit_x_min,
        fit_x_max,
        curvature_sample_count,
        dtype=np.float64,
    )

    def speed_log_spline(coefficients):
        # No annotation constrains the derivative just outside the cut range.
        # A natural spline permits an arbitrary linear ramp there (zero second
        # derivative), which caused the visible endpoint departures reported
        # on otherwise constant-speed trains. Clamped ends encode the physical
        # expectation that speed is locally steady at the annotated boundaries.
        return CubicSpline(knots, coefficients, bc_type="clamped")

    def evaluate(coefficients, values):
        values_array = np.asarray(values, dtype=np.float64)
        flat_values = values_array.reshape(-1)
        model = speed_log_spline(coefficients)
        log_magnitude = np.empty_like(flat_values)
        left = flat_values < fit_x_min
        right = flat_values > fit_x_max
        inside = ~(left | right)
        log_magnitude[inside] = model(flat_values[inside])
        log_magnitude[left] = model(fit_x_min) + model(
            fit_x_min, 1
        ) * (flat_values[left] - fit_x_min)
        log_magnitude[right] = model(fit_x_max) + model(
            fit_x_max, 1
        ) * (flat_values[right] - fit_x_max)
        log_magnitude = log_magnitude.reshape(values_array.shape)
        result = direction * np.exp(log_magnitude)
        if values_array.ndim == 0:
            return float(result)
        return result

    def residuals(parameters):
        coefficients = parameters[:speed_coefficient_count]
        group_log_widths = parameters[speed_coefficient_count:]
        result = list(
            (
                speed_log_spline(coefficients)(correlation_x)
                - correlation_log_magnitude
            )
            * correlation_weights
            / SPEED_CONSTRAINT_PRIOR_LOG_SIGMA
        )
        # Penalize curvature in physical x units.  Unlike coefficient second
        # differences, this does not make the fit wigglier merely because the
        # basis contains more representation points.
        result.extend(
            speed_log_spline(coefficients)(curvature_x, 2)
            * SPEED_CONSTRAINT_CURVATURE_LENGTH**2
            / SPEED_CONSTRAINT_CURVATURE_LOG_SIGMA
        )
        for circle, target in circle_data:
            speed = abs(evaluate(coefficients, circle.center_x))
            result.append(math.log(speed / target) / circle.log_speed_sigma)
        for group_index, (_group, segment_grids) in enumerate(group_grids):
            lengths = []
            log_sigmas = []
            for segment, xs in segment_grids:
                speed = np.abs(evaluate(coefficients, xs))
                lengths.append(float(np.trapezoid(1.0 / speed, xs)))
                log_sigmas.append(segment.log_width_sigma)
            log_lengths = np.log(np.asarray(lengths, dtype=np.float64))
            result.extend(
                (log_lengths - group_log_widths[group_index])
                / np.asarray(log_sigmas, dtype=np.float64)
            )
        return np.asarray(result, dtype=np.float64)

    solution = least_squares(
        residuals,
        initial,
        method="trf",
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=1000,
    )
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise RuntimeError(f"manual speed-constraint fit failed: {solution.message}")

    coefficients = solution.x[:speed_coefficient_count].copy()
    group_log_widths = solution.x[speed_coefficient_count:].copy()
    equal_width_group_widths = {
        group.identifier: float(math.exp(log_width))
        for (group, _segment_grids), log_width in zip(
            group_grids,
            group_log_widths,
        )
    }

    def refined_spline(values):
        return evaluate(coefficients, values)

    refined_spline.equal_width_group_widths = equal_width_group_widths

    circle_diagnostics, group_diagnostics = _speed_constraint_diagnostics(
        base_spline,
        refined_spline,
        ManualSpeedConstraints(
            constraints.path,
            tuple(circle for circle, _target in circle_data),
            tuple(
                EqualWidthGroup(
                    group.identifier,
                    group.name,
                    tuple(segment for segment, _xs in grids),
                )
                for group, grids in group_grids
            ),
            constraints.cut_positions,
        ),
        refined_group_widths=equal_width_group_widths,
    )

    if verbose:
        fitted_knot_speeds = np.exp(coefficients)
        circle_before = [
            abs(float(entry["correlation_relative_residual"]))
            for entry in circle_diagnostics
        ]
        circle_after = [
            abs(float(entry["refined_relative_residual"]))
            for entry in circle_diagnostics
        ]
        print(
            "manual speed constraints: "
            f"{len(circle_diagnostics)} circles, {len(group_diagnostics)} width groups; "
            f"fitted speed magnitude {np.min(fitted_knot_speeds):.6f} to "
            f"{np.max(fitted_knot_speeds):.6f}"
        )
        if circle_before:
            print(
                "circle median absolute residual: "
                f"{100 * np.median(circle_before):.3f}% -> "
                f"{100 * np.median(circle_after):.3f}%"
            )
        for group in group_diagnostics:
            before = [
                abs(float(segment["correlation_annotation_residual_pixels"]))
                for segment in group["segments"]
            ]
            after = [
                abs(float(segment["refined_annotation_residual_pixels"]))
                for segment in group["segments"]
            ]
            print(
                f"equal-width group {group['name']!r} max annotation residual: "
                f"{max(before):.3f}px -> {max(after):.3f}px"
            )

    if diagnostic_dir is not None:
        diagnostic_dir = Path(diagnostic_dir)
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(constraints.path),
            "motion_direction": int(direction),
            "annotation_dimension_sigma_pixels": (
                SPEED_ANNOTATION_DIMENSION_SIGMA_PIXELS
            ),
            "optimizer": {
                "success": bool(solution.success),
                "message": str(solution.message),
                "cost": float(solution.cost),
                "function_evaluations": int(solution.nfev),
            },
            "speed_knots": [
                {
                    "x": float(x + global_x_offset),
                    "speed_magnitude": float(math.exp(value)),
                }
                for x, value in zip(knots, coefficients)
            ],
            "correction_knots": [
                {
                    "x": float(x + global_x_offset),
                    "scale": float(
                        math.exp(value) / abs(float(base_spline(x)))
                    ),
                }
                for x, value in zip(knots, coefficients)
            ],
            "circles": [
                {**entry, "x": float(entry["x"]) + global_x_offset}
                for entry in circle_diagnostics
            ],
            "equal_width_groups": [
                {
                    **group,
                    "segments": [
                        {
                            **segment,
                            "x1": float(segment["x1"]) + global_x_offset,
                            "x2": float(segment["x2"]) + global_x_offset,
                        }
                        for segment in group["segments"]
                    ],
                }
                for group in group_diagnostics
            ],
        }
        json_path = diagnostic_dir / f"{diagnostic_prefix}.json"
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        dense_x = np.linspace(fit_x_min, fit_x_max, 1000, dtype=np.float64)
        fig, (ax_speed, ax_scale) = plt.subplots(
            2,
            1,
            figsize=(16, 9),
            dpi=200,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        global_dense_x = dense_x + global_x_offset
        base_dense = np.abs(np.asarray(base_spline(dense_x), dtype=np.float64))
        correlation_support = getattr(
            base_spline,
            "support",
            (fit_x_min, fit_x_max),
        )
        correlation_mask = (dense_x >= correlation_support[0]) & (
            dense_x <= correlation_support[1]
        )
        ax_speed.plot(
            global_dense_x[correlation_mask],
            base_dense[correlation_mask],
            "#7f8c8d",
            linewidth=1.5,
            label="Correlation fit",
        )
        ax_speed.plot(
            global_dense_x,
            np.abs(refined_spline(dense_x)),
            "#00a8ff",
            linewidth=2.0,
            label="Constraint-refined fit",
        )
        if circle_data:
            ax_speed.plot(
                [circle.center_x + global_x_offset for circle, _target in circle_data],
                [target for _circle, target in circle_data],
                "o",
                color="#00b84a",
                label="Circle target",
            )
        ax_speed.set_ylabel("|input px / output px|")
        ax_speed.set_title("Manual speed-constraint refinement")
        ax_speed.legend()
        ax_speed.grid(alpha=0.2)

        refined_dense = np.abs(np.asarray(refined_spline(dense_x), dtype=np.float64))
        ax_scale.plot(
            global_dense_x[correlation_mask],
            (refined_dense[correlation_mask] / base_dense[correlation_mask] - 1.0)
            * 100.0,
            "#ff8c42",
        )
        ax_scale.axhline(0.0, color="black", linewidth=0.7)
        ax_scale.set_xlabel("global binned Bayer x")
        ax_scale.set_ylabel("difference from correlation (%)")
        ax_scale.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(diagnostic_dir / f"{diagnostic_prefix}.png")
        plt.close(fig)

    return refined_spline


def raw_channels(data: np.ndarray):
    raw_red = data[0::2, 1::2]
    raw_green_1 = data[1::2, 1::2]
    raw_green_2 = data[0::2, 0::2]
    raw_blue = data[1::2, 0::2]

    return raw_red, raw_green_1, raw_green_2, raw_blue


def sharpen(rgb: np.ndarray, *, in_place: bool = False) -> np.ndarray:
    blur_small = cv2.GaussianBlur(rgb, (0, 0), 1)
    blur_large = cv2.GaussianBlur(rgb, (0, 0), 2)
    blur_small += blur_large
    del blur_large
    if not in_place:
        rgb = rgb.copy()
    np.multiply(rgb, 1.4, out=rgb)
    np.multiply(blur_small, 0.2, out=blur_small)
    np.subtract(rgb, blur_small, out=rgb)
    return rgb


def patch_denoise(
    data: np.ndarray,
    neighbour_size: int = 64,
    similarity_thresh: float = 3,
    *,
    show_progress: bool = True,
):
    # Patch-based denoising by searching horizontally along rows for similar 3x3 patches.
    denoised = data.copy().astype(np.float32)  # Ensure float32 for speed/memory.
    sqrt_data = np.sqrt(data).astype(np.float32)

    feature_size = 9
    height, width, channels = data.shape
    N = width - 2  # Number of patches per row.

    # Center indices in the feature vector (one per channel).
    center_indices = np.array([0, feature_size, 2 * feature_size])

    time_ab = 0
    time_bc = 0
    time_cd = 0

    for row in progress(
        range(1, height - 1),
        desc="denoising",
        show=show_progress,
    ):
        time_a = time.time()

        # Extract features (27 x N, float32).
        feature = np.zeros((3 * feature_size, N), dtype=np.float32)
        for channel in range(3):
            offset = channel * feature_size
            feature[offset + 0] = sqrt_data[row, 1:-1, channel]  # Center.
            feature[offset + 1] = sqrt_data[row, :-2, channel]  # Left.
            feature[offset + 2] = sqrt_data[row, 2:, channel]  # Right.
            feature[offset + 3] = sqrt_data[row - 1, 1:-1, channel]  # Top center.
            feature[offset + 4] = sqrt_data[row - 1, :-2, channel]  # Top left.
            feature[offset + 5] = sqrt_data[row - 1, 2:, channel]  # Top right.
            feature[offset + 6] = sqrt_data[row + 1, 1:-1, channel]  # Bottom center.
            feature[offset + 7] = sqrt_data[row + 1, :-2, channel]  # Bottom left.
            feature[offset + 8] = sqrt_data[row + 1, 2:, channel]  # Bottom right.

        time_b = time.time()

        sort_index = np.argsort(np.sum(feature, axis=0))

        feature_sorted = feature[:, sort_index]

        time_c = time.time()

        pad_value = -100.0
        w = 2 * neighbour_size + 1
        padded = np.pad(
            feature_sorted,
            ((0, 0), (neighbour_size, neighbour_size)),
            mode="constant",
            constant_values=pad_value,
        )

        # Sliding window view: (27, N, w).
        windows = sliding_window_view(padded, w, axis=1)

        # Center position in each window.
        center_pos = neighbour_size

        # Compute mask (N, w) by accumulating per-dimension checks (saves memory).
        mask = np.ones((N, w), dtype=bool)
        for dim in range(feature.shape[0]):
            dim_window = windows[dim, :, :]  # (N, w)
            center_dim = dim_window[:, center_pos]  # (N,)
            abs_diff = np.abs(dim_window - center_dim[:, np.newaxis])  # (N, w)
            mask &= abs_diff < similarity_thresh

        # Compute Gaussian weights based on distances for valid patches only
        gaussian_weights = np.zeros((N, w), dtype=np.float32)
        for i in range(N):
            center_feature = feature_sorted[:, i]  # (27,)
            window_features = windows[:, i, :]  # (27, w)
            valid_mask = mask[i, :] & (
                window_features[0, :] != pad_value
            )  # Valid patches (True in mask and not padded)
            valid_window = window_features[:, valid_mask]  # (27, num_valid)
            if valid_window.size > 0:  # Only compute if there are valid patches
                diff = valid_window - center_feature[:, np.newaxis]  # (27, num_valid)
                distances = np.sqrt(np.sum(diff**2, axis=0))  # (num_valid,)
                weights = np.exp(
                    -0.5 * (distances**2) / (similarity_thresh**2)
                )  # (num_valid,)
                weights_sum = np.sum(weights)
                if weights_sum > 0:  # Avoid division by zero
                    weights /= weights_sum  # Normalize to sum to 1
                gaussian_weights[i, valid_mask] = weights  # Assign to valid positions

        # Compute weighted means for each channel using Gaussian weights.
        for ch in range(3):
            center_window = windows[center_indices[ch], :, :]  # (N, w)
            masked = np.ma.masked_array(center_window, mask=~mask)
            # Apply Gaussian weights to the valid (unmasked) data
            weighted = masked * gaussian_weights
            means_sqrt = np.ma.sum(
                weighted, axis=1
            )  # Weighted sum (normalized to mean)
            # Assign to original positions.
            original_cols = sort_index + 1  # Offset for image indexing.
            denoised[row, original_cols, ch] = means_sqrt**2

        time_d = time.time()

        time_ab += time_b - time_a
        time_bc += time_c - time_b
        time_cd += time_d - time_c

    if show_progress:
        print("times", time_ab, time_bc, time_cd)
    return denoised


def bin2to1(data):
    h = (data.shape[0] // 2) * 2
    w = (data.shape[1] // 2) * 2
    if h == 0 or w == 0:
        return np.empty((0, 0), dtype=data.dtype)
    data = data[:h, :w]
    return (data[::2, ::2] + data[1::2, ::2] + data[::2, 1::2] + data[1::2, 1::2]) / 4


def bin2to1_rgb_u8(data: np.ndarray) -> np.ndarray:
    """Average 2x2 uint8 RGB/BGR pixels without a full-size float32 copy."""
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[2] != 3 or data.dtype != np.uint8:
        raise ValueError("data must be an HxWx3 uint8 array")
    height = (data.shape[0] // 2) * 2
    width = (data.shape[1] // 2) * 2
    if height == 0 or width == 0:
        return np.empty((height // 2, width // 2, 3), dtype=np.uint8)

    summed = data[:height:2, :width:2].astype(np.uint16)
    summed += data[1:height:2, :width:2]
    summed += data[:height:2, 1:width:2]
    summed += data[1:height:2, 1:width:2]
    np.right_shift(summed, 2, out=summed)
    return summed.astype(np.uint8)


def moving_region_block_scores(
    green_view,
    max_green: float,
    *,
    block_columns: int = MOVING_REGION_BLOCK_COLUMNS,
    show_progress: bool = True,
) -> list[tuple[int, int, float]]:
    """Score motion-bearing structure in fixed global-column windows."""
    if block_columns < 2 or block_columns % 2:
        raise ValueError("block_columns must be a positive even number")
    results = []
    starts = range(0, green_view.width, block_columns)
    total_blocks = math.ceil(green_view.width / block_columns)
    overlap = 4
    signal_floor = max(0.1 * float(max_green), np.finfo(np.float32).eps)
    for start in progress(
        starts,
        total=total_blocks,
        desc="finding moving region",
        show=show_progress,
    ):
        stop = min(start + block_columns, green_view.width)
        read_start = max(0, start - overlap)
        read_stop = min(green_view.width, stop + overlap)
        raw_green = bin2to1(green_view[:, slice(read_start, read_stop)])

        grad_y, grad_x = np.gradient(raw_green)
        magnitude = np.square(grad_x)
        magnitude += np.square(grad_y)
        np.sqrt(magnitude, out=magnitude)
        magnitude += signal_floor
        np.abs(grad_x, out=grad_x)
        grad_x /= magnitude
        core_start = (start - read_start) // 2
        core_width = (stop - start) // 2
        score = float(
            np.percentile(
                grad_x[:, core_start : core_start + core_width],
                99,
            )
        )
        results.append((start, stop, score))
    return results


def estimate_white_balance(
    bin_paths,
    max_sample_columns: int = WHITE_BALANCE_SAMPLE_COLUMNS,
    *,
    show_progress: bool = True,
) -> np.ndarray:
    if not bin_paths:
        return np.ones(3, dtype=np.float32)
    if max_sample_columns < 1:
        raise ValueError("max_sample_columns must be positive")

    sample_plan = []
    remaining_columns = max_sample_columns
    bytes_per_raw_column = 4096 * np.dtype(np.uint16).itemsize
    for bin_path in bin_paths:
        raw_columns = bin_path.stat().st_size // bytes_per_raw_column
        display_columns = raw_columns // 2
        take_columns = min(display_columns, remaining_columns)
        # The final 2x2 average needs an even count in Bayer-channel space.
        take_columns -= take_columns % 2
        if take_columns:
            sample_plan.append((bin_path, take_columns))
            remaining_columns -= take_columns
        if remaining_columns < 2:
            break

    channel_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for bin_path, display_columns in progress(
        sample_plan,
        desc="white balance",
        total=len(sample_plan),
        show=show_progress,
    ):
        dat1_reshaped = load_raw_frame(
            bin_path,
            rows=4096,
            columns=display_columns * 2,
            output_dtype=np.float32,
        )
        raw_red = bin2to1(dat1_reshaped[0::2, 1::2])
        raw_green = bin2to1(dat1_reshaped[1::2, 1::2])
        raw_blue = bin2to1(dat1_reshaped[1::2, 0::2])

        channel_sum[0] += raw_red.sum()
        channel_sum[1] += raw_green.sum()
        channel_sum[2] += raw_blue.sum()
        total_pixels += raw_red.size

    means = channel_sum / max(total_pixels, 1)
    mean_of_means = np.mean(means)
    with np.errstate(divide="ignore", invalid="ignore"):
        white_balance = mean_of_means / means
    white_balance[~np.isfinite(white_balance)] = 1.0
    return white_balance.astype(np.float32)


def get_percentile(hist, bin_edges, total, percent):
    if total == 0:
        return 0.0
    cum = np.cumsum(hist)
    thresh = (percent / 100.0) * total
    idx = np.searchsorted(cum, thresh, side="left")
    if idx == 0:
        return bin_edges[0]
    if idx == len(hist):
        return bin_edges[-1]
    prev_cum = cum[idx - 1]
    needed = thresh - prev_cum
    bin_count = hist[idx]
    frac = needed / bin_count if bin_count > 0 else 0
    bin_width = bin_edges[idx + 1] - bin_edges[idx]
    return bin_edges[idx] + frac * bin_width


def autoexposure(selected_bins, white_balance, *, show_progress: bool = True):
    num_bins = 65536
    bin_edges = np.arange(0, num_bins + 1, dtype=np.float64)
    histogram = np.zeros(num_bins, dtype=np.int64)
    total_values = 0

    for i, bin in progress(
        enumerate(selected_bins),
        total=len(selected_bins),
        desc="autoexposure",
        show=show_progress,
    ):
        dat1_reshaped = load_raw_frame(
            bin,
            rows=4096,
            output_dtype=np.float32,
        )
        raw_red = dat1_reshaped[0::2, 1::2]
        raw_red = bin2to1(raw_red)

        raw_blue = dat1_reshaped[1::2, 0::2]
        raw_blue = bin2to1(raw_blue)

        raw_green = dat1_reshaped[1::2, 1::2]
        raw_green = bin2to1(raw_green)

        stacked = np.stack((raw_red, raw_green, raw_blue), axis=2)
        rgb = apply_color_transform(stacked, white_balance)
        rgb = np.clip(rgb, 0, 65536)

        h, _ = np.histogram(rgb.ravel(), bins=bin_edges)
        histogram += h
        total_values += rgb.size
    p2 = get_percentile(histogram, bin_edges, total_values, 2)
    p98 = get_percentile(histogram, bin_edges, total_values, 98)
    return p2, p98


def build_structure_mask(
    green_view,
    *,
    downsample_steps: int,
    max_green_down: float,
    block_columns: int = STRUCTURE_MASK_BLOCK_COLUMNS,
    show_progress: bool = True,
) -> np.ndarray:
    """Build the jitter mask in fixed global-column blocks.

    The overlap supplies gradient context at analysis-block edges. Neither the
    mask values nor their boundaries depend on how the raw capture was split
    into files.
    """
    downsample = 2**downsample_steps
    if block_columns < downsample or block_columns % downsample:
        raise ValueError("block_columns must be divisible by the downsample factor")
    mask_blocks = []
    starts = range(0, green_view.width, block_columns)
    total_blocks = math.ceil(green_view.width / block_columns)
    overlap = 2 * downsample
    for start in progress(
        starts,
        total=total_blocks,
        desc="stripe mask",
        show=show_progress,
    ):
        stop = min(start + block_columns, green_view.width)
        read_start = max(0, start - overlap)
        read_stop = min(green_view.width, stop + overlap)
        raw_green = green_view[:, slice(read_start, read_stop)]
        for _ in range(downsample_steps):
            raw_green = bin2to1(raw_green)

        grad_y, grad_x = np.gradient(raw_green)
        magnitude = np.square(grad_x)
        magnitude += np.square(grad_y)
        np.sqrt(magnitude, out=magnitude)
        magnitude += 0.1 * max_green_down
        np.abs(grad_x, out=grad_x)
        grad_x /= magnitude
        core_start = (start - read_start) // downsample
        core_width = (stop - start) // downsample
        mask_blocks.append(
            grad_x[:, core_start : core_start + core_width]
            > STRUCTURE_MASK_THRESHOLD
        )

    if not mask_blocks:
        return np.empty((0, 0), dtype=bool)
    return np.hstack(mask_blocks)


class ColumnJitterModel:
    """Channel-specific additive corrections at vertical band centers."""

    def __init__(
        self,
        band_corrections: np.ndarray,
        image_height: int,
        black_levels: np.ndarray,
        white_levels: np.ndarray,
        confidence: np.ndarray,
        fine_band_corrections: np.ndarray | None = None,
        scene_protection: np.ndarray | None = None,
    ):
        corrections = np.asarray(band_corrections)
        if corrections.ndim != 3 or corrections.shape[0] != 4:
            raise ValueError("band_corrections must have shape (4, bands, columns)")
        if image_height < 1 or corrections.shape[1] < 1:
            raise ValueError("image height and jitter band count must be positive")
        self.band_corrections = corrections
        self.image_height = int(image_height)
        self.black_levels = np.asarray(black_levels, dtype=np.float32)
        self.white_levels = np.asarray(white_levels, dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.num_bands = corrections.shape[1]
        self.width = corrections.shape[2]
        if fine_band_corrections is None:
            self.fine_band_corrections = None
        else:
            fine = np.asarray(fine_band_corrections)
            if fine.shape != corrections.shape:
                raise ValueError("fine_band_corrections must match band_corrections")
            self.fine_band_corrections = fine
        if scene_protection is None:
            self.scene_protection = None
        else:
            protection = np.asarray(scene_protection)
            if protection.shape != corrections.shape[1:]:
                raise ValueError("scene_protection must have shape (bands, columns)")
            if protection.dtype == np.bool_:
                self.scene_protection = protection.astype(np.float16)
            else:
                for start in range(0, protection.shape[1], JITTER_BLOCK_COLUMNS):
                    block = protection[:, start : start + JITTER_BLOCK_COLUMNS]
                    if (
                        not np.all(np.isfinite(block))
                        or np.any(block < 0.0)
                        or np.any(block > 1.0)
                    ):
                        raise ValueError(
                            "scene_protection values must be between 0 and 1"
                        )
                self.scene_protection = protection.astype(np.float16, copy=False)
        if self.black_levels.shape != (4,) or self.white_levels.shape != (4,):
            raise ValueError("black_levels and white_levels must each contain 4 values")
        if self.confidence.shape != (self.width,):
            raise ValueError("confidence must contain one value per modeled column")

        # Band values live at band centers. Linear interpolation keeps application
        # cheap during motion-corrected sampling while avoiding band boundaries.
        band_position = (
            np.arange(self.image_height, dtype=np.float32) + 0.5
        ) * self.num_bands / self.image_height - 0.5
        lower = np.floor(band_position).astype(np.int32)
        upper = lower + 1
        weight = band_position - lower
        below = lower < 0
        above = upper >= self.num_bands
        lower = np.clip(lower, 0, self.num_bands - 1)
        upper = np.clip(upper, 0, self.num_bands - 1)
        owner = np.floor(
            (np.arange(self.image_height, dtype=np.float32) + 0.5)
            * self.num_bands
            / self.image_height
        ).astype(np.int32)
        owner = np.clip(owner, 0, self.num_bands - 1)
        weight[below] = 0.0
        weight[above] = 0.0
        self._row_band_lower = lower
        self._row_band_upper = upper
        self._row_band_owner = owner
        self._row_band_weight = weight.astype(np.float32)

    def _interpolate_component(
        self,
        component: np.ndarray,
        channel: int,
        column_grid: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        """Interpolate one independently supported correction component."""
        lower = self._row_band_lower[rows]
        upper = self._row_band_upper[rows]
        weight = self._row_band_weight[rows, None]
        channel_model = component[channel]
        correction_low = channel_model[lower[:, None], column_grid].astype(np.float32)
        correction_high = channel_model[upper[:, None], column_grid].astype(np.float32)
        correction = correction_low + (correction_high - correction_low) * weight
        # Interpolation may taper a supported band toward an unsupported
        # neighbor, but it must never carry that neighbor's value across the
        # boundary into a band whose own estimate is zero.
        correction_owner = channel_model[
            self._row_band_owner[rows, None],
            column_grid,
        ]
        return np.where(correction_owner != 0.0, correction, 0.0)

    def _interpolate_scene_protection(
        self,
        column_grid: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        """Interpolate feathered scene confidence between sensor-row bands."""
        lower = self._row_band_lower[rows]
        upper = self._row_band_upper[rows]
        weight = self._row_band_weight[rows, None]
        protection_low = self.scene_protection[
            lower[:, None],
            column_grid,
        ].astype(np.float32)
        protection_high = self.scene_protection[
            upper[:, None],
            column_grid,
        ].astype(np.float32)
        return protection_low + (protection_high - protection_low) * weight

    def correct_samples(
        self,
        values: np.ndarray,
        channel: int,
        columns: np.ndarray,
        rows: np.ndarray | None = None,
        *,
        protect_scene: bool = True,
    ) -> np.ndarray:
        """Correct a 2-D sample array using either 1-D or per-pixel columns."""
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("values must be two-dimensional")
        if not 0 <= channel < 4:
            raise ValueError("channel must be in the range 0..3")
        if rows is None:
            rows = np.arange(values.shape[0], dtype=np.int32)
        else:
            rows = np.asarray(rows, dtype=np.int32)
        if rows.ndim != 1 or rows.size != values.shape[0]:
            raise ValueError("rows must contain one index per values row")

        columns = np.asarray(columns, dtype=np.int32)
        if columns.ndim == 1:
            if columns.size != values.shape[1]:
                raise ValueError("columns must contain one index per values column")
            column_grid = columns[None, :]
        elif columns.shape == values.shape:
            column_grid = columns
        else:
            raise ValueError("columns must be 1-D or have the same shape as values")
        if (
            np.any(rows < 0)
            or np.any(rows >= self.image_height)
            or np.any(column_grid < 0)
            or np.any(column_grid >= self.width)
        ):
            raise IndexError("jitter model sample is outside the modeled image")

        correction = self._interpolate_component(
            self.band_corrections,
            channel,
            column_grid,
            rows,
        )
        if self.fine_band_corrections is not None:
            # The residual component has a scene-edge support mask distinct
            # from the recurrent model. Interpolate and owner-gate it
            # independently so an unmasked neighboring band cannot leak into
            # a protected wheel, window, or bellows band.
            correction += self._interpolate_component(
                self.fine_band_corrections,
                channel,
                column_grid,
                rows,
            )

        black = float(self.black_levels[channel])
        white = float(self.white_levels[channel])
        span = max(white - black, 1.0)
        if protect_scene:
            signal = np.maximum(values - black, 0.0)
            # The model is estimated in full-scale additive units, but that
            # makes the same legal correction disproportionately large near
            # black. In regions with corroborated scene structure, limit the
            # combined recurrent and fine correction relative to the actual
            # black-subtracted sample. Models without motion-aligned scene
            # evidence use this conservative limit everywhere.
            relative_limit = JITTER_MAX_RELATIVE_SIGNAL_CORRECTION * signal
            limited_correction = np.clip(
                correction,
                -relative_limit,
                relative_limit,
            )
            if self.scene_protection is not None:
                # Motion-aligned green edges identify locations where an
                # additive correction could alter real scene contrast. Flat
                # regions retain the full correction.
                protection = self._interpolate_scene_protection(
                    column_grid,
                    rows,
                )
                correction = correction + protection * (limited_correction - correction)
            else:
                correction = limited_correction
        highlight_margin = max(
            JITTER_WHITE_MARGIN_MIN,
            JITTER_WHITE_MARGIN_FRACTION * span,
        )
        shadow_weight = np.clip(
            (values - black) / max(JITTER_BLACK_MARGIN, 1.0), 0.0, 1.0
        )
        highlight_weight = np.clip((white - values) / highlight_margin, 0.0, 1.0)
        clip_weight = np.minimum(shadow_weight, highlight_weight)
        return values + correction * clip_weight

    def apply_window(
        self,
        window: np.ndarray,
        channel: int,
        start_col: int,
        *,
        protect_scene: bool = True,
    ) -> np.ndarray:
        """Apply the model to a contiguous full-height image window."""
        if window.size == 0:
            return window
        window = np.asarray(window, dtype=np.float32)
        result = window.copy()
        local_start = max(0, -start_col)
        local_stop = min(window.shape[1], self.width - start_col)
        if local_stop <= local_start:
            return result
        columns = np.arange(
            start_col + local_start,
            start_col + local_stop,
            dtype=np.int32,
        )
        result[:, local_start:local_stop] = self.correct_samples(
            window[:, local_start:local_stop],
            channel,
            columns,
            protect_scene=protect_scene,
        )
        return result


def apply_column_models(
    window: np.ndarray,
    models: ColumnJitterModel | None,
    start_col: int,
    channel: int,
    *,
    protect_scene: bool = True,
) -> np.ndarray:
    if models is None:
        return window
    return models.apply_window(
        window,
        channel,
        start_col,
        protect_scene=protect_scene,
    )


def _estimate_jitter_levels(channel_views, max_sample_columns: int = 8192):
    """Estimate clipping endpoints from windows spread across the capture."""
    width = channel_views[0].shape[1]
    sample_width = min(512, width)
    num_windows = max(1, min(16, math.ceil(max_sample_columns / sample_width)))
    max_start = max(0, width - sample_width)
    starts = np.unique(np.linspace(0, max_start, num_windows).astype(np.int64))
    black = np.full(4, np.inf, dtype=np.float32)
    white = np.full(4, -np.inf, dtype=np.float32)
    for channel, view in enumerate(channel_views):
        for start in starts:
            sample = view[:, slice(int(start), int(start) + sample_width)]
            if sample.size:
                black[channel] = min(black[channel], float(np.min(sample)))
                white[channel] = max(white[channel], float(np.max(sample)))
    black[~np.isfinite(black)] = 0.0
    white[~np.isfinite(white)] = 65535.0
    white = np.maximum(white, black + 1.0)
    return black, white


def _jitter_band_statistics(
    data: np.ndarray,
    num_bands: int,
    black: float,
    white: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = data.shape
    if height % num_bands:
        raise ValueError(
            f"image height {height} must be divisible by {num_bands} jitter bands"
        )
    banded = data.reshape(num_bands, height // num_bands, width)
    medians = np.median(banded, axis=1).astype(np.float32, copy=False)
    span = max(white - black, 1.0)
    high_margin = max(
        JITTER_WHITE_MARGIN_MIN,
        JITTER_WHITE_MARGIN_FRACTION * span,
    )
    invalid = (
        np.mean(banded <= black + JITTER_BLACK_MARGIN, axis=1)
        > JITTER_MAX_CLIPPED_FRACTION
    )
    invalid |= (
        np.mean(banded >= white - high_margin, axis=1) > JITTER_MAX_CLIPPED_FRACTION
    )
    return medians, ~invalid


def _centered_moving_median(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """Fast centered median with nearest-value padding at both ends."""
    radius = window // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (radius, radius)
    padded = np.pad(values, pad_width, mode="edge")
    trailing = bn.move_median(
        padded,
        window=window,
        min_count=window,
        axis=-1,
    )
    start = window - 1
    return trailing[..., start : start + values.shape[-1]].astype(
        np.float32,
        copy=False,
    )


def _jitter_structure_bands(
    structure_mask: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    num_bands: int,
    scale: int,
) -> np.ndarray:
    """Reduce a downsampled scene-edge mask to jitter band/column support."""
    if scale < 1:
        raise ValueError("structure_mask_scale must be positive")
    if image_height % scale or image_width % scale:
        raise ValueError("jitter image dimensions must be divisible by mask scale")
    expected_shape = (image_height // scale, image_width // scale)
    structure_mask = np.asarray(structure_mask, dtype=bool)
    if structure_mask.shape != expected_shape:
        raise ValueError(
            "structure_mask has shape "
            f"{structure_mask.shape}, expected {expected_shape}"
        )
    if expected_shape[0] % num_bands:
        raise ValueError("downsampled mask height must be divisible by num_bands")
    return structure_mask.reshape(
        num_bands,
        expected_shape[0] // num_bands,
        expected_shape[1],
    ).any(axis=1)


def _winsorize_jitter_candidates(
    normalized_candidates: np.ndarray,
    valid: np.ndarray,
    *,
    num_bands: int,
) -> np.ndarray:
    """
    Limit isolated scene residuals before learning recurrent sensor modes.

    The two physical Bayer columns have different responses, so each
    red/green1 and green2/blue pair gets its own robust center. A wheel edge or
    colored window artifact occupies a minority of the rows in either group
    and is clipped toward that center; a genuine line-wide exposure change
    moves the group center and survives.
    """
    normalized_candidates = np.asarray(normalized_candidates, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    expected_shape = (4 * num_bands, normalized_candidates.shape[1])
    if normalized_candidates.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError("jitter candidates have an unexpected feature shape")

    winsorized = np.zeros_like(normalized_candidates)
    for group_start, group_stop in (
        (0, 2 * num_bands),
        (2 * num_bands, 4 * num_bands),
    ):
        group_valid = valid[group_start:group_stop]
        columns_with_data = np.any(group_valid, axis=0)
        if not np.any(columns_with_data):
            continue
        group_values = np.where(
            group_valid[:, columns_with_data],
            normalized_candidates[group_start:group_stop, columns_with_data],
            np.nan,
        )
        center = bn.nanmedian(group_values, axis=0)
        deviation = bn.nanmedian(np.abs(group_values - center), axis=0)
        robust_scale = np.maximum(
            1.4826 * deviation,
            JITTER_LOW_RANK_MIN_ROBUST_SCALE,
        )
        lower = center - JITTER_LOW_RANK_WINSOR_SIGMA * robust_scale
        upper = center + JITTER_LOW_RANK_WINSOR_SIGMA * robust_scale
        clipped = np.clip(group_values, lower, upper)
        clipped[~np.isfinite(clipped)] = 0.0
        winsorized[
            group_start:group_stop,
            columns_with_data,
        ] = clipped
    return winsorized


def _jitter_low_rank_basis(
    normalized_candidates: np.ndarray,
    valid: np.ndarray,
    *,
    num_bands: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Learn recurrent, sensor-wide correction modes from sampled columns.

    Candidate residuals alone cannot distinguish a real vertical edge from an
    exposure error. The accepted eigenmodes must instead be well above the
    covariance noise floor, occupy many row/channel features, and recur over a
    substantial fraction of the capture. Sparse scene details fail at least
    the latter two tests.
    """
    winsorized = _winsorize_jitter_candidates(
        normalized_candidates,
        valid,
        num_bands=num_bands,
    )
    counts = np.sum(valid, axis=1)
    feature_means = np.divide(
        np.sum(winsorized, axis=1),
        counts,
        out=np.zeros(winsorized.shape[0], dtype=np.float32),
        where=counts > 0,
    )
    centered = np.where(
        valid,
        winsorized - feature_means[:, None],
        0.0,
    )
    if centered.shape[1] < 2 or not np.any(centered):
        return (
            np.empty((centered.shape[0], 0), dtype=np.float32),
            feature_means,
        )

    covariance = centered @ centered.T / float(centered.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    leading = float(eigenvalues[0])
    if not np.isfinite(leading) or leading <= 0.0:
        return (
            np.empty((centered.shape[0], 0), dtype=np.float32),
            feature_means,
        )

    # A coherent sensor mode stands well clear of the independent feature
    # residuals in the middle of the spectrum. The relative epsilon preserves
    # exact low-rank synthetic/capture signals whose remaining eigenvalues are
    # zero up to roundoff.
    noise_floor = max(
        float(np.median(eigenvalues)),
        leading * 1.0e-9,
    )
    candidate_count = min(
        JITTER_LOW_RANK_MAX_COMPONENTS,
        eigenvectors.shape[1],
    )
    candidate_basis = eigenvectors[:, :candidate_count]
    scores = (candidate_basis.T @ centered).astype(np.float64)
    score_power = np.sum(np.square(scores), axis=1)
    temporal_denominator = centered.shape[1] * np.sum(
        np.power(scores, 4),
        axis=1,
    )
    temporal_participation = np.divide(
        np.square(score_power),
        temporal_denominator,
        out=np.zeros(candidate_count, dtype=np.float64),
        where=temporal_denominator > 0.0,
    )
    spatial_participation = np.divide(
        1.0,
        centered.shape[0] * np.sum(np.power(candidate_basis, 4), axis=0),
        out=np.zeros(candidate_count, dtype=np.float32),
        where=np.any(candidate_basis != 0.0, axis=0),
    )
    channel_power = np.sum(
        np.square(candidate_basis.reshape(4, num_bands, candidate_count)),
        axis=1,
    )
    channel_supported = np.all(
        channel_power >= JITTER_LOW_RANK_MIN_CHANNEL_POWER,
        axis=0,
    )
    accepted = (
        (eigenvalues[:candidate_count] >= JITTER_LOW_RANK_MIN_NOISE_RATIO * noise_floor)
        & (temporal_participation >= JITTER_LOW_RANK_MIN_TEMPORAL_PARTICIPATION)
        & (spatial_participation >= JITTER_LOW_RANK_MIN_SPATIAL_PARTICIPATION)
        & channel_supported
    )
    return (
        candidate_basis[:, accepted].astype(np.float32, copy=False),
        feature_means,
    )


def _project_jitter_low_rank(
    normalized_candidates: np.ndarray,
    valid: np.ndarray,
    basis: np.ndarray,
    feature_means: np.ndarray,
    *,
    num_bands: int,
) -> np.ndarray:
    """Project one column block onto the accepted recurrent sensor modes."""
    if basis.shape[1] == 0:
        return np.zeros_like(normalized_candidates, dtype=np.float32)
    winsorized = _winsorize_jitter_candidates(
        normalized_candidates,
        valid,
        num_bands=num_bands,
    )
    centered = np.where(
        valid,
        winsorized - feature_means[:, None],
        0.0,
    )
    numerator = basis.T @ centered
    denominator = np.square(basis).T @ valid.astype(np.float32)
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-6,
    )
    return basis @ scores


def _sample_jitter_band_columns(
    values: np.ndarray,
    positions: np.ndarray,
    *,
    start_column: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly sample band traces at fractional global column positions."""
    values = np.asarray(values, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or positions.ndim != 1:
        raise ValueError("band traces must be 2-D and positions must be 1-D")

    local = positions - float(start_column)
    lower = np.floor(local).astype(np.int64)
    valid = np.isfinite(local) & (lower >= 0) & (lower + 1 < values.shape[1])
    safe_lower = np.clip(lower, 0, max(values.shape[1] - 2, 0))
    fraction = (local - lower)[None, :]
    if values.shape[1] < 2:
        return (
            np.zeros((values.shape[0], positions.size), dtype=np.float32),
            np.zeros(positions.size, dtype=bool),
        )
    sampled = (
        values[:, safe_lower] * (1.0 - fraction) + values[:, safe_lower + 1] * fraction
    )
    sampled[:, ~valid] = 0.0
    return sampled.astype(np.float32, copy=False), valid


def _sample_centered_jitter_gradient(
    values: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``np.gradient(values, axis=1)`` without materializing it."""
    values = np.asarray(values, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or positions.ndim != 1:
        raise ValueError("band traces must be 2-D and positions must be 1-D")
    if values.shape[1] < 3:
        return (
            np.zeros((values.shape[0], positions.size), dtype=np.float32),
            np.zeros(positions.size, dtype=bool),
        )

    lower = np.floor(positions).astype(np.int64)
    valid = np.isfinite(positions) & (lower >= 1) & (lower + 2 < values.shape[1])
    safe_lower = np.clip(lower, 1, values.shape[1] - 3)
    gradient_lower = 0.5 * (values[:, safe_lower + 1] - values[:, safe_lower - 1])
    gradient_upper = 0.5 * (values[:, safe_lower + 2] - values[:, safe_lower])
    fraction = (positions - lower)[None, :]
    sampled = gradient_lower + (gradient_upper - gradient_lower) * fraction
    sampled[:, ~valid] = 0.0
    return sampled.astype(np.float32, copy=False), valid


def _dilate_jitter_edges(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a band/column mask horizontally without coupling sensor rows."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or mask.size == 0:
        return mask
    kernel = np.ones((1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) != 0


def _feather_jitter_scene_protection(
    mask: np.ndarray,
    *,
    block_columns: int = JITTER_BLOCK_COLUMNS,
) -> np.ndarray:
    """Turn strong aligned edges into a smooth correction-limit confidence."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return mask.astype(np.float16)
    if block_columns < 1:
        raise ValueError("block_columns must be positive")

    output = np.empty(mask.shape, dtype=np.float16)
    # OpenCV's automatic Gaussian kernel is narrower than four sigma. Retain a
    # slightly larger halo so block cores are identical to one whole-image
    # GaussianBlur without its image-wide float32 input and output arrays.
    halo = max(
        1,
        int(math.ceil(4.0 * JITTER_SCENE_PROTECTION_FEATHER_COLUMNS)),
    )
    for core_start in range(0, mask.shape[1], block_columns):
        core_stop = min(mask.shape[1], core_start + block_columns)
        read_start = max(0, core_start - halo)
        read_stop = min(mask.shape[1], core_stop + halo)
        protection = cv2.GaussianBlur(
            mask[:, read_start:read_stop].astype(np.float32),
            (0, 0),
            sigmaX=JITTER_SCENE_PROTECTION_FEATHER_COLUMNS,
            sigmaY=JITTER_SCENE_PROTECTION_FEATHER_BANDS,
            borderType=cv2.BORDER_REPLICATE,
        )
        np.clip(protection, 0.0, 1.0, out=protection)
        output[:, core_start:core_stop] = protection[
            :,
            core_start - read_start : core_stop - read_start,
        ].astype(np.float16)
    return output


def refine_column_exposure_jitter_correction(
    recurrent_model: ColumnJitterModel,
    red_view,
    green1,
    green2,
    blue_view,
    green_displacements: np.ndarray,
    *,
    fine_temporal_window: int = JITTER_FINE_TEMPORAL_WINDOW,
    block_columns: int = JITTER_BLOCK_COLUMNS,
    show_progress: bool = True,
) -> ColumnJitterModel:
    """
    Add the motion-aligned fine residual to an existing recurrent model.

    The recurrent radiometric model is independent of geometry and can
    therefore correct the two green planes before speed fitting. Once that
    corrected speed fit supplies their displacement, this function adds the
    fine residual and scene-protection confidence used by skew fitting and
    final rendering without re-estimating the recurrent component.
    """
    channel_views = (red_view, green1, green2, blue_view)
    heights = {view.shape[0] for view in channel_views}
    widths = {view.shape[1] for view in channel_views}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("all four jitter channels must have the same dimensions")
    image_height = heights.pop()
    image_width = widths.pop()
    if (
        recurrent_model.image_height != image_height
        or recurrent_model.width != image_width
    ):
        raise ValueError("recurrent jitter model does not match the channel views")
    if fine_temporal_window < 3 or fine_temporal_window % 2 != 1:
        raise ValueError("fine_temporal_window must be an odd integer of at least 3")
    if block_columns < 1:
        raise ValueError("block_columns must be positive")

    green_displacements = np.asarray(
        green_displacements,
        dtype=np.float64,
    )
    if green_displacements.shape != (image_width,):
        raise ValueError("green_displacements must contain one value per image column")
    if not np.all(np.isfinite(green_displacements)):
        raise ValueError("green_displacements must all be finite")
    if image_width < 4:
        return recurrent_model

    band_corrections = recurrent_model.band_corrections
    black_levels = recurrent_model.black_levels
    white_levels = recurrent_model.white_levels
    num_bands = recurrent_model.num_bands
    alignment_margin = int(math.ceil(float(np.max(np.abs(green_displacements))))) + 2

    # Estimate robust aligned-green gradient scales using the same raw-plus-
    # model operation order as the original one-call implementation. Keeping
    # this pass separate lets the recurrent model feed geometry first without
    # changing even threshold-adjacent fine-mask decisions during rendering.
    gradient_sample_count = min(
        image_width - 2,
        JITTER_LOW_RANK_MAX_SAMPLE_COLUMNS,
    )
    gradient_sample_columns = np.unique(
        np.linspace(
            1,
            image_width - 2,
            gradient_sample_count,
            dtype=np.int64,
        )
    )
    raw_green_gradient_samples = np.full(
        (2, num_bands, gradient_sample_columns.size),
        np.nan,
        dtype=np.float32,
    )
    gradient_read_radius = alignment_margin + 2
    green_views = (
        (JITTER_GREEN1, green1),
        (JITTER_GREEN2, green2),
    )
    for core_start in progress(
        range(0, image_width, block_columns),
        total=math.ceil(image_width / block_columns),
        desc="aligned green gradient scale",
        show=show_progress,
    ):
        core_stop = min(image_width, core_start + block_columns)
        sample_start = int(
            np.searchsorted(
                gradient_sample_columns,
                core_start,
                side="left",
            )
        )
        sample_stop = int(
            np.searchsorted(
                gradient_sample_columns,
                core_stop,
                side="left",
            )
        )
        if sample_stop <= sample_start:
            continue
        read_start = max(0, core_start - gradient_read_radius)
        read_stop = min(image_width, core_stop + gradient_read_radius)
        sample_x = gradient_sample_columns[sample_start:sample_stop]
        for green_index, (channel, view) in enumerate(green_views):
            statistics, _ = _jitter_band_statistics(
                view[:, slice(read_start, read_stop)],
                num_bands,
                float(black_levels[channel]),
                float(white_levels[channel]),
            )
            sample_positions = sample_x.astype(np.float64)
            if channel == JITTER_GREEN2:
                sample_positions += green_displacements[sample_x]
            sampled_gradient, sampled_valid = _sample_jitter_band_columns(
                np.gradient(statistics, axis=1),
                sample_positions,
                start_column=read_start,
            )
            sampled_gradient[:, ~sampled_valid] = np.nan
            raw_green_gradient_samples[
                green_index,
                :,
                sample_start:sample_stop,
            ] = sampled_gradient

    sample_x = gradient_sample_columns.astype(np.float64)
    aligned_sample_x = sample_x + green_displacements[gradient_sample_columns]
    model_green1_gradient, model_green1_valid = _sample_centered_jitter_gradient(
        band_corrections[JITTER_GREEN1],
        sample_x,
    )
    model_green2_gradient, model_green2_valid = _sample_centered_jitter_gradient(
        band_corrections[JITTER_GREEN2],
        aligned_sample_x,
    )
    corrected_green_gradient_samples = raw_green_gradient_samples.copy()
    corrected_green_gradient_samples[0] += model_green1_gradient
    corrected_green_gradient_samples[1] += model_green2_gradient
    corrected_green_gradient_samples[0, :, ~model_green1_valid] = np.nan
    corrected_green_gradient_samples[1, :, ~model_green2_valid] = np.nan
    green_gradient_scales = 1.4826 * bn.nanmedian(
        np.abs(corrected_green_gradient_samples),
        axis=2,
    )
    green_gradient_scales = np.maximum(
        green_gradient_scales,
        JITTER_ALIGNED_MIN_GRADIENT_SCALE,
    )
    green_gradient_scales[~np.isfinite(green_gradient_scales)] = np.inf

    # The recurrent model deliberately leaves ambiguous high-rank residuals
    # alone. Revisit only that narrow temporal band after the corrected speed
    # fit has aligned the greens well enough to separate scene edges from
    # capture-column errors.
    fine_band_corrections = np.zeros_like(band_corrections)
    scene_protection = np.zeros(
        (num_bands, image_width),
        dtype=bool,
    )
    fine_radius = fine_temporal_window // 2
    protection_radius = max(
        fine_radius,
        JITTER_SCENE_PROTECTION_DILATE_COLUMNS,
    )
    fine_read_radius = protection_radius + alignment_margin
    for core_start in progress(
        range(0, image_width, block_columns),
        total=math.ceil(image_width / block_columns),
        desc="fine residual jitter",
        show=show_progress,
    ):
        core_stop = min(image_width, core_start + block_columns)
        read_start = max(0, core_start - fine_read_radius)
        read_stop = min(image_width, core_stop + fine_read_radius)
        core_slice = slice(
            core_start - read_start,
            core_stop - read_start,
        )
        corrected_statistics = np.empty(
            (4, num_bands, read_stop - read_start),
            dtype=np.float32,
        )
        band_valid = np.empty_like(
            corrected_statistics,
            dtype=bool,
        )
        for channel, view in enumerate(channel_views):
            statistics, valid_statistics = _jitter_band_statistics(
                view[:, slice(read_start, read_stop)],
                num_bands,
                float(black_levels[channel]),
                float(white_levels[channel]),
            )
            corrected_statistics[channel] = statistics + band_corrections[
                channel,
                :,
                read_start:read_stop,
            ].astype(np.float32)
            band_valid[channel] = valid_statistics

        fine_target = _centered_moving_median(
            corrected_statistics,
            fine_temporal_window,
        )
        fine_offsets = (fine_target - corrected_statistics)[:, :, core_slice]

        # Compute scene-edge support on an expanded interval so horizontal
        # dilation is identical on either side of a processing boundary.
        mask_start = max(0, core_start - protection_radius)
        mask_stop = min(image_width, core_stop + protection_radius)
        mask_columns = np.arange(
            mask_start,
            mask_stop,
            dtype=np.int64,
        )
        green1_gradient = np.gradient(
            corrected_statistics[JITTER_GREEN1],
            axis=1,
        )
        green2_gradient = np.gradient(
            corrected_statistics[JITTER_GREEN2],
            axis=1,
        )
        sampled_green1, green1_valid = _sample_jitter_band_columns(
            green1_gradient,
            mask_columns,
            start_column=read_start,
        )
        aligned_columns = (
            mask_columns.astype(np.float64) + green_displacements[mask_columns]
        )
        sampled_green2, green2_valid = _sample_jitter_band_columns(
            green2_gradient,
            aligned_columns,
            start_column=read_start,
        )
        aligned_valid = green1_valid & green2_valid
        edge_score = np.sign(sampled_green1 * sampled_green2) * np.minimum(
            np.abs(sampled_green1) / green_gradient_scales[0, :, None],
            np.abs(sampled_green2) / green_gradient_scales[1, :, None],
        )
        weak_edges = _dilate_jitter_edges(
            edge_score >= JITTER_ALIGNED_EDGE_THRESHOLD,
            JITTER_ALIGNED_EDGE_DILATE_COLUMNS,
        )
        strong_edge_seed = edge_score >= JITTER_ALIGNED_STRONG_EDGE_THRESHOLD
        strong_edges = _dilate_jitter_edges(
            strong_edge_seed,
            fine_radius,
        )
        protected_scene = _dilate_jitter_edges(
            strong_edge_seed,
            JITTER_SCENE_PROTECTION_DILATE_COLUMNS,
        )
        structured = weak_edges | strong_edges
        structured[:, ~aligned_valid] = True
        mask_core_slice = slice(
            core_start - mask_start,
            core_stop - mask_start,
        )
        structured = structured[:, mask_core_slice]
        scene_protection[:, core_start:core_stop] = protected_scene[
            :,
            mask_core_slice,
        ]

        normalized_magnitude = (
            np.abs(fine_offsets) / (white_levels - black_levels)[:, None, None]
        )
        peak_magnitude = np.max(normalized_magnitude, axis=0)
        channel_threshold = np.maximum(
            JITTER_RESIDUAL_MIN_CHANNEL_FRACTION,
            (JITTER_RESIDUAL_MIN_RELATIVE_CHANNEL_POWER * peak_magnitude),
        )
        participating_channels = np.sum(
            normalized_magnitude >= channel_threshold[None, :, :],
            axis=0,
        )
        residual_event = participating_channels >= 2
        supported = (
            band_valid[:, :, core_slice]
            & ~structured[None, :, :]
            & residual_event[None, :, :]
        )
        maximum = (
            JITTER_RESIDUAL_MAX_CORRECTION_FRACTION
            * (white_levels - black_levels)[:, None, None]
        )
        residual_correction = np.clip(
            fine_offsets,
            -maximum,
            maximum,
        )
        fine_band_corrections[:, :, core_start:core_stop] = np.where(
            supported,
            residual_correction,
            0.0,
        ).astype(np.float16)

    scene_protection = _feather_jitter_scene_protection(
        scene_protection,
        block_columns=block_columns,
    )
    return ColumnJitterModel(
        band_corrections,
        image_height,
        black_levels,
        white_levels,
        recurrent_model.confidence,
        fine_band_corrections=fine_band_corrections,
        scene_protection=scene_protection,
    )


def column_exposure_jitter_correction(
    red_view,
    green1,
    green2,
    blue_view,
    g: Path | None = None,
    *,
    black_levels: np.ndarray | None = None,
    white_levels: np.ndarray | None = None,
    num_bands: int = JITTER_VERTICAL_BANDS,
    fine_temporal_window: int = JITTER_FINE_TEMPORAL_WINDOW,
    temporal_window: int = JITTER_TEMPORAL_WINDOW,
    block_columns: int = JITTER_BLOCK_COLUMNS,
    structure_mask: np.ndarray | None = None,
    structure_mask_scale: int = 1,
    green_displacements: np.ndarray | None = None,
    show_progress: bool = True,
) -> ColumnJitterModel:
    """
    Estimate absolute, channel/row-specific exposure corrections.

    A centered temporal median supplies the local unstriped target. Unlike the
    former adjacent-column composition, this has no cumulative state: monotonic
    scene gradients are fixed points and cannot turn into correction drift.
    A short centered median detects narrow line jitter and a longer median
    detects broader exposure cycles. Those per-band values are candidates only.
    The applied model is a small set of recurrent low-rank modes shared across
    the sensor. A wheel, window edge, bellows pleat, or one-channel color detail
    is sparse in time or row/channel support and cannot become a correction
    mode. Unlike local temporal smoothing, the model therefore does not erase
    the very detail that resembles a stripe.

    Strong scene structure is excluded by a downsampled edge mask. Clipped and
    structured bands do not vote when the model is learned or projected. Once
    a sensor-wide mode is established, its row/channel response is applied
    consistently through those regions, with per-pixel clipping protection.

    When the motion fit supplies the displacement between the two green
    columns, a second pass removes the fine residual left by the recurrent
    model. It operates only where both greens agree that there is no scene
    edge. Strong corroborated edges protect their full temporal-median
    neighborhood, and an event must appear in at least two Bayer channels.
    This prevents a wheel, bellows pleat, grill, or one-channel color feature
    from being replaced by a locally smoothed estimate. The recurrent and fine
    components retain separate support masks during row interpolation. The
    same aligned scene evidence limits the recurrent additive correction
    relative to local shadow signal, while flat regions still receive the full
    correction.
    """
    del g  # Retained in the signature for callers that group outputs by capture.
    channel_views = (red_view, green1, green2, blue_view)
    heights = {view.shape[0] for view in channel_views}
    widths = {view.shape[1] for view in channel_views}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("all four jitter channels must have the same dimensions")
    image_height = heights.pop()
    image_width = widths.pop()
    if image_height % num_bands:
        raise ValueError(
            f"image height {image_height} must be divisible by {num_bands}"
        )
    if fine_temporal_window < 3 or fine_temporal_window % 2 != 1:
        raise ValueError("fine_temporal_window must be an odd integer of at least 3")
    if temporal_window < fine_temporal_window or temporal_window % 2 != 1:
        raise ValueError(
            "temporal_window must be an odd integer no smaller than "
            "fine_temporal_window"
        )
    if block_columns < 1:
        raise ValueError("block_columns must be positive")
    if green_displacements is not None:
        green_displacements = np.asarray(
            green_displacements,
            dtype=np.float64,
        )
        if green_displacements.shape != (image_width,):
            raise ValueError(
                "green_displacements must contain one value per image column"
            )
        if not np.all(np.isfinite(green_displacements)):
            raise ValueError("green_displacements must all be finite")

    if black_levels is None or white_levels is None:
        estimated_black, estimated_white = _estimate_jitter_levels(channel_views)
        if black_levels is None:
            black_levels = estimated_black
        if white_levels is None:
            white_levels = estimated_white
    black_levels = np.asarray(black_levels, dtype=np.float32)
    white_levels = np.asarray(white_levels, dtype=np.float32)
    if black_levels.shape != (4,) or white_levels.shape != (4,):
        raise ValueError("black_levels and white_levels must each contain 4 values")
    white_levels = np.maximum(white_levels, black_levels + 1.0)

    structure_bands = None
    if structure_mask is not None:
        structure_bands = _jitter_structure_bands(
            structure_mask,
            image_height=image_height,
            image_width=image_width,
            num_bands=num_bands,
            scale=structure_mask_scale,
        )

    band_corrections = np.empty(
        (4, num_bands, image_width),
        dtype=np.float16,
    )
    radius = temporal_window // 2
    block_starts = range(0, image_width, block_columns)

    for core_start in progress(
        block_starts,
        total=math.ceil(image_width / block_columns),
        desc="column exposure jitter",
        show=show_progress,
    ):
        core_stop = min(image_width, core_start + block_columns)
        read_start = max(0, core_start - radius)
        read_stop = min(image_width, core_stop + radius)
        core_slice = slice(core_start - read_start, core_stop - read_start)
        core_width = core_stop - core_start
        fine_offsets = np.empty((4, num_bands, core_width), dtype=np.float32)
        broad_offsets = np.empty((4, num_bands, core_width), dtype=np.float32)
        valid = np.empty((4, num_bands, core_width), dtype=bool)

        for channel, view in enumerate(channel_views):
            data = view[:, slice(read_start, read_stop)]
            statistics, band_valid = _jitter_band_statistics(
                data,
                num_bands,
                float(black_levels[channel]),
                float(white_levels[channel]),
            )
            fine_target = _centered_moving_median(
                statistics,
                fine_temporal_window,
            )
            broad_target = _centered_moving_median(
                statistics,
                temporal_window,
            )
            fine_offsets[channel] = (fine_target - statistics)[:, core_slice]
            broad_offsets[channel] = (broad_target - fine_target)[:, core_slice]
            valid[channel] = band_valid[:, core_slice]

        if structure_bands is not None:
            lowres_columns = (
                np.arange(core_start, core_stop, dtype=np.int64) // structure_mask_scale
            )
            structured = structure_bands[:, lowres_columns]
            valid &= ~structured[None, :, :]

        spans = (white_levels - black_levels)[:, None, None]
        fine_supported = valid & (
            np.abs(fine_offsets) <= JITTER_FINE_MAX_CORRECTION_FRACTION * spans
        )
        broad_supported = valid & (
            np.abs(broad_offsets) <= JITTER_BROAD_MAX_CORRECTION_FRACTION * spans
        )
        candidate_corrections = np.where(
            fine_supported,
            fine_offsets,
            0.0,
        ) + np.where(
            broad_supported,
            broad_offsets,
            0.0,
        )

        normalized_candidates = np.clip(
            candidate_corrections / spans,
            -JITTER_MODEL_MAX_CORRECTION_FRACTION,
            JITTER_MODEL_MAX_CORRECTION_FRACTION,
        )
        # NaN is an intentional temporary sentinel: a valid zero residual must
        # remain distinguishable from an unsupported clipped/structured band.
        band_corrections[:, :, core_start:core_stop] = np.clip(
            np.where(valid, normalized_candidates, np.nan),
            -np.finfo(np.float16).max,
            np.finfo(np.float16).max,
        )

    feature_candidates = band_corrections.reshape(4 * num_bands, image_width)
    sample_count = min(image_width, JITTER_LOW_RANK_MAX_SAMPLE_COLUMNS)
    sample_columns = np.linspace(
        0,
        image_width - 1,
        sample_count,
        dtype=np.int64,
    )
    training_candidates = feature_candidates[:, sample_columns].astype(np.float32)
    training_valid = np.isfinite(training_candidates)
    training_candidates[~training_valid] = 0.0
    basis, feature_means = _jitter_low_rank_basis(
        training_candidates,
        training_valid,
        num_bands=num_bands,
    )
    if show_progress:
        print(
            f"column exposure jitter: accepted {basis.shape[1]} recurrent sensor modes"
        )

    confidence_full = np.zeros(image_width, dtype=np.float32)
    minimum_valid = math.ceil(JITTER_MODEL_MIN_VALID_FRACTION * 4 * num_bands)
    spans_flat = np.repeat(white_levels - black_levels, num_bands)
    for core_start in range(0, image_width, block_columns):
        core_stop = min(image_width, core_start + block_columns)
        candidates = feature_candidates[:, core_start:core_stop].astype(np.float32)
        valid = np.isfinite(candidates)
        candidates[~valid] = 0.0
        reconstructed = _project_jitter_low_rank(
            candidates,
            valid,
            basis,
            feature_means,
            num_bands=num_bands,
        )
        valid_count = np.sum(valid, axis=0)
        supported_columns = valid_count >= minimum_valid
        reconstructed[:, ~supported_columns] = 0.0
        reconstructed = np.clip(
            reconstructed,
            -JITTER_MODEL_MAX_CORRECTION_FRACTION,
            JITTER_MODEL_MAX_CORRECTION_FRACTION,
        )
        feature_candidates[:, core_start:core_stop] = np.clip(
            reconstructed * spans_flat[:, None],
            -np.finfo(np.float16).max,
            np.finfo(np.float16).max,
        )
        confidence_full[core_start:core_stop] = np.where(
            supported_columns,
            valid_count / float(4 * num_bands),
            0.0,
        )

    recurrent_model = ColumnJitterModel(
        band_corrections,
        image_height,
        black_levels,
        white_levels,
        confidence_full,
    )
    if green_displacements is None:
        return recurrent_model
    return refine_column_exposure_jitter_correction(
        recurrent_model,
        red_view,
        green1,
        green2,
        blue_view,
        green_displacements,
        fine_temporal_window=fine_temporal_window,
        block_columns=block_columns,
        show_progress=show_progress,
    )


def interpolate_upsample(arr, off):
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.empty(0)
    res = np.zeros(2 * n)
    if off == 0:
        # Place original at even indices
        res[::2] = arr
        # Interpolate inner odd indices
        if n > 1:
            res[1:-1:2] = (res[:-2:2] + res[2::2]) / 2
        # Repeat at the end for the last odd index
        res[-1] = res[-2]
    elif off == 1:
        # Place original at odd indices
        res[1::2] = arr
        # Interpolate inner even indices (starting from 2)
        if n > 1:
            res[2::2] = (res[1:-1:2] + res[3::2]) / 2
        # Repeat at the start for index 0
        res[0] = res[1]
    return res


def generate_sample_positions(spline, sample_xs, width):
    sample_positions = []
    widths = []
    steps = []
    x = 0
    initial_step = float(spline(sample_xs[0]))
    if not math.isfinite(initial_step) or initial_step == 0.0:
        raise ValueError("speed spline must have a finite non-zero direction")
    direction = math.copysign(1.0, initial_step)
    if direction < 0.0:
        x = width - 1
    while 0 <= x < width:
        sx = float(spline(x))
        if not math.isfinite(sx) or sx * direction <= 0.0:
            raise ValueError("speed spline changed direction or became invalid")
        if abs(sx) < 0.1:
            sx = direction * 0.1
        next_x = x + sx
        if next_x == x:
            raise ValueError("speed spline step is too small for floating-point progress")

        sample_positions.append(x)
        widths.append(abs(sx))
        steps.append(sx)
        x = next_x

    # This is the reconstruction aperture, not a speed clamp: upsampling still
    # needs at least one input-pixel support radius for stable interpolation.
    widths = np.clip(widths, 1, width)
    return sample_positions, widths, steps


def truncate_sample_positions_for_skew(
    sample_positions,
    widths,
    *,
    skew: float,
    step_scales=None,
    row_offsets: np.ndarray,
    width: int,
    return_slice: bool = False,
):
    if not sample_positions:
        result = (sample_positions, widths)
        return (*result, slice(0, 0)) if return_slice else result
    positions = np.asarray(sample_positions, dtype=np.float32)
    widths_arr = np.asarray(widths, dtype=np.float32)
    if step_scales is None:
        step_scales = np.ones_like(positions)
    step_scales = np.asarray(step_scales, dtype=np.float32)
    if step_scales.shape != positions.shape:
        raise ValueError("step_scales must match sample_positions")
    # The skew offset is linear in the row coordinate, so its extrema are
    # always at the two sensor edges. Materializing every output-column/row
    # combination used 7.6 GiB per million columns at the native 2048 rows.
    scaled_skew = np.float32(skew) * step_scales
    first_edge = scaled_skew * np.float32(np.min(row_offsets))
    second_edge = scaled_skew * np.float32(np.max(row_offsets))
    min_offset = np.minimum(first_edge, second_edge)
    max_offset = np.maximum(first_edge, second_edge)

    min_center = positions + min_offset - widths_arr
    max_center = positions + max_offset + widths_arr
    valid = (min_center >= 0.0) & (max_center <= (width - 1))
    if not np.any(valid):
        result = ([], [])
        return (*result, slice(0, 0)) if return_slice else result

    first = int(np.argmax(valid))
    last = int(len(valid) - 1 - np.argmax(valid[::-1]))
    kept = slice(first, last + 1)
    result = (sample_positions[kept], widths[kept])
    return (*result, kept) if return_slice else result


def build_cut_aligned_output_ranges(
    sample_positions,
    cut_positions,
) -> list[tuple[int, int]]:
    """Map ordered raw cut coordinates to output-sample index ranges.

    Ranges are returned in sampling order, which may be right-to-left when the
    correlation pass detected negative motion.  The existing output rotation
    logic subsequently maps those ranges back to stable left-to-right names.
    """
    positions = np.asarray(sample_positions, dtype=np.float64)
    cuts = np.asarray(cut_positions, dtype=np.float64)
    if positions.ndim != 1 or len(positions) < 2:
        raise ValueError("at least two sample positions are required")
    if cuts.ndim != 1 or len(cuts) < 2:
        raise ValueError("at least two cut positions are required")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cuts)):
        raise ValueError("sample and cut positions must be finite")
    if np.any(np.diff(cuts) <= 0.0):
        raise ValueError("cut positions must be strictly increasing")

    position_deltas = np.diff(positions)
    increasing = bool(positions[-1] > positions[0])
    if increasing:
        if np.any(position_deltas <= 0.0):
            raise ValueError("sample positions must be strictly monotonic")
        oriented_positions = positions
        oriented_cuts = cuts
    else:
        if np.any(position_deltas >= 0.0):
            raise ValueError("sample positions must be strictly monotonic")
        oriented_positions = -positions
        oriented_cuts = -cuts[::-1]

    # Skew-safe sampling can trim a few raw columns from either sensor edge.
    # searchsorted intentionally clamps an outer start/end annotation to the
    # first/last renderable sample; interior cuts still have to produce unique
    # boundaries below.
    boundaries = np.searchsorted(oriented_positions, oriented_cuts, side="left")
    boundaries = np.clip(boundaries, 0, len(positions))
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("two cut positions map to the same output sample")
    return [
        (int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]


def build_equal_output_chunk_ranges(
    output_range: tuple[int, int],
    *,
    chunk_target: int = OUTPUT_CHUNK_TARGET,
) -> list[tuple[int, int]]:
    """Split an output index range evenly, keeping every chunk below target."""
    start, stop = (int(value) for value in output_range)
    if start < 0 or stop <= start:
        raise ValueError("output range must be non-empty and non-negative")
    if chunk_target <= 1:
        raise ValueError("chunk_target must be greater than one")

    length = stop - start
    chunk_count = max(1, math.ceil(length / (chunk_target - 1)))
    base_width, wider_chunk_count = divmod(length, chunk_count)
    widths = [base_width + 1] * wider_chunk_count + [base_width] * (
        chunk_count - wider_chunk_count
    )

    ranges = []
    cursor = start
    for width in widths:
        ranges.append((cursor, cursor + width))
        cursor += width
    return ranges


def build_padded_cut_output_ranges(
    chunk_ranges: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """Extend interior cut slices into their neighbors to match the widest."""
    widths = [stop - start for start, stop in chunk_ranges]
    if not widths or min(widths) < 1:
        raise ValueError("cut output ranges must be non-empty")
    if any(
        left[1] != right[0]
        for left, right in zip(chunk_ranges, chunk_ranges[1:])
    ):
        raise ValueError("cut output ranges must be contiguous")

    widest = max(widths)
    padded = list(chunk_ranges)
    for index in range(1, len(padded) - 1):
        start, stop = padded[index]
        left = (widest - widths[index]) // 2
        right = widest - widths[index] - left
        if left > widths[index - 1] or right > widths[index + 1]:
            raise ValueError("adjacent cut slices are too narrow for centered padding")
        padded[index] = (start - left, stop + right)
    return padded, widest


def pad_random_edge_columns(
    data: np.ndarray,
    output_width: int,
    *,
    prepend: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pad one outer slice with random columns from its outward 128 columns."""
    missing = output_width - data.shape[1]
    if missing < 0 or data.shape[1] < 1:
        raise ValueError("output width must not be narrower than the input")
    if not missing:
        return data

    padded = np.empty((data.shape[0], output_width, *data.shape[2:]), data.dtype)
    data_start = missing if prepend else 0
    padded[:, data_start : data_start + data.shape[1]] = data
    edge = (
        data[:, :EDGE_PADDING_SAMPLE_COLUMNS]
        if prepend
        else data[:, -EDGE_PADDING_SAMPLE_COLUMNS:]
    )
    padding_start = 0 if prepend else data.shape[1]
    for start in range(0, missing, EDGE_PADDING_SAMPLE_COLUMNS):
        count = min(EDGE_PADDING_SAMPLE_COLUMNS, missing - start)
        padded[:, padding_start + start : padding_start + start + count] = edge[
            :, rng.integers(edge.shape[1], size=count)
        ]
    return padded


def sample_motion_corrected_chunks(
    red_view,
    green1,
    green2,
    blue_view,
    sample_positions,
    widths,
    *,
    stride_x: int = 1,
    stride_y: int = 1,
    skew: float = 0.0,
    max_pix: int = 16384,
    jitter_models=None,
    jitter_protect_scene: bool = True,
    step_scales=None,
    chunk_ranges=None,
    outer_padding_width: int | None = None,
    show_progress: bool = True,
):
    if stride_x < 1 or stride_y < 1 or max_pix < 1:
        raise ValueError("sampling strides and chunk size must be positive")
    indices = list(range(0, len(sample_positions), stride_x))
    if not indices:
        return
    if step_scales is None:
        step_scales = np.ones(len(sample_positions), dtype=np.float32)
    if len(step_scales) != len(sample_positions):
        raise ValueError("step_scales must match sample_positions")
    if outer_padding_width is not None and outer_padding_width < 1:
        raise ValueError("outer padding width must be positive")
    padding_rng = np.random.default_rng() if outer_padding_width is not None else None

    if chunk_ranges is None:
        batches = [
            indices[start : start + max_pix]
            for start in range(0, len(indices), max_pix)
        ]
    else:
        if stride_x != 1:
            raise ValueError("explicit chunk ranges require stride_x=1")
        batches = []
        previous_start = -1
        for range_index, chunk_range in enumerate(chunk_ranges):
            if len(chunk_range) != 2:
                raise ValueError(f"chunk_ranges[{range_index}] must be (start, stop)")
            start, stop = map(int, chunk_range)
            if not 0 <= start < stop <= len(sample_positions):
                raise ValueError(f"chunk_ranges[{range_index}] is out of bounds")
            if start < previous_start:
                raise ValueError("chunk ranges must be ordered")
            batches.append(list(range(start, stop)))
            previous_start = start

    out_height = green1.height * 2
    center_y = (out_height - 1) / 2.0
    row_offsets = (2 * np.arange(green1.height) - center_y).astype(np.float32)

    for chunk_i, batch_indices in enumerate(batches):
        pad_left = batch_indices[0] > 0
        if pad_left:
            previous_index = batch_indices[0] - (
                1 if chunk_ranges is not None else stride_x
            )
            use_indices = [previous_index] + batch_indices
        else:
            use_indices = list(batch_indices)

        out_width = len(use_indices)
        raw_sampled = np.zeros((2 * green1.height, out_width, 3), dtype=np.float32)

        for j, sample_idx in progress(
            enumerate(use_indices),
            total=len(use_indices),
            desc="sampling chunk",
            show=show_progress,
        ):
            xi = sample_positions[sample_idx]
            wi = widths[sample_idx]
            skew_scaled = skew * step_scales[sample_idx]

            centers = xi + skew_scaled * row_offsets
            window_half = int(math.ceil(wi))
            offsets = np.arange(-window_half, window_half + 1, dtype=np.int32)
            base = np.floor(centers).astype(np.int32)
            window_indices = base[:, None] + offsets[None, :]
            valid = (window_indices >= 0) & (window_indices < green1.width)
            indices_clipped = np.clip(window_indices, 0, green1.width - 1)

            start_col = int(indices_clipped.min())
            end_col = int(indices_clipped.max()) + 1
            if end_col <= start_col:
                continue

            r_win = red_view[:, slice(start_col, end_col)]
            g1_win = green1[:, slice(start_col, end_col)]
            g2_win = green2[:, slice(start_col, end_col)]
            b_win = blue_view[:, slice(start_col, end_col)]

            indices_local = indices_clipped - start_col
            indices_local = np.clip(indices_local, 0, r_win.shape[1] - 1)
            sample_rows = np.arange(green1.height, dtype=np.int32)
            row_idx = sample_rows[:, None]

            r_vals = r_win[row_idx, indices_local]
            g1_vals = g1_win[row_idx, indices_local]
            g2_vals = g2_win[row_idx, indices_local]
            b_vals = b_win[row_idx, indices_local]

            if jitter_models is not None:
                r_vals = jitter_models.correct_samples(
                    r_vals,
                    JITTER_RED,
                    indices_clipped,
                    sample_rows,
                    protect_scene=jitter_protect_scene,
                )
                g1_vals = jitter_models.correct_samples(
                    g1_vals,
                    JITTER_GREEN1,
                    indices_clipped,
                    sample_rows,
                    protect_scene=jitter_protect_scene,
                )
                g2_vals = jitter_models.correct_samples(
                    g2_vals,
                    JITTER_GREEN2,
                    indices_clipped,
                    sample_rows,
                    protect_scene=jitter_protect_scene,
                )
                b_vals = jitter_models.correct_samples(
                    b_vals,
                    JITTER_BLUE,
                    indices_clipped,
                    sample_rows,
                    protect_scene=jitter_protect_scene,
                )

            dist = window_indices.astype(np.float32) - centers[:, None].astype(
                np.float32
            )
            weights = np.zeros_like(dist, dtype=np.float32)
            mask = (np.abs(dist) < wi) & valid
            weights[mask] = 1 + np.cos(np.pi * dist[mask] / wi)
            sum_weight = np.sum(weights, axis=1)
            sum_weight[sum_weight == 0] = 1.0

            r_weighted = np.sum(r_vals * weights, axis=1) / sum_weight
            g1_weighted = np.sum(g1_vals * weights, axis=1) / sum_weight
            g2_weighted = np.sum(g2_vals * weights, axis=1) / sum_weight
            b_weighted = np.sum(b_vals * weights, axis=1) / sum_weight

            r_interp = interpolate_upsample(r_weighted, 0)
            g1_interp = interpolate_upsample(g1_weighted, 1)
            g2_interp = interpolate_upsample(g2_weighted, 0)
            b_interp = interpolate_upsample(b_weighted, 1)

            if j == 0:
                raw_sampled[:, 0, 0] = r_interp
                raw_sampled[:, 0, 1] += g1_interp * 0.5
            if j < len(use_indices) - 1:
                raw_sampled[:, j + 1, 0] = r_interp
                raw_sampled[:, j + 1, 1] += g1_interp * 0.5
            raw_sampled[:, j, 1] += g2_interp * 0.5
            raw_sampled[:, j, 2] = b_interp

        if pad_left:
            raw_sampled = raw_sampled[:, 1:, :]
        if stride_y > 1:
            raw_sampled = raw_sampled[::stride_y, :, :]
        if outer_padding_width is not None:
            if chunk_i == 0:
                raw_sampled = pad_random_edge_columns(
                    raw_sampled,
                    outer_padding_width,
                    prepend=True,
                    rng=padding_rng,
                )
            elif chunk_i == len(batches) - 1:
                raw_sampled = pad_random_edge_columns(
                    raw_sampled,
                    outer_padding_width,
                    prepend=False,
                    rng=padding_rng,
                )
            if raw_sampled.shape[1] != outer_padding_width:
                raise RuntimeError("equal-width padding produced an invalid width")

        yield chunk_i, raw_sampled
        # The consumer drops its reference before requesting the next chunk;
        # drop the generator's reference too before allocating its successor.
        del raw_sampled


def sample_skew_corrected_green_chunks(
    green1,
    green2,
    sample_positions,
    widths,
    *,
    stride_x: int = 1,
    stride_y: int = 1,
    max_pix: int = 16384,
    jitter_models=None,
    show_progress: bool = True,
):
    """
    Produce the speed-corrected green image used for zero-skew fitting.

    Skew estimation only consumes the green channel and samples at zero
    candidate skew. Correct each contiguous raw-coordinate green block once,
    then reuse those corrected pixels across the overlapping output apertures.
    This preserves the same radiometry-before-geometry operation order as the
    full Bayer sampler without reading or correcting red and blue.
    """
    if stride_x < 1 or stride_y < 1 or max_pix < 1:
        raise ValueError("sampling strides and chunk size must be positive")
    if green1.shape != green2.shape:
        raise ValueError("green channel views must have the same shape")
    if len(sample_positions) != len(widths):
        raise ValueError("sample_positions and widths must have equal length")

    indices = list(range(0, len(sample_positions), stride_x))
    if not indices:
        return

    image_height, image_width = green1.shape
    chunk_i = 0
    for start in range(0, len(indices), max_pix):
        batch_indices = indices[start : start + max_pix]
        pad_left = start > 0
        if pad_left:
            use_indices = [indices[start - 1]] + batch_indices
        else:
            use_indices = list(batch_indices)

        use_positions = np.asarray(
            [sample_positions[i] for i in use_indices],
            dtype=np.float64,
        )
        use_widths = np.asarray(
            [widths[i] for i in use_indices],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(use_positions))
            or not np.all(np.isfinite(use_widths))
            or np.any(use_widths <= 0.0)
        ):
            raise ValueError("sample positions and widths must be finite and positive")

        window_halves = np.ceil(use_widths).astype(np.int64)
        bases = np.floor(use_positions).astype(np.int64)
        left_columns = np.clip(
            bases - window_halves,
            0,
            image_width - 1,
        )
        right_columns = np.clip(
            bases + window_halves,
            0,
            image_width - 1,
        )
        read_start = int(np.min(left_columns))
        read_stop = int(np.max(right_columns)) + 1

        green1_block = green1[:, slice(read_start, read_stop)]
        green2_block = green2[:, slice(read_start, read_stop)]
        if jitter_models is not None:
            green1_block = apply_column_models(
                green1_block,
                jitter_models,
                read_start,
                JITTER_GREEN1,
                protect_scene=False,
            )
            green2_block = apply_column_models(
                green2_block,
                jitter_models,
                read_start,
                JITTER_GREEN2,
                protect_scene=False,
            )

        sampled_green = np.zeros(
            (2 * image_height, len(use_indices)),
            dtype=np.float32,
        )
        for j, sample_idx in progress(
            enumerate(use_indices),
            total=len(use_indices),
            desc="sampling skew chunk",
            show=show_progress,
        ):
            xi = sample_positions[sample_idx]
            wi = widths[sample_idx]
            window_half = int(math.ceil(wi))
            base = int(np.floor(xi))
            offsets = np.arange(
                -window_half,
                window_half + 1,
                dtype=np.int32,
            )
            window_indices = base + offsets
            valid = (window_indices >= 0) & (window_indices < image_width)
            indices_clipped = np.clip(window_indices, 0, image_width - 1)
            indices_local = indices_clipped - read_start

            green1_values = green1_block[:, indices_local]
            green2_values = green2_block[:, indices_local]

            dist = indices_clipped.astype(np.float32) - np.float32(xi)
            weights = np.zeros_like(dist, dtype=np.float32)
            mask = (np.abs(dist) < wi) & valid
            weights[mask] = 1 + np.cos(np.pi * dist[mask] / wi)
            sum_weight = np.sum(weights)
            if sum_weight == 0.0:
                sum_weight = np.float32(1.0)

            green1_weighted = (
                np.sum(green1_values * weights[None, :], axis=1) / sum_weight
            )
            green2_weighted = (
                np.sum(green2_values * weights[None, :], axis=1) / sum_weight
            )
            green1_interp = interpolate_upsample(green1_weighted, 1)
            green2_interp = interpolate_upsample(green2_weighted, 0)

            if j == 0:
                sampled_green[:, 0] += green1_interp * 0.5
            if j < len(use_indices) - 1:
                sampled_green[:, j + 1] += green1_interp * 0.5
            sampled_green[:, j] += green2_interp * 0.5

        if pad_left:
            sampled_green = sampled_green[:, 1:]
        if stride_y > 1:
            sampled_green = sampled_green[::stride_y, :]

        yield chunk_i, sampled_green
        chunk_i += 1


def _sparse_skew_histogram(
    edge_y: np.ndarray,
    edge_x: np.ndarray,
    edge_weights: np.ndarray,
    *,
    skew: float,
    hist_len: int,
    hist_offset: int,
) -> np.ndarray:
    """Accumulate one Hough projection using only nonzero edge weights."""
    shifts = np.rint(skew * edge_y).astype(np.int32)
    bin_indices = edge_x - shifts + hist_offset
    valid = (bin_indices >= 0) & (bin_indices < hist_len)
    return np.bincount(
        bin_indices[valid],
        weights=edge_weights[valid],
        minlength=hist_len,
    )


def estimate_skew_hough(
    chunk_iter,
    *,
    g: Path,
    stride_x: int,
    stride_y: int,
    skew_min: float = -0.05,
    skew_max: float = 0.05,
    bins: int = 69,
    energy_percentile: float = 98,
    write_debug: bool = False,
    show_progress: bool = True,
):
    skews = np.linspace(skew_min, skew_max, bins, dtype=np.float32)
    scores = np.zeros_like(skews, dtype=np.float64)

    def find_peaks_nms(values: np.ndarray, min_distance: int) -> np.ndarray:
        if values.size < 3:
            return np.empty((0,), dtype=np.int32)
        peaks = np.zeros_like(values, dtype=bool)
        peaks[1:-1] = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
        peak_idxs = np.where(peaks)[0]
        if peak_idxs.size == 0:
            return peak_idxs
        peak_vals = values[peak_idxs]
        order = np.argsort(peak_vals)[::-1]
        selected = []
        for idx in peak_idxs[order]:
            if all(abs(idx - s) >= min_distance for s in selected):
                selected.append(int(idx))
        return np.array(selected, dtype=np.int32)

    for _chunk_i, chunk in progress(
        chunk_iter,
        desc="estimating skew",
        show=show_progress,
    ):
        if chunk.size == 0:
            continue
        if chunk.ndim == 2:
            gray = np.asarray(chunk, dtype=np.float32)
        elif chunk.ndim == 3 and chunk.shape[2] >= 2:
            gray = chunk[:, :, 1].astype(np.float32)
        else:
            raise ValueError("skew chunks must be grayscale or RGB arrays")
        grad_y, grad_x = np.gradient(gray)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 0.1 * np.max(gray)
        energy = np.abs(grad_x) / magnitude

        thresh = np.percentile(energy, energy_percentile)
        mask = energy >= thresh
        if not np.any(mask):
            continue

        energy = np.sqrt(energy)

        weights = energy * mask
        width = chunk.shape[1]
        hist_len = width * 3
        hist_offset = width
        edge_y, edge_x = np.nonzero(mask)
        edge_weights = weights[edge_y, edge_x]
        scale = stride_y / stride_x
        total_weight = float(np.sum(weights)) + 1e-6

        chunk_scores = np.zeros_like(skews, dtype=np.float64)
        histograms = (
            np.zeros((len(skews), hist_len), dtype=np.float64) if write_debug else None
        )
        peak_values = [[] for _ in range(len(skews))]
        for i, skew in enumerate(skews):
            skew_ds = skew * scale
            hist = _sparse_skew_histogram(
                edge_y,
                edge_x,
                edge_weights,
                skew=skew_ds,
                hist_len=hist_len,
                hist_offset=hist_offset,
            )
            if hist.size:
                hist_norm = hist / total_weight
                if histograms is not None:
                    histograms[i, :] = hist_norm
                peak_idxs = find_peaks_nms(hist_norm, min_distance=12)
                if peak_idxs.size:
                    peak_values[i] = sorted(
                        (float(hist_norm[p]) for p in peak_idxs), reverse=True
                    )
                else:
                    peak_values[i] = [float(np.max(hist_norm))]

        peak_max = np.array(
            [vals[0] if vals else 0.0 for vals in peak_values], dtype=np.float64
        )
        top_s = min(5, len(skews))
        top_indices = np.argsort(peak_max)[-top_s:]
        k = None
        for idx in top_indices:
            n_peaks = len(peak_values[idx])
            if n_peaks == 0:
                continue
            k = n_peaks if k is None else min(k, n_peaks)
        if k is None or k == 0:
            k = 1

        for i in range(len(skews)):
            vals = peak_values[i]
            if not vals:
                continue
            chunk_scores[i] = float(np.sum(vals[:k]))

        if write_debug:
            best_idx = int(np.argmax(chunk_scores))
            best_skew = skews[best_idx]
            skew_ds = best_skew * scale
            best_hist = histograms[best_idx]
            best_offset = hist_offset

            energy_norm = energy / max(np.max(energy), 1e-6)
            mask_u8 = mask.astype(np.uint8) * 255
            energy_u8 = (np.clip(energy_norm, 0, 1) * 255).astype(np.uint8)

            energy_vis = cv2.cvtColor(energy_u8, cv2.COLOR_GRAY2BGR)
            energy_vis = np.ascontiguousarray(energy_vis[::-1, :, :])
            mask_vis = np.ascontiguousarray(mask_u8[::-1, :])

            peak_thresh = 0.7 * np.max(best_hist) if best_hist.size else 0
            if peak_thresh > 0:
                peaks = np.zeros_like(best_hist, dtype=bool)
                peaks[1:-1] = (best_hist[1:-1] > best_hist[:-2]) & (
                    best_hist[1:-1] >= best_hist[2:]
                )
                peaks &= best_hist >= peak_thresh
                peak_idxs = np.where(peaks)[0]
            else:
                peak_idxs = []

            h = energy_vis.shape[0]
            for b in peak_idxs:
                intercept = b - best_offset
                x0 = int(round(intercept + skew_ds * (h - 1)))
                x1 = int(round(intercept))
                y0 = 0
                y1 = h - 1
                if (0 <= x0 < energy_vis.shape[1]) or (0 <= x1 < energy_vis.shape[1]):
                    cv2.line(energy_vis, (x0, y0), (x1, y1), (0, 0, 255), 1)

            cv2.imwrite(str(g / f"skew_energy_{_chunk_i:02d}.png"), energy_vis)
            cv2.imwrite(str(g / f"skew_mask_{_chunk_i:02d}.png"), mask_vis)

            fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
            ax.imshow(
                histograms,
                cmap="gray",
                aspect="auto",
                origin="lower",
                extent=[-hist_offset, hist_len - hist_offset, skews[0], skews[-1]],
            )
            ax.axhline(best_skew, color="red", linewidth=1)
            ax.set_ylabel("skew slope")
            ax.set_xlabel("intercept")
            fig.savefig(g / f"skew_histogram{_chunk_i:02d}.png")
            plt.close(fig)

            plt.figure(figsize=(16, 9), dpi=300)
            plt.plot(chunk_scores, "b.", label="Skew response")
            plt.legend()
            plt.savefig(g / f"skew_response_{_chunk_i:04d}.png")
            plt.close()
        scores += chunk_scores

    if np.all(scores == 0):
        return 0.0
    return float(skews[int(np.argmax(scores))])


def validate_skew_hough_extents(extents) -> tuple[float, float]:
    """Return finite, increasing lower and upper skew-search bounds."""
    try:
        skew_min, skew_max = (float(value) for value in extents)
    except (TypeError, ValueError):
        raise ValueError("skew Hough extents must contain exactly two numbers") from None
    if not math.isfinite(skew_min) or not math.isfinite(skew_max):
        raise ValueError("skew Hough extents must be finite")
    if skew_min >= skew_max:
        raise ValueError("skew Hough minimum must be less than its maximum")
    return skew_min, skew_max


def render_output_chunk(
    raw_sampled: np.ndarray,
    *,
    output_dir: Path,
    name_prefix: str,
    white_balance: np.ndarray,
    p2: float,
    p98: float,
    capture_dir: Path,
    jpeg_exif: np.ndarray,
    flip_output_180: bool,
) -> np.ndarray:
    """Write one full-resolution chunk and return its compact stack preview."""
    dng_sampled = raw_sampled
    if flip_output_180:
        dng_sampled = np.ascontiguousarray(dng_sampled[::-1, ::-1, :])
    _dng_path, dng_sha1 = write_hashed_linear_dng(
        output_dir,
        name_prefix,
        dng_sampled,
        white_balance,
        capture_dir=capture_dir,
    )
    if dng_sampled is not raw_sampled:
        del dng_sampled

    out_name = f"{name_prefix}_{dng_sha1}"
    rgb = apply_color_transform(raw_sampled, white_balance)
    np.clip(rgb, 0, 65536, out=rgb)
    sharpen(rgb, in_place=True)
    rgb -= p2
    rgb += 0.1
    rgb /= p98
    np.clip(rgb, 0, 1, out=rgb)
    rgb = rgb[::-1, :, :]
    np.sqrt(rgb, out=rgb)
    rgb *= 255

    rgb_u8 = rgb[:, :, ::-1].astype(np.uint8)
    del rgb
    if flip_output_180:
        rgb_u8 = np.ascontiguousarray(rgb_u8[::-1, ::-1, :])
    try:
        write_jpeg(output_dir / f"{out_name}.jpg", rgb_u8, jpeg_exif)
    except Exception as e:
        print(f'oh no when writing jpeg {e}')
        raise

    return bin2to1_rgb_u8(rgb_u8)


def process_preview(
    g: Path,
    padding_columns: int = PROCESSING_PADDING_COLUMNS,
    *,
    show_progress: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_speed_constraints: bool = True,
    equal_length_chunks: bool = False,
    pad_output_to_equal_widths: bool = False,
    flip_output_180: bool | None = FLIP_OUTPUT_180,
    skew_hough_extents: tuple[float, float] = DEFAULT_SKEW_HOUGH_EXTENTS,
) -> int:
    bins = sorted(list(g.glob("*.bin")))
    if not bins:
        raise ValueError(f"no .bin frames found in {g}")
    if padding_columns < 0:
        raise ValueError("padding_columns must be non-negative")
    skew_min, skew_max = validate_skew_hough_extents(skew_hough_extents)
    capture_timestamp = g.name
    flip_output_180 = resolve_flip_output_180(g, flip_output_180)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    jpeg_exif = build_jpeg_exif(
        parse_capture_datetime(g),
        parse_exposure_time(g),
    )

    frame_starts, frame_widths = bayer_channel_frame_geometry(bins)
    capture_width = int(frame_starts[-1])
    constraints = (
        load_manual_speed_constraints(g, bins) if use_speed_constraints else None
    )
    if equal_length_chunks and pad_output_to_equal_widths:
        raise ValueError(
            "equal-length chunks and cut-based padding are mutually exclusive"
        )
    if equal_length_chunks and (
        constraints is None or not constraints.has_cut_regions
    ):
        raise ValueError(
            "equal-length chunks require annotated start and end cut positions"
        )
    if pad_output_to_equal_widths and (
        constraints is None or not constraints.has_cut_regions
    ):
        raise ValueError(
            "equal-width output padding requires annotated cut positions"
        )
    if constraints is not None and show_progress:
        print(
            f"loaded {constraints.path}: {len(constraints.circles)} circles, "
            f"{sum(len(group.segments) for group in constraints.width_groups)} "
            f"equal-width segments, {len(constraints.cut_positions)} positions"
        )

    if constraints is not None and constraints.has_cut_regions:
        processing_x_start, processing_x_stop = aligned_processing_column_bounds(
            capture_width,
            constraints.cut_positions[0] - padding_columns,
            constraints.cut_positions[-1] + padding_columns,
        )
        selected = select_frames_for_cut_region(
            frame_starts,
            frame_widths,
            constraints.cut_positions,
            padding_columns=padding_columns,
        )
        scan_bins = [bins[index] for index in selected]
    else:
        processing_x_start = None
        processing_x_stop = None
        selected = None
        scan_bins = bins

    downsample_steps = int(math.log2(STRUCTURE_MASK_DOWNSAMPLE))
    if 2**downsample_steps != STRUCTURE_MASK_DOWNSAMPLE:
        raise ValueError("STRUCTURE_MASK_DOWNSAMPLE must be a power of two")

    max_green = 0
    max_green_down = 0
    jitter_black_levels = np.full(4, np.inf, dtype=np.float32)
    jitter_white_levels = np.full(4, -np.inf, dtype=np.float32)
    for i, bin in progress(
        enumerate(scan_bins),
        total=len(scan_bins),
        desc="scanning raw frames",
        show=show_progress,
    ):
        dat1_reshaped = load_raw_frame(
            bin,
            rows=4096,
            output_dtype=np.float32,
        )
        raw_green = dat1_reshaped[1::2, 1::2]
        max_green = max(max_green, np.max(raw_green))
        if ENABLE_COLUMN_JITTER_CORRECTION:
            raw_channels = (
                dat1_reshaped[0::2, 1::2],
                dat1_reshaped[1::2, 1::2],
                dat1_reshaped[0::2, 0::2],
                dat1_reshaped[1::2, 0::2],
            )
            for channel, raw_channel in enumerate(raw_channels):
                jitter_black_levels[channel] = min(
                    jitter_black_levels[channel],
                    float(np.min(raw_channel)),
                )
                jitter_white_levels[channel] = max(
                    jitter_white_levels[channel],
                    float(np.max(raw_channel)),
                )
        raw_green_down = raw_green
        for _ in range(downsample_steps):
            raw_green_down = bin2to1(raw_green_down)
        max_green_down = max(max_green_down, np.max(raw_green_down))
        if ENABLE_COLUMN_JITTER_CORRECTION:
            del raw_channels, raw_channel
        del raw_green_down, raw_green, dat1_reshaped

    if selected is None:
        motion_cache = FileDataCache(bins, dtype=np.uint16, rows=4096)
        motion_green = ChannelView(motion_cache, row_offset=1, col_offset=1)
        scored_blocks = moving_region_block_scores(
            motion_green,
            max_green,
            show_progress=show_progress,
        )
        del motion_green, motion_cache

        min_score = min(score for _start, _stop, score in scored_blocks)
        frame_ends = frame_starts[:-1] + frame_widths
        relative_threshold = min_score * 1.3 if len(scored_blocks) > 1 else 0.0
        score_threshold = max(STRUCTURE_MASK_THRESHOLD, relative_threshold)
        interesting_ranges = [
            (start - padding_columns, stop + padding_columns)
            for start, stop, score in scored_blocks
            if score > score_threshold
        ]
        # If meaningful structure fills every fixed block, there may be no
        # low-scoring background from which to form a 1.3x relative contrast.
        if not interesting_ranges:
            interesting_ranges = [
                (start - padding_columns, stop + padding_columns)
                for start, stop, score in scored_blocks
                if score > STRUCTURE_MASK_THRESHOLD
            ]
        if not interesting_ranges:
            raise ValueError("motion detector did not find a moving image region")
        processing_x_start, processing_x_stop = aligned_processing_column_bounds(
            capture_width,
            min(start for start, _stop in interesting_ranges),
            max(stop for _start, stop in interesting_ranges),
        )
        selected = np.flatnonzero(
            (frame_ends > processing_x_start)
            & (frame_starts[:-1] < processing_x_stop)
        ).tolist()
    if show_progress:
        selection_source = (
            "manual cut ROI" if constraints is not None and constraints.has_cut_regions
            else "motion detector"
        )
        print(
            f"selected ({selection_source})",
            format_contiguous_segments(selected),
        )

    selected_bins = [bins[s] for s in selected]

    white_balance = estimate_white_balance(
        selected_bins,
        show_progress=show_progress,
    )
    if show_progress:
        print("white balance", white_balance)

    p2, p98 = autoexposure(
        selected_bins,
        white_balance,
        show_progress=show_progress,
    )

    # lazily load binary data across selected files without full-memory concatenation
    cache = FileDataCache(selected_bins, dtype=np.uint16, rows=4096)
    data = cache

    base_red = ChannelView(data, row_offset=0, col_offset=1)
    base_blue = ChannelView(data, row_offset=1, col_offset=0)
    base_green1 = ChannelView(data, row_offset=1, col_offset=1)
    base_green2 = ChannelView(data, row_offset=0, col_offset=0)
    assert processing_x_start is not None and processing_x_stop is not None
    selected_file_x0 = int(frame_starts[selected[0]])
    local_x_start = int(processing_x_start - selected_file_x0)
    local_x_stop = int(processing_x_stop - selected_file_x0)
    red_view = ColumnSliceView(base_red, local_x_start, local_x_stop)
    blue_view = ColumnSliceView(base_blue, local_x_start, local_x_stop)
    green1 = ColumnSliceView(base_green1, local_x_start, local_x_stop)
    green2 = ColumnSliceView(base_green2, local_x_start, local_x_stop)
    selected_global_x0 = float(processing_x_start)
    local_constraints = None
    if constraints is not None:
        if selected != list(range(selected[0], selected[-1] + 1)):
            raise ValueError(
                "manual speed constraints require a contiguous raw-frame selection"
            )
        local_constraints = constraints.translated(
            selected_global_x0,
            green1.width,
        )

    mask_full = build_structure_mask(
        green1,
        downsample_steps=downsample_steps,
        max_green_down=max_green_down,
        show_progress=show_progress,
    )

    if STRUCTURE_MASK_DILATE_PX > 0 and mask_full.size > 0:
        dilate_px = max(1, STRUCTURE_MASK_DILATE_PX // STRUCTURE_MASK_DOWNSAMPLE)
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_full = cv2.dilate(mask_full.astype(np.uint8), kernel) > 0

    # Radiometric correction is a property of the raw capture coordinates and
    # must precede geometry estimation. In particular, a line-wide exposure
    # stripe appears at zero displacement in both greens and can otherwise
    # overwhelm the scene displacement that encodes speed.
    jitter_models = None
    if ENABLE_COLUMN_JITTER_CORRECTION:
        jitter_models = column_exposure_jitter_correction(
            red_view,
            green1,
            green2,
            blue_view,
            g=g,
            black_levels=jitter_black_levels,
            white_levels=jitter_white_levels,
            structure_mask=mask_full,
            structure_mask_scale=STRUCTURE_MASK_DOWNSAMPLE,
            show_progress=show_progress,
        )
    valid_correlation_range = None
    if local_constraints is not None and local_constraints.has_cut_regions:
        valid_correlation_range = (
            local_constraints.cut_positions[0],
            local_constraints.cut_positions[-1],
        )
    sample_xs, sample_ys, _weights = windowed_cross_correlation(
        green1,
        green2,
        jitter_models=jitter_models,
        structure_mask=mask_full,
        structure_mask_scale=STRUCTURE_MASK_DOWNSAMPLE,
        valid_x_range=valid_correlation_range,
        show_progress=show_progress,
    )
    del mask_full
    fit_sample_xs = sample_xs
    fit_sample_ys = sample_ys
    fit_weights = _weights
    if local_constraints is not None and local_constraints.has_cut_regions:
        inside_manual_cuts = (
            (sample_xs >= local_constraints.cut_positions[0])
            & (sample_xs <= local_constraints.cut_positions[-1])
        )
        if np.count_nonzero(inside_manual_cuts) >= 4:
            fit_sample_xs = sample_xs[inside_manual_cuts]
            fit_sample_ys = sample_ys[inside_manual_cuts]
            fit_weights = _weights[inside_manual_cuts]
            if show_progress:
                print(
                    "speed correlation: using "
                    f"{len(fit_sample_xs)}/{len(sample_xs)} samples inside "
                    "the annotated start/end cuts"
                )
        elif show_progress:
            print(
                "speed correlation: fewer than four samples lie inside the "
                "annotated cuts; using all samples to determine direction"
            )
    correlation_spline = robust_bspline_fit(
        x=fit_sample_xs,
        y=fit_sample_ys,
        weights=fit_weights,
        smoothness=80000,
        visualize=False,
        g=g,
        verbose=show_progress,
    )
    spline = correlation_spline
    if local_constraints is not None and local_constraints.has_speed_constraints:
        spline = refine_speed_spline_with_constraints(
            correlation_spline,
            local_constraints,
            x_min=0.0,
            x_max=float(green1.width - 1),
            diagnostic_dir=output_dir,
            diagnostic_prefix=f"{capture_timestamp}_speed_constraint_fit",
            global_x_offset=selected_global_x0,
            verbose=show_progress,
        )
    green_displacements = np.interp(
        np.arange(green1.width, dtype=np.float64),
        sample_xs,
        sample_ys,
        left=sample_ys[0],
        right=sample_ys[-1],
    )

    if show_progress:
        print(g, len(bins), data.shape)

    sample_positions, widths, step_scales = generate_sample_positions(
        spline, sample_xs, green1.width
    )
    step_scales = np.asarray(step_scales, dtype=np.float32)
    # chunk size for sampling
    skew_chunk_size = 4096
    hough_stride = 3

    skew = estimate_skew_hough(
        sample_skew_corrected_green_chunks(
            green1,
            green2,
            sample_positions,
            widths,
            stride_x=hough_stride,
            stride_y=hough_stride,
            max_pix=skew_chunk_size,
            jitter_models=jitter_models,
            show_progress=show_progress,
        ),
        g=g,
        stride_x=hough_stride,
        stride_y=hough_stride,
        skew_min=skew_min,
        skew_max=skew_max,
        write_debug=ENABLE_SKEW_DEBUG_OUTPUT,
        show_progress=show_progress,
    )

    if show_progress:
        print("estimated skew", skew)

    # Geometry is now final. The optional motion-aligned residual and
    # scene-protection mask are rendering safeguards, not inputs to either
    # geometry fit; keeping them separate preserves the acyclic raw
    # radiometry -> speed -> skew dependency.
    if jitter_models is not None:
        jitter_models = refine_column_exposure_jitter_correction(
            jitter_models,
            red_view,
            green1,
            green2,
            blue_view,
            green_displacements,
            show_progress=show_progress,
        )
    if ENABLE_JITTER_DEBUG_PLOT and jitter_models is not None:
        row_idx = min(JITTER_DEBUG_ROW, green1.height - 1)
        cols = np.arange(green1.width, dtype=np.int32)
        pre_channels = (
            red_view[row_idx, :],
            green1[row_idx, :],
            green2[row_idx, :],
            blue_view[row_idx, :],
        )
        colors = ("r", "g", "k", "b")
        labels = ("red", "green1", "green2", "blue")
        post_channels = tuple(
            jitter_models.correct_samples(
                values[None, :],
                channel,
                cols,
                np.array([row_idx], dtype=np.int32),
            )[0]
            for channel, values in enumerate(pre_channels)
        )

        fig, (ax_pre, ax_post, ax_delta, ax_confidence) = plt.subplots(
            4, 1, figsize=(16, 14), dpi=200, sharex=True
        )
        for values, color, label in zip(pre_channels, colors, labels):
            ax_pre.plot(cols, values, color, linewidth=0.35, label=label)
        ax_pre.set_ylabel("value")
        ax_pre.set_title(f"Jitter correction row {row_idx} (pre)")
        ax_pre.legend()

        for values, color, label in zip(post_channels, colors, labels):
            ax_post.plot(cols, values, color, linewidth=0.35, label=label)
        ax_post.set_ylabel("value")
        ax_post.set_title("post")
        ax_post.legend()

        for pre, post, color, label in zip(pre_channels, post_channels, colors, labels):
            ax_delta.plot(
                cols,
                post - pre,
                color,
                linewidth=0.35,
                label=label,
            )
        ax_delta.set_title("applied additive correction")
        ax_delta.set_ylabel("raw units")
        ax_delta.legend()

        ax_confidence.plot(cols, jitter_models.confidence, "m", linewidth=0.5)
        ax_confidence.set_title("cross-channel/row consensus")
        ax_confidence.set_ylabel("confidence")
        ax_confidence.set_xlabel("column")
        ax_confidence.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        fig.savefig(g / "jitter_row_debug.png")
        plt.close(fig)

    out_height = green1.height * 2
    center_y = (out_height - 1) / 2.0
    row_offsets = (2 * np.arange(green1.height) - center_y).astype(np.float32)
    sample_positions, widths, kept_slice = truncate_sample_positions_for_skew(
        sample_positions,
        widths,
        skew=skew,
        step_scales=step_scales,
        row_offsets=row_offsets,
        width=green1.width,
        return_slice=True,
    )
    step_scales = step_scales[kept_slice]

    n = len(sample_positions)
    if n == 0:
        return 0
    chunk_ranges = None
    stack_chunk_widths = None
    padded_output_width = None
    if local_constraints is not None and local_constraints.has_cut_regions:
        if equal_length_chunks:
            outer_range = build_cut_aligned_output_ranges(
                sample_positions,
                (
                    local_constraints.cut_positions[0],
                    local_constraints.cut_positions[-1],
                ),
            )[0]
            chunk_ranges = build_equal_output_chunk_ranges(outer_range)
        else:
            chunk_ranges = build_cut_aligned_output_ranges(
                sample_positions,
                local_constraints.cut_positions,
            )
        unpadded_widths = [stop - start for start, stop in chunk_ranges]
        if pad_output_to_equal_widths:
            chunk_ranges, padded_output_width = build_padded_cut_output_ranges(
                chunk_ranges
            )
        output_chunk_count = len(chunk_ranges)
        generator_binned_widths = np.asarray(
            [
                (padded_output_width or stop - start) // 2
                for start, stop in chunk_ranges
            ],
            dtype=np.int64,
        )
        if np.any(generator_binned_widths < 1):
            raise ValueError("an annotated cut region is too narrow to render")
        stack_chunk_widths = (
            generator_binned_widths[::-1]
            if flip_output_180
            else generator_binned_widths
        )
        chunk_size = OUTPUT_CHUNK_TARGET  # Ignored for explicit cut ranges.
        rendered_sample_count = (
            padded_output_width * output_chunk_count
            if padded_output_width is not None
            else sum(stop - start for start, stop in chunk_ranges)
        )
        if show_progress:
            if padded_output_width is not None:
                print(
                    "padded cut-aligned output widths",
                    unpadded_widths,
                    "->",
                    padded_output_width,
                    "(sampling order)",
                )
            else:
                range_description = (
                    "equal-length output widths"
                    if equal_length_chunks
                    else "cut-aligned output widths"
                )
                print(range_description, unpadded_widths, "(sampling order)")
    else:
        target = OUTPUT_CHUNK_TARGET
        n_chunks = max(1, int(math.ceil(n / target)))
        chunk_size = max(1, n // n_chunks)
        n_use = chunk_size * n_chunks
        if n_use < n:
            sample_positions = sample_positions[:n_use]
            widths = widths[:n_use]
            step_scales = step_scales[:n_use]
        output_chunk_count = int(math.ceil(len(sample_positions) / chunk_size))
        rendered_sample_count = len(sample_positions)

    # Analysis stays parallel, but cap the image-sized render working sets.
    # Each 32768-column raw RGB chunk is 1.5 GiB before its DNG/JPEG
    # temporaries, so six simultaneous renderers remain unnecessarily risky.
    output_dir.mkdir(parents=True, exist_ok=True)
    with _render_slots:
        with tempfile.TemporaryDirectory(
            dir=output_dir,
            prefix=f".{capture_timestamp}_render_",
        ) as staging_name:
            staging_dir = Path(staging_name)
            with DiskBackedStackPreview(
                staging_dir,
                output_chunk_count,
                chunk_widths=stack_chunk_widths,
            ) as stacked_preview:
                for chunk_i, raw_sampled in sample_motion_corrected_chunks(
                    red_view,
                    green1,
                    green2,
                    blue_view,
                    sample_positions,
                    widths,
                    stride_x=1,
                    stride_y=1,
                    skew=skew,
                    max_pix=chunk_size,
                    jitter_models=jitter_models,
                    step_scales=step_scales,
                    chunk_ranges=chunk_ranges,
                    outer_padding_width=padded_output_width,
                    show_progress=show_progress,
                ):
                    output_chunk_i = (
                        output_chunk_count - 1 - chunk_i
                        if flip_output_180
                        else chunk_i
                    )
                    name_prefix = f"{capture_timestamp}_NECTA{output_chunk_i:04d}"
                    binned = render_output_chunk(
                        raw_sampled,
                        output_dir=staging_dir,
                        name_prefix=name_prefix,
                        white_balance=white_balance,
                        p2=p2,
                        p98=p98,
                        capture_dir=g,
                        jpeg_exif=jpeg_exif,
                        flip_output_180=flip_output_180,
                    )
                    del raw_sampled
                    stacked_preview.add_chunk(output_chunk_i, binned)
                    del binned

                stacked_preview.write(
                    staging_dir / f"{capture_timestamp}_stack.jpg",
                    jpeg_exif,
                )

            removed_count, migrated_xmps = replace_capture_outputs(
                output_dir,
                staging_dir,
                capture_timestamp,
            )
            if show_progress and (removed_count or migrated_xmps):
                print(
                    f"replaced {removed_count} stale output files; "
                    f"carried forward {migrated_xmps} XMP sidecars"
                )

    return rendered_sample_count


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Demosaic, dewarp, and render Alkeria line-scan captures."
    )
    parser.add_argument(
        "captures",
        nargs="*",
        help=(f"capture directories; relative names are resolved under {linescans}"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for final DNG and JPEG files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--jitter",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_COLUMN_JITTER_CORRECTION,
        help="enable the clipping-aware vertical-stripe correction",
    )
    parser.add_argument(
        "--flip-output-180",
        action=argparse.BooleanOptionalAction,
        default=FLIP_OUTPUT_180,
        help=(
            "rotate final output by 180 degrees; by default this is disabled "
            "before 2026 and enabled from 2026 onward"
        ),
    )
    parser.add_argument(
        "--speed-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            f"load {SPEED_CONSTRAINTS_JSON} for spline refinement and cut-aligned output"
        ),
    )
    parser.add_argument(
        "--equal-length-chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use only annotated start/end positions and split the corrected "
            f"output into nearly equal chunks smaller than {OUTPUT_CHUNK_TARGET} pixels"
        ),
    )
    parser.add_argument(
        "--pad-output-to-equal-widths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "keep annotated cuts and pad their corrected slices to the widest one"
        ),
    )
    parser.add_argument(
        "--jitter-debug",
        action="store_true",
        default=ENABLE_JITTER_DEBUG_PLOT,
        help="write a row-level before/after jitter diagnostic plot",
    )
    parser.add_argument(
        "--skew-debug",
        action="store_true",
        default=ENABLE_SKEW_DEBUG_OUTPUT,
        help="write per-chunk skew energy, mask, histogram, and response plots",
    )
    parser.add_argument(
        "--skew-hough-extents",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help=(
            "minimum and maximum skew slopes searched by the Hough transform "
            f"(default: {DEFAULT_SKEW_HOUGH_EXTENTS[0]} "
            f"{DEFAULT_SKEW_HOUGH_EXTENTS[1]})"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args(argv)
    if args.skew_hough_extents is not None:
        try:
            args.skew_hough_extents = validate_skew_hough_extents(
                args.skew_hough_extents
            )
        except ValueError as error:
            parser.error(str(error))
    return args


def main(argv=None):
    global ENABLE_COLUMN_JITTER_CORRECTION
    global ENABLE_JITTER_DEBUG_PLOT
    global ENABLE_SKEW_DEBUG_OUTPUT

    args = _parse_args(argv)
    ENABLE_COLUMN_JITTER_CORRECTION = args.jitter
    ENABLE_JITTER_DEBUG_PLOT = args.jitter_debug
    ENABLE_SKEW_DEBUG_OUTPUT = args.skew_debug
    output_dir = args.output_dir.expanduser()

    if args.captures:
        candidates = []
        for capture_arg in args.captures:
            candidate = Path(capture_arg).expanduser()
            if not candidate.is_absolute() and not candidate.exists():
                candidate = linescans / candidate
            candidates.append(candidate)
    else:
        candidates = sorted(
            linescans.glob("2026-06-16-02-21-07*"),
            reverse=True,
        )

    dirs = []
    for candidate in candidates:
        if candidate.is_dir():
            dirs.append(candidate)
        else:
            print("skipping directory", candidate)

    if len(dirs) == 1:
        process_kwargs = {"output_dir": output_dir}
        if not args.speed_constraints:
            process_kwargs["use_speed_constraints"] = False
        if args.equal_length_chunks:
            process_kwargs["equal_length_chunks"] = True
        if args.pad_output_to_equal_widths:
            process_kwargs["pad_output_to_equal_widths"] = True
        if args.flip_output_180 is not None:
            process_kwargs["flip_output_180"] = args.flip_output_180
        if args.skew_hough_extents is not None:
            process_kwargs["skew_hough_extents"] = args.skew_hough_extents
        process_preview(dirs[0], **process_kwargs)
        return

    process_kwargs = {
        "show_progress": False,
        "output_dir": output_dir,
    }
    if not args.speed_constraints:
        process_kwargs["use_speed_constraints"] = False
    if args.equal_length_chunks:
        process_kwargs["equal_length_chunks"] = True
    if args.pad_output_to_equal_widths:
        process_kwargs["pad_output_to_equal_widths"] = True
    if args.flip_output_180 is not None:
        process_kwargs["flip_output_180"] = args.flip_output_180
    if args.skew_hough_extents is not None:
        process_kwargs["skew_hough_extents"] = args.skew_hough_extents
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {
            executor.submit(
                process_preview,
                g,
                **process_kwargs,
            ): g
            for g in dirs
        }
        try:
            for future in as_completed(tuple(future_map)):
                # Remove completed futures immediately. In particular, a
                # failed Future owns its exception traceback, whose
                # process_preview frame can otherwise retain gigabytes of
                # arrays until every capture finishes.
                g = future_map.pop(future)
                try:
                    final_width = future.result()
                    print(f"{g} final_width={final_width}")
                except Exception as e:
                    print(f"oh no! {g} {e}")
                finally:
                    del future
        except KeyboardInterrupt:
            for future in future_map:
                future.cancel()


if __name__ == "__main__":
    main()
