"""Input helpers for Nectar raw camera frames and Bayer-column annotations."""

from __future__ import annotations

from pathlib import Path

import lz4.block
import numpy as np


RAW_DTYPE = np.dtype("<u2")


def annotation_column_stride(entry: dict) -> int:
    """Old annotations were all placed at one displayed pixel per Bayer column."""
    stride = entry.get("column_stride", 1)
    if type(stride) is not int or stride < 1 or stride & (stride - 1):
        raise ValueError("annotation column_stride must be a positive power of two")
    return stride


def load_raw_frame(
    path: Path,
    *,
    rows: int,
    columns: int | None = None,
    dtype: np.dtype | type = RAW_DTYPE,
    output_dtype: np.dtype | type | None = None,
    buffer: bytes | memoryview | None = None,
) -> np.ndarray:
    """Read sensor columns sequentially and return a ``rows x columns`` view.

    Each raw file stores all sensor rows for one time column contiguously.  The
    disk reader materializes the byte range before transposing it, keeping
    reads sequential even when callers select strided Bayer planes. A cached
    frame can supply the same on-disk bytes in ``buffer``.

    ``columns`` may be supplied to read an exact prefix of a frame.  When it is
    omitted, the entire file is read and must contain a whole number of sensor
    columns.
    """
    path = Path(path)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if columns is not None and columns <= 0:
        raise ValueError("columns must be positive")

    source_dtype = np.dtype(dtype)
    if source_dtype.itemsize <= 0 or source_dtype.hasobject:
        raise ValueError("dtype must have a fixed-width binary representation")

    if columns is None:
        size = path.stat().st_size if buffer is None else memoryview(buffer).nbytes
        bytes_per_column = rows * source_dtype.itemsize
        if size % bytes_per_column:
            raise ValueError(
                f"{path.name} is {size} bytes; expected a multiple of "
                f"{bytes_per_column} for {rows} sensor rows"
            )
        value_count = -1
        expected_values = size // source_dtype.itemsize
    else:
        expected_values = rows * columns
        value_count = expected_values

    if buffer is None:
        flat = np.fromfile(path, dtype=source_dtype, count=value_count)
    else:
        count = min(expected_values, memoryview(buffer).nbytes // source_dtype.itemsize)
        flat = np.frombuffer(buffer, dtype=source_dtype, count=count)
    if flat.size != expected_values:
        raise ValueError(
            f"{path.name} contains only {flat.size // rows} complete sensor "
            f"columns; expected {expected_values // rows}"
        )
    if output_dtype is not None:
        flat = flat.astype(output_dtype, copy=False)
    return flat.reshape(-1, rows).T


class RawFrameStore:
    """Keep frames from the sequential scan for subsequent preview passes."""

    def __init__(self, budget: int, total_bytes: int):
        self.budget = budget
        self.compress = total_bytes > budget
        self.frames: dict[Path, bytes | memoryview] = {}
        self.used = 0
        self.full = budget == 0

    def remember(self, path: Path, raw: np.ndarray) -> None:
        if self.full or path in self.frames:
            return
        payload = memoryview(raw.T).cast("B")
        # LZ4's block format supports at most 0x7E000000 bytes per block.
        if self.compress and len(payload) <= 0x7E000000:
            compressed = lz4.block.compress(payload)
            if len(compressed) < len(payload):
                payload = compressed
        if self.used + len(payload) > self.budget:
            self.full = True
            return
        self.frames[path] = payload
        self.used += len(payload)

    def retain(self, paths) -> None:
        for path in self.frames.keys() - set(paths):
            self.used -= len(self.frames.pop(path))

    def read(self, path: Path, **kwargs) -> np.ndarray:
        payload = self.frames.get(path)
        if isinstance(payload, bytes):
            payload = lz4.block.decompress(payload)
        return load_raw_frame(path, buffer=payload, **kwargs)
