"""Sequential loading helpers for Nectar raw camera frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np


RAW_DTYPE = np.dtype("<u2")


def load_raw_frame(
    path: Path,
    *,
    rows: int,
    columns: int | None = None,
    dtype: np.dtype | type = RAW_DTYPE,
    output_dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Read sensor columns sequentially and return a ``rows x columns`` view.

    Each raw file stores all sensor rows for one time column contiguously.  The
    complete byte range is deliberately materialized with ``np.fromfile``
    before the array is transposed.  This keeps disk access sequential even
    when callers subsequently select strided Bayer planes.

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
        size = path.stat().st_size
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

    flat = np.fromfile(path, dtype=source_dtype, count=value_count)
    if flat.size != expected_values:
        raise ValueError(
            f"{path.name} contains only {flat.size // rows} complete sensor "
            f"columns; expected {expected_values // rows}"
        )
    if output_dtype is not None:
        flat = flat.astype(output_dtype, copy=False)
    return flat.reshape(-1, rows).T
