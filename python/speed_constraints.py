#!/usr/bin/env python3
"""Interactively annotate geometric speed constraints on raw line-scan data.

The displayed image is deliberately geometric-operation-free: every 2x2 Bayer
cell becomes one RGB pixel, with the two green samples averaged.  Annotation x
coordinates therefore match the half-resolution green planes used by
``preview.py`` and continue monotonically across raw chunk boundaries.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from image_processing import auto_tone_map_to_uint8
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QCursor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPalette,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyleFactory,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

SCHEMA_VERSION = 1
DEFAULT_JSON_NAME = "speed_constraints.json"
DEFAULT_FRAME_ROWS = 4096
DEFAULT_WINDOW_SIZE = 16
BAYER_BIN = 2
MIN_ELLIPSE_SIZE = 3.0
MIN_SEGMENT_WIDTH = 1.0
NAVIGATOR_PREFERRED_HEIGHT = 96
NAVIGATOR_MAX_BYTES = 32 * 1024 * 1024

CIRCLE_COLOR = QColor("#00ff00")
CUT_COLOR = QColor("#ffcf33")
GROUP_COLORS = (
    "#00d7ff",
    "#ff66cc",
    "#ff8c42",
    "#b388ff",
    "#63e6be",
    "#ff6b6b",
    "#74c0fc",
    "#ffd43b",
)


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def capture_defaults_to_rotated_180(directory: Path) -> bool:
    """Choose the view transform that displays each camera-mount era upright."""
    match = re.match(r"^(\d{4})-", Path(directory).name)
    return match is not None and int(match.group(1)) < 2026


def _finite_number(value: object, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{description} must be finite")
    return number


def _rounded(value: float) -> int | float:
    rounded = round(float(value), 3)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _unique_id(prefix: str, existing: set[str]) -> str:
    index = 1
    while f"{prefix}_{index:03d}" in existing:
        index += 1
    return f"{prefix}_{index:03d}"


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    raw_columns: int
    x_start: int

    @property
    def display_columns(self) -> int:
        return self.raw_columns // BAYER_BIN

    @property
    def x_end(self) -> int:
        return self.x_start + self.display_columns


@dataclass
class CaptureSequence:
    directory: Path
    frame_rows: int
    frames: list[FrameInfo]

    @classmethod
    def discover(
        cls,
        directory: Path,
        *,
        frame_rows: int = DEFAULT_FRAME_ROWS,
    ) -> CaptureSequence:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"capture directory does not exist: {directory}")
        if frame_rows <= 0 or frame_rows % BAYER_BIN:
            raise ValueError("frame rows must be a positive even number")

        paths = sorted(directory.glob("*.bin"), key=_natural_key)
        if not paths:
            raise ValueError(f"no .bin files found in {directory}")

        frames: list[FrameInfo] = []
        x_start = 0
        bytes_per_raw_column = frame_rows * np.dtype(np.uint16).itemsize
        for path in paths:
            size = path.stat().st_size
            if size % bytes_per_raw_column:
                raise ValueError(
                    f"{path.name} is {size} bytes; expected a multiple of "
                    f"{bytes_per_raw_column} for {frame_rows} sensor rows"
                )
            raw_columns = size // bytes_per_raw_column
            if raw_columns == 0 or raw_columns % BAYER_BIN:
                raise ValueError(
                    f"{path.name} contains {raw_columns} raw columns; "
                    "an even, non-zero count is required"
                )
            frame = FrameInfo(path=path, raw_columns=raw_columns, x_start=x_start)
            frames.append(frame)
            x_start = frame.x_end

        return cls(directory=directory, frame_rows=frame_rows, frames=frames)

    @property
    def image_height(self) -> int:
        return self.frame_rows // BAYER_BIN

    @property
    def total_width(self) -> int:
        return self.frames[-1].x_end

    @property
    def frame_ends(self) -> list[int]:
        return [frame.x_end for frame in self.frames]

    def frame_index_at_x(self, x: float) -> int:
        index = bisect.bisect_right(self.frame_ends, x)
        return min(max(index, 0), len(self.frames) - 1)

    def window_bounds(self, start: int, count: int) -> tuple[int, int, int, int]:
        count = min(max(1, count), len(self.frames))
        start = min(max(0, start), len(self.frames) - count)
        stop = start + count
        return start, stop, self.frames[start].x_start, self.frames[stop - 1].x_end

    def source_json(self) -> dict[str, object]:
        return {
            "frame_rows": self.frame_rows,
            "bayer_bin": BAYER_BIN,
            "frames": [
                {
                    "file": frame.path.name,
                    "raw_columns": frame.raw_columns,
                    "x_start": frame.x_start,
                    "x_end": frame.x_end,
                }
                for frame in self.frames
            ],
        }


def decode_raw_frame(
    frame: FrameInfo,
    frame_rows: int,
) -> np.ndarray:
    """Return one Bayer-binned frame as an RGB uint8 display image."""
    raw_by_time = np.memmap(
        frame.path,
        dtype="<u2",
        mode="r",
        shape=(frame.raw_columns, frame_rows),
    )
    raw = raw_by_time.T
    height = frame_rows // BAYER_BIN
    width = frame.raw_columns // BAYER_BIN
    rgb = np.empty((height, width, 3), dtype=np.uint8)

    # This is the same Bayer layout used by preview.raw_channels().  Taking the
    # high byte matches the live camera preview and leaves the values entirely
    # un-white-balanced and un-tone-mapped.
    np.right_shift(raw[0::2, 1::2], 8, out=rgb[:, :, 0], casting="unsafe")
    green_1 = np.right_shift(raw[1::2, 1::2], 8).astype(np.uint16)
    green_2 = np.right_shift(raw[0::2, 0::2], 8).astype(np.uint16)
    green_1 += green_2
    np.right_shift(green_1, 1, out=green_1)
    rgb[:, :, 1] = green_1
    np.right_shift(raw[1::2, 0::2], 8, out=rgb[:, :, 2], casting="unsafe")
    del raw, raw_by_time
    return rgb


def decode_raw_frame_thumbnail(
    frame: FrameInfo,
    frame_rows: int,
    target_height: int,
    target_width: int,
    *,
    temporal_samples: int = 2,
) -> np.ndarray:
    """Read a sparse, vertically area-averaged navigator thumbnail.

    Only a few Bayer time-column pairs are touched for each output column, so
    building the overview does not need to stream every byte of every chunk.
    """
    channel_height = frame_rows // BAYER_BIN
    channel_width = frame.raw_columns // BAYER_BIN
    temporal_samples = max(1, temporal_samples)
    bin_edges = np.linspace(0, channel_width, target_width + 1)
    positions = np.empty((target_width, temporal_samples), dtype=np.intp)
    sample_fractions = (
        np.arange(temporal_samples, dtype=np.float64) + 0.5
    ) / temporal_samples
    for column in range(target_width):
        left = bin_edges[column]
        right = bin_edges[column + 1]
        positions[column] = np.clip(
            (left + sample_fractions * (right - left)).astype(np.intp),
            0,
            channel_width - 1,
        )

    even_time = (positions.ravel() * BAYER_BIN).astype(np.intp, copy=False)
    odd_time = even_time + 1
    raw_by_time = np.memmap(
        frame.path,
        dtype="<u2",
        mode="r",
        shape=(frame.raw_columns, frame_rows),
    )

    def channel(time_indices: np.ndarray, sensor_offset: int) -> np.ndarray:
        sampled = np.right_shift(
            raw_by_time[time_indices, sensor_offset::BAYER_BIN],
            8,
        ).astype(np.uint16)
        sampled = sampled.reshape(
            target_width,
            temporal_samples,
            channel_height,
        )
        summed = sampled.sum(axis=1, dtype=np.uint32)
        summed //= temporal_samples
        return summed.T.astype(np.uint16)

    red = channel(odd_time, 0)
    green_1 = channel(odd_time, 1)
    green_2 = channel(even_time, 0)
    blue = channel(even_time, 1)

    narrow_rgb = np.empty((channel_height, target_width, 3), dtype=np.uint8)
    narrow_rgb[:, :, 0] = red
    green_1 += green_2
    green_1 //= 2
    narrow_rgb[:, :, 1] = green_1
    narrow_rgb[:, :, 2] = blue
    del red, green_1, green_2, blue

    if target_height == channel_height:
        thumbnail = narrow_rgb
    else:
        thumbnail = cv2.resize(
            narrow_rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        del narrow_rgb
    return thumbnail


@dataclass
class NavigatorData:
    """Memory-bounded, low-resolution strip spanning the entire capture."""

    rgb: np.ndarray
    frame_starts: tuple[int, ...]
    frame_widths: tuple[int, ...]
    thumbnail_height: int

    @property
    def total_width(self) -> int:
        return self.rgb.shape[1]

    @property
    def memory_bytes(self) -> int:
        return self.rgb.nbytes


def navigator_thumbnail_geometry(
    capture: CaptureSequence,
    *,
    preferred_height: int = NAVIGATOR_PREFERRED_HEIGHT,
    max_bytes: int = NAVIGATOR_MAX_BYTES,
) -> tuple[int, tuple[int, ...]]:
    """Choose aspect-preserving thumbnail sizes within a fixed RGB budget."""
    height = min(max(1, preferred_height), capture.image_height)
    max_pixels = max(1, max_bytes // 3)

    while True:
        widths = tuple(
            max(
                1,
                round(frame.display_columns * height / capture.image_height),
            )
            for frame in capture.frames
        )
        pixels = sum(widths) * height
        if pixels <= max_pixels or height == 1:
            return height, widths
        scale = math.sqrt(max_pixels / pixels)
        height = max(1, min(height - 1, int(height * scale)))


def build_navigator_data(
    capture: CaptureSequence,
    *,
    preferred_height: int = NAVIGATOR_PREFERRED_HEIGHT,
    max_bytes: int = NAVIGATOR_MAX_BYTES,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> NavigatorData | None:
    """Decode one frame at a time into a bounded low-resolution overview."""
    thumbnail_height, frame_widths = navigator_thumbnail_geometry(
        capture,
        preferred_height=preferred_height,
        max_bytes=max_bytes,
    )
    starts: list[int] = []
    total_width = 0
    for width in frame_widths:
        starts.append(total_width)
        total_width += width

    strip = np.empty((thumbnail_height, total_width, 3), dtype=np.uint8)
    total_frames = len(capture.frames)
    progress_stride = max(1, total_frames // 500)
    for index, (frame, start, width) in enumerate(
        zip(capture.frames, starts, frame_widths, strict=True)
    ):
        if cancelled is not None and cancelled():
            return None
        thumbnail = decode_raw_frame_thumbnail(
            frame,
            capture.frame_rows,
            thumbnail_height,
            width,
        )
        strip[:, start : start + width] = thumbnail
        del thumbnail
        completed = index + 1
        if progress is not None and (
            completed == total_frames or completed % progress_stride == 0
        ):
            progress(completed, total_frames)

    return NavigatorData(
        rgb=auto_tone_map_to_uint8(strip),
        frame_starts=tuple(starts),
        frame_widths=frame_widths,
        thumbnail_height=thumbnail_height,
    )


class NavigatorLoader(QThread):
    progress_changed = Signal(int, int)
    navigator_ready = Signal(object)
    loading_failed = Signal(str)

    def __init__(
        self,
        capture: CaptureSequence,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.capture = capture

    def run(self) -> None:
        try:
            data = build_navigator_data(
                self.capture,
                progress=self.progress_changed.emit,
                cancelled=self.isInterruptionRequested,
            )
        # A navigator failure is non-fatal; the detailed annotation GUI remains usable.
        except Exception as exc:  # noqa: BLE001
            self.loading_failed.emit(str(exc))
            return
        if data is not None and not self.isInterruptionRequested():
            self.navigator_ready.emit(data)


@dataclass
class CircleConstraint:
    id: str
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> QPointF:
        return QPointF(self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class WidthSegment:
    id: str
    x1: float
    x2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2


@dataclass
class WidthGroup:
    id: str
    name: str
    segments: list[WidthSegment] = field(default_factory=list)


@dataclass
class CutPosition:
    id: str
    x: float


@dataclass
class AnnotationDocument:
    circles: list[CircleConstraint] = field(default_factory=list)
    width_groups: list[WidthGroup] = field(default_factory=list)
    cut_positions: list[CutPosition] = field(default_factory=list)
    loaded_source: dict[str, object] | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path) -> AnnotationDocument:
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version!r} in {path}; "
                f"expected {SCHEMA_VERSION}"
            )

        circles: list[CircleConstraint] = []
        circle_ids: set[str] = set()
        for index, entry in enumerate(payload.get("circles", [])):
            if not isinstance(entry, dict) or not isinstance(entry.get("bounds"), dict):
                raise TypeError(f"circles[{index}] must contain a bounds object")
            identifier = str(entry.get("id", ""))
            if not identifier or identifier in circle_ids:
                raise ValueError(f"circles[{index}] has a missing or duplicate id")
            bounds = entry["bounds"]
            width = _finite_number(bounds.get("width"), f"circles[{index}].width")
            height = _finite_number(bounds.get("height"), f"circles[{index}].height")
            if width <= 0 or height <= 0:
                raise ValueError(f"circles[{index}] width and height must be positive")
            circles.append(
                CircleConstraint(
                    id=identifier,
                    x=_finite_number(bounds.get("x"), f"circles[{index}].x"),
                    y=_finite_number(bounds.get("y"), f"circles[{index}].y"),
                    width=width,
                    height=height,
                )
            )
            circle_ids.add(identifier)

        width_groups: list[WidthGroup] = []
        group_ids: set[str] = set()
        segment_ids: set[str] = set()
        for group_index, entry in enumerate(payload.get("equal_width_groups", [])):
            if not isinstance(entry, dict):
                raise TypeError(f"equal_width_groups[{group_index}] must be an object")
            identifier = str(entry.get("id", ""))
            if not identifier or identifier in group_ids:
                raise ValueError(
                    f"equal_width_groups[{group_index}] has a missing or duplicate id"
                )
            name = (
                str(entry.get("name", "")).strip() or f"Width group {group_index + 1}"
            )
            segments: list[WidthSegment] = []
            for segment_index, segment_entry in enumerate(entry.get("segments", [])):
                if not isinstance(segment_entry, dict):
                    raise TypeError(
                        f"equal_width_groups[{group_index}].segments[{segment_index}] "
                        "must be an object"
                    )
                segment_id = str(segment_entry.get("id", ""))
                if not segment_id or segment_id in segment_ids:
                    raise ValueError(
                        f"equal_width_groups[{group_index}].segments[{segment_index}] "
                        "has a missing or duplicate id"
                    )
                x1 = _finite_number(
                    segment_entry.get("x1"),
                    f"equal_width_groups[{group_index}].segments[{segment_index}].x1",
                )
                x2 = _finite_number(
                    segment_entry.get("x2"),
                    f"equal_width_groups[{group_index}].segments[{segment_index}].x2",
                )
                if x1 == x2:
                    raise ValueError(
                        f"equal_width_groups[{group_index}].segments[{segment_index}] "
                        "must have non-zero width"
                    )
                segments.append(WidthSegment(segment_id, min(x1, x2), max(x1, x2)))
                segment_ids.add(segment_id)
            width_groups.append(WidthGroup(identifier, name, segments))
            group_ids.add(identifier)

        cut_positions: list[CutPosition] = []
        cut_ids: set[str] = set()
        for index, entry in enumerate(payload.get("cut_positions", [])):
            if not isinstance(entry, dict):
                raise TypeError(f"cut_positions[{index}] must be an object")
            identifier = str(entry.get("id", ""))
            if not identifier or identifier in cut_ids:
                raise ValueError(
                    f"cut_positions[{index}] has a missing or duplicate id"
                )
            cut_positions.append(
                CutPosition(
                    identifier,
                    _finite_number(entry.get("x"), f"cut_positions[{index}].x"),
                )
            )
            cut_ids.add(identifier)
        cut_positions.sort(key=lambda cut: (cut.x, cut.id))

        source = payload.get("source")
        return cls(
            circles=circles,
            width_groups=width_groups,
            cut_positions=cut_positions,
            loaded_source=source if isinstance(source, dict) else None,
        )

    def to_json(self, capture: CaptureSequence) -> dict[str, object]:
        ordered_cuts = sorted(self.cut_positions, key=lambda cut: (cut.x, cut.id))
        cut_entries = []
        for index, cut in enumerate(ordered_cuts):
            if len(ordered_cuts) == 1:
                role = "start_end"
            elif index == 0:
                role = "start"
            elif index == len(ordered_cuts) - 1:
                role = "end"
            else:
                role = "cut"
            cut_entries.append({"id": cut.id, "x": _rounded(cut.x), "role": role})

        return {
            "schema_version": SCHEMA_VERSION,
            "coordinate_system": {
                "name": "bayer_channel_pixels",
                "origin": "top_left",
                "x": "global capture column after 2x Bayer binning",
                "y": "sensor row after 2x Bayer binning",
                "bayer_bin": BAYER_BIN,
            },
            "source": capture.source_json(),
            "circles": [
                {
                    "id": circle.id,
                    "bounds": {
                        "x": _rounded(circle.x),
                        "y": _rounded(circle.y),
                        "width": _rounded(circle.width),
                        "height": _rounded(circle.height),
                    },
                }
                for circle in self.circles
            ],
            "equal_width_groups": [
                {
                    "id": group.id,
                    "name": group.name,
                    "segments": [
                        {
                            "id": segment.id,
                            "x1": _rounded(segment.x1),
                            "x2": _rounded(segment.x2),
                        }
                        for segment in group.segments
                    ],
                }
                for group in self.width_groups
            ],
            "cut_positions": cut_entries,
        }

    def save(self, path: Path, capture: CaptureSequence) -> None:
        payload = self.to_json(capture)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(payload, temporary, indent=2, ensure_ascii=False)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def source_warning(self, capture: CaptureSequence) -> str | None:
        if self.loaded_source is None:
            return None
        old_rows = self.loaded_source.get("frame_rows")
        old_frames = self.loaded_source.get("frames")
        old_names = []
        old_widths = []
        if isinstance(old_frames, list):
            old_names = [
                entry.get("file") for entry in old_frames if isinstance(entry, dict)
            ]
            old_widths = [
                entry.get("raw_columns")
                for entry in old_frames
                if isinstance(entry, dict)
            ]
        current_names = [frame.path.name for frame in capture.frames]
        current_widths = [frame.raw_columns for frame in capture.frames]
        if (
            old_rows != capture.frame_rows
            or old_names != current_names
            or old_widths != current_widths
        ):
            return (
                "The JSON was written for a different raw-frame set. Its annotations "
                "were loaded unchanged; verify their positions before saving."
            )
        return None

    def width_group(self, identifier: str) -> WidthGroup | None:
        return next(
            (group for group in self.width_groups if group.id == identifier), None
        )


def _cosmetic_pen(color: QColor | str, width: float = 1.0) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCosmetic(True)
    return pen


class EllipseHandle(QGraphicsRectItem):
    SIZE = 8.0

    def __init__(self, role: str, parent: EllipseGraphicsItem):
        super().__init__(-self.SIZE / 2, -self.SIZE / 2, self.SIZE, self.SIZE, parent)
        self.role = role
        self.setBrush(QBrush(QColor(0, 255, 0, 145)))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        if role in ("tl", "br"):
            self.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.SizeBDiagCursor))

    @property
    def ellipse(self) -> EllipseGraphicsItem:
        parent = self.parentItem()
        assert isinstance(parent, EllipseGraphicsItem)
        return parent

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.ellipse.begin_resize(self.role)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.ellipse.resize_to(self.role, event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.ellipse.finish_resize()
        event.accept()


class EllipseGraphicsItem(QGraphicsEllipseItem):
    def __init__(
        self,
        constraint: CircleConstraint,
        bounds: QRectF,
        changed: Callable[[str, QRectF], None],
    ):
        super().__init__(0.0, 0.0, constraint.width, constraint.height)
        self.constraint_id = constraint.id
        self.annotation_key = ("circle", constraint.id)
        self._bounds = bounds
        self._changed = changed
        self._resize_opposite = QPointF()
        self.setPos(constraint.x, constraint.y)
        self.setPen(_cosmetic_pen(CIRCLE_COLOR))
        # A practically invisible fill makes the ellipse interior easy to select.
        self.setBrush(QBrush(QColor(0, 255, 0, 1)))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(20)
        self.handles = {
            role: EllipseHandle(role, self) for role in ("tl", "tr", "bl", "br")
        }
        self._position_handles()
        self._show_handles(False)

    def _show_handles(self, visible: bool) -> None:
        for handle in self.handles.values():
            handle.setVisible(visible)

    def _position_handles(self) -> None:
        rect = self.rect()
        self.handles["tl"].setPos(rect.topLeft())
        self.handles["tr"].setPos(rect.topRight())
        self.handles["bl"].setPos(rect.bottomLeft())
        self.handles["br"].setPos(rect.bottomRight())

    def global_rect(self) -> QRectF:
        return QRectF(self.pos(), self.rect().size())

    def itemChange(self, change, value):  # type: ignore[no-untyped-def]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            point = QPointF(value)
            width = self.rect().width()
            height = self.rect().height()
            point.setX(
                min(
                    max(point.x(), self._bounds.left()),
                    max(self._bounds.left(), self._bounds.right() - width),
                )
            )
            point.setY(
                min(
                    max(point.y(), self._bounds.top()),
                    max(self._bounds.top(), self._bounds.bottom() - height),
                )
            )
            return point
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._show_handles(bool(value))
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().mouseReleaseEvent(event)
        self._changed(self.constraint_id, self.global_rect())

    def begin_resize(self, role: str) -> None:
        self.setSelected(True)
        rect = self.global_rect()
        opposites = {
            "tl": rect.bottomRight(),
            "tr": rect.bottomLeft(),
            "bl": rect.topRight(),
            "br": rect.topLeft(),
        }
        self._resize_opposite = opposites[role]

    def resize_to(self, role: str, scene_position: QPointF) -> None:
        point = QPointF(
            min(max(scene_position.x(), self._bounds.left()), self._bounds.right()),
            min(max(scene_position.y(), self._bounds.top()), self._bounds.bottom()),
        )
        opposite = self._resize_opposite
        if "l" in role:
            left = min(point.x(), opposite.x() - MIN_ELLIPSE_SIZE)
            right = opposite.x()
        else:
            left = opposite.x()
            right = max(point.x(), opposite.x() + MIN_ELLIPSE_SIZE)
        if "t" in role:
            top = min(point.y(), opposite.y() - MIN_ELLIPSE_SIZE)
            bottom = opposite.y()
        else:
            top = opposite.y()
            bottom = max(point.y(), opposite.y() + MIN_ELLIPSE_SIZE)

        left = max(left, self._bounds.left())
        right = min(right, self._bounds.right())
        top = max(top, self._bounds.top())
        bottom = min(bottom, self._bounds.bottom())
        rect = QRectF(left, top, right - left, bottom - top)
        if rect.width() < MIN_ELLIPSE_SIZE or rect.height() < MIN_ELLIPSE_SIZE:
            return
        self.setPos(rect.topLeft())
        self.setRect(0.0, 0.0, rect.width(), rect.height())
        self._position_handles()

    def finish_resize(self) -> None:
        self._changed(self.constraint_id, self.global_rect())


class WidthEdgeHandle(QGraphicsRectItem):
    def __init__(self, edge: str, parent: WidthGraphicsItem):
        super().__init__(parent)
        self.edge = edge
        self.setPen(QPen(Qt.PenStyle.NoPen))
        color = QColor(parent.color)
        color.setAlpha(75)
        self.setBrush(QBrush(color))
        self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        self.setZValue(2)
        self.update_geometry()

    @property
    def segment(self) -> WidthGraphicsItem:
        parent = self.parentItem()
        assert isinstance(parent, WidthGraphicsItem)
        return parent

    def update_geometry(self) -> None:
        x = 0.0 if self.edge == "left" else self.segment.width
        self.setRect(x - 4.0, 0.0, 8.0, self.segment.image_height)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.segment.begin_resize(self.edge)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.segment.resize_to(self.edge, event.scenePos().x())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.segment.finish_resize()
        event.accept()


class WidthGraphicsItem(QGraphicsItem):
    def __init__(
        self,
        segment: WidthSegment,
        image_height: float,
        total_width: float,
        color: QColor,
        changed: Callable[[str, float, float], None],
    ):
        super().__init__()
        self.segment_id = segment.id
        self.annotation_key = ("segment", segment.id)
        self.image_height = image_height
        self.total_width = total_width
        self.width = segment.x2 - segment.x1
        self.color = QColor(color)
        self._changed = changed
        self._opposite_x = 0.0
        self.setPos(segment.x1, 0.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(10)
        self.handles = {edge: WidthEdgeHandle(edge, self) for edge in ("left", "right")}
        self._show_handles(False)

    def boundingRect(self) -> QRectF:
        return QRectF(-4.0, 0.0, self.width + 8.0, self.image_height)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(QRectF(0.0, 0.0, self.width, self.image_height))
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[no-untyped-def]
        fill = QColor(self.color)
        fill.setAlpha(28 if not self.isSelected() else 46)
        painter.fillRect(QRectF(0.0, 0.0, self.width, self.image_height), fill)
        painter.setPen(_cosmetic_pen(self.color))
        painter.drawLine(QPointF(0.0, 0.0), QPointF(0.0, self.image_height))
        painter.drawLine(
            QPointF(self.width, 0.0), QPointF(self.width, self.image_height)
        )

    def _show_handles(self, visible: bool) -> None:
        for handle in self.handles.values():
            handle.setVisible(visible)

    def itemChange(self, change, value):  # type: ignore[no-untyped-def]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            point = QPointF(value)
            point.setX(
                min(max(point.x(), 0.0), max(0.0, self.total_width - self.width))
            )
            point.setY(0.0)
            return point
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._show_handles(bool(value))
            self.update()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().mouseReleaseEvent(event)
        self._changed(self.segment_id, self.pos().x(), self.pos().x() + self.width)

    def begin_resize(self, edge: str) -> None:
        self.setSelected(True)
        self._opposite_x = self.pos().x() + (self.width if edge == "left" else 0.0)

    def resize_to(self, edge: str, scene_x: float) -> None:
        scene_x = min(max(scene_x, 0.0), self.total_width)
        if edge == "left":
            x1 = min(scene_x, self._opposite_x - MIN_SEGMENT_WIDTH)
            x2 = self._opposite_x
        else:
            x1 = self._opposite_x
            x2 = max(scene_x, self._opposite_x + MIN_SEGMENT_WIDTH)
        if x2 - x1 < MIN_SEGMENT_WIDTH:
            return
        self.prepareGeometryChange()
        self.width = x2 - x1
        self.setPos(x1, 0.0)
        for handle in self.handles.values():
            handle.update_geometry()
        self.update()

    def finish_resize(self) -> None:
        self._changed(self.segment_id, self.pos().x(), self.pos().x() + self.width)


class CutGraphicsItem(QGraphicsLineItem):
    def __init__(
        self,
        cut: CutPosition,
        image_height: float,
        total_width: float,
        changed: Callable[[str, float], None],
    ):
        super().__init__(0.0, 0.0, 0.0, image_height)
        self.cut_id = cut.id
        self.annotation_key = ("cut", cut.id)
        self.image_height = image_height
        self.total_width = total_width
        self._changed = changed
        self.setPos(cut.x, 0.0)
        self.setPen(_cosmetic_pen(CUT_COLOR))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        self.setZValue(30)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)
        path = QPainterPath(QPointF(0.0, 0.0))
        path.lineTo(0.0, self.image_height)
        return stroker.createStroke(path)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[no-untyped-def]
        painter.setPen(self.pen())
        painter.drawLine(self.line())
        if self.isSelected():
            handle = QColor(CUT_COLOR)
            handle.setAlpha(135)
            painter.fillRect(QRectF(-4.0, 6.0, 8.0, 18.0), handle)

    def itemChange(self, change, value):  # type: ignore[no-untyped-def]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            point = QPointF(value)
            point.setX(min(max(point.x(), 0.0), self.total_width))
            point.setY(0.0)
            return point
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().mouseReleaseEvent(event)
        self._changed(self.cut_id, self.pos().x())


class AnnotationView(QGraphicsView):
    annotation_created = Signal(str, QPointF, QPointF)
    cursor_moved = Signal(QPointF)
    fit_mode_changed = Signal(bool)

    def __init__(self, scene: QGraphicsScene, image_height: int):
        super().__init__(scene)
        self.image_height = float(image_height)
        self.mode = "select"
        self._rotated_180 = False
        self._fit_mode = True
        self._draw_start: QPointF | None = None
        self._preview_item: QGraphicsItem | None = None
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setBackgroundBrush(QBrush(QColor("#101216")))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setDragMode(
            QGraphicsView.DragMode.RubberBandDrag
            if mode == "select"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            QCursor(Qt.CursorShape.ArrowCursor)
            if mode == "select"
            else QCursor(Qt.CursorShape.CrossCursor)
        )

    def _clamped_scene_position(self, event) -> QPointF:  # type: ignore[no-untyped-def]
        point = self.mapToScene(event.position().toPoint())
        rect = self.sceneRect()
        return QPointF(
            min(max(point.x(), rect.left()), rect.right()),
            min(max(point.y(), 0.0), self.image_height),
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.mode == "select" or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._draw_start = self._clamped_scene_position(event)
        if self.mode == "circle":
            preview = QGraphicsEllipseItem()
            preview.setPen(_cosmetic_pen(CIRCLE_COLOR))
            preview.setBrush(QBrush(QColor(0, 255, 0, 12)))
            preview.setZValue(100)
            self._preview_item = preview
        elif self.mode == "width":
            preview = QGraphicsRectItem()
            preview.setPen(_cosmetic_pen(QColor("#00d7ff")))
            preview.setBrush(QBrush(QColor(0, 215, 255, 28)))
            preview.setZValue(100)
            self._preview_item = preview
        else:
            preview = QGraphicsLineItem()
            preview.setPen(_cosmetic_pen(CUT_COLOR))
            preview.setZValue(100)
            self._preview_item = preview
        self.scene().addItem(self._preview_item)
        self._update_preview(self._draw_start)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        point = self._clamped_scene_position(event)
        self.cursor_moved.emit(point)
        if self._draw_start is not None:
            self._update_preview(point)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _update_preview(self, point: QPointF) -> None:
        assert self._draw_start is not None and self._preview_item is not None
        if self.mode == "circle":
            assert isinstance(self._preview_item, QGraphicsEllipseItem)
            self._preview_item.setRect(QRectF(self._draw_start, point).normalized())
        elif self.mode == "width":
            assert isinstance(self._preview_item, QGraphicsRectItem)
            x1, x2 = sorted((self._draw_start.x(), point.x()))
            self._preview_item.setRect(x1, 0.0, x2 - x1, self.image_height)
        else:
            assert isinstance(self._preview_item, QGraphicsLineItem)
            self._preview_item.setLine(point.x(), 0.0, point.x(), self.image_height)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._draw_start is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        end = self._clamped_scene_position(event)
        start = self._draw_start
        if self._preview_item is not None:
            self.scene().removeItem(self._preview_item)
        self._draw_start = None
        self._preview_item = None
        if self.mode == "circle":
            rect = QRectF(start, end).normalized()
            if rect.width() >= MIN_ELLIPSE_SIZE and rect.height() >= MIN_ELLIPSE_SIZE:
                self.annotation_created.emit(
                    self.mode, rect.topLeft(), rect.bottomRight()
                )
        elif self.mode == "width":
            if abs(end.x() - start.x()) >= MIN_SEGMENT_WIDTH:
                self.annotation_created.emit(self.mode, start, end)
        else:
            self.annotation_created.emit(self.mode, end, end)
        event.accept()

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
            event.accept()
            return
        super().wheelEvent(event)

    def zoom(self, factor: float) -> None:
        self.set_fit_mode(False)
        transform = self.transform()
        current = math.hypot(transform.m11(), transform.m12())
        target = min(max(current * factor, 0.05), 16.0)
        self.scale(target / current, target / current)

    @property
    def rotated_180(self) -> bool:
        return self._rotated_180

    @property
    def fit_mode(self) -> bool:
        return self._fit_mode

    def set_rotated_180(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._rotated_180:
            return
        if self._fit_mode:
            self._rotated_180 = enabled
            self._apply_fit()
            return
        center = self.mapToScene(self.viewport().rect().center())
        transform = self.transform()
        zoom = math.hypot(transform.m11(), transform.m12())
        self._rotated_180 = enabled
        self.resetTransform()
        self.scale(zoom, zoom)
        if enabled:
            self.rotate(180.0)
        self.centerOn(center)

    def _apply_fit(self) -> None:
        if not self.sceneRect().isEmpty():
            self.resetTransform()
            if self._rotated_180:
                self.rotate(180.0)
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(self.sceneRect().center())

    def set_fit_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled != self._fit_mode:
            self._fit_mode = enabled
            policy = (
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                if enabled
                else Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self.setHorizontalScrollBarPolicy(policy)
            self.setVerticalScrollBarPolicy(policy)
            self.fit_mode_changed.emit(enabled)
        if enabled:
            self._apply_fit()

    def fit_window(self) -> None:
        self.set_fit_mode(True)

    def actual_size(self) -> None:
        self.set_fit_mode(False)
        self.resetTransform()
        if self._rotated_180:
            self.rotate(180.0)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self._fit_mode:
            self._apply_fit()


@dataclass(frozen=True)
class NavigatorRow:
    first_frame: int
    stop_frame: int
    strip_start: int
    strip_end: int

    @property
    def width(self) -> int:
        return self.strip_end - self.strip_start


class CaptureNavigator(QWidget):
    """Responsive, wrapped view of the low-resolution capture strip."""

    position_selected = Signal(float)
    layout_changed = Signal()

    MARGIN = 6
    ROW_GAP = 6

    def __init__(self, capture: CaptureSequence, parent: QWidget | None = None):
        super().__init__(parent)
        self.capture = capture
        self.data: NavigatorData | None = None
        self.image: QImage | None = None
        self.rows: list[NavigatorRow] = []
        self.rotated_180 = False
        self.window_x_left = 0.0
        self.window_x_right = 0.0
        self.setMinimumHeight(100)
        self.setMinimumWidth(1)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setToolTip("Click to center the detailed frame window")

    def set_data(self, data: NavigatorData) -> None:
        self.data = data
        self.image = QImage(
            data.rgb.data,
            data.rgb.shape[1],
            data.rgb.shape[0],
            data.rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        self._reflow()
        self.update()

    def set_window_range(self, x_left: float, x_right: float) -> None:
        self.window_x_left = x_left
        self.window_x_right = x_right
        self.update()

    def set_rotated_180(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled != self.rotated_180:
            self.rotated_180 = enabled
            self.update()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self) -> None:
        if self.data is None:
            return
        available_width = max(1, self.width() - 2 * self.MARGIN)
        rows: list[NavigatorRow] = []
        first_frame = 0
        row_width = 0
        for index, width in enumerate(self.data.frame_widths):
            if index > first_frame and row_width + width > available_width:
                rows.append(
                    NavigatorRow(
                        first_frame=first_frame,
                        stop_frame=index,
                        strip_start=self.data.frame_starts[first_frame],
                        strip_end=self.data.frame_starts[index],
                    )
                )
                first_frame = index
                row_width = 0
            row_width += width
        rows.append(
            NavigatorRow(
                first_frame=first_frame,
                stop_frame=len(self.data.frame_widths),
                strip_start=self.data.frame_starts[first_frame],
                strip_end=self.data.total_width,
            )
        )
        rows_changed = rows != self.rows
        self.rows = rows
        required_height = (
            2 * self.MARGIN
            + len(rows) * self.data.thumbnail_height
            + max(0, len(rows) - 1) * self.ROW_GAP
        )
        if self.minimumHeight() != required_height:
            self.setMinimumHeight(required_height)
        self.update()
        if rows_changed:
            self.layout_changed.emit()

    def _row_top(self, row_index: int) -> int:
        assert self.data is not None
        return self.MARGIN + row_index * (self.data.thumbnail_height + self.ROW_GAP)

    def _strip_x_from_global(self, global_x: float) -> float:
        assert self.data is not None
        frame_index = self.capture.frame_index_at_x(global_x)
        frame = self.capture.frames[frame_index]
        fraction = (global_x - frame.x_start) / frame.display_columns
        fraction = min(max(fraction, 0.0), 1.0)
        return (
            self.data.frame_starts[frame_index]
            + fraction * self.data.frame_widths[frame_index]
        )

    def _global_x_from_strip(self, strip_x: float) -> float:
        assert self.data is not None
        frame_ends = [
            start + width
            for start, width in zip(
                self.data.frame_starts,
                self.data.frame_widths,
                strict=True,
            )
        ]
        frame_index = min(
            bisect.bisect_right(frame_ends, strip_x),
            len(self.capture.frames) - 1,
        )
        start = self.data.frame_starts[frame_index]
        width = self.data.frame_widths[frame_index]
        fraction = min(max((strip_x - start) / width, 0.0), 1.0)
        frame = self.capture.frames[frame_index]
        return frame.x_start + fraction * frame.display_columns

    def widget_position_for_global_x(self, global_x: float) -> QPointF | None:
        if self.data is None:
            return None
        strip_x = self._strip_x_from_global(global_x)
        for row_index, row in enumerate(self.rows):
            if row.strip_start <= strip_x <= row.strip_end:
                return QPointF(
                    self._widget_x_from_strip(row, strip_x),
                    self._row_top(row_index) + self.data.thumbnail_height / 2,
                )
        return None

    def _widget_x_from_strip(self, row: NavigatorRow, strip_x: float) -> float:
        if self.rotated_180:
            return self.MARGIN + row.strip_end - strip_x
        return self.MARGIN + strip_x - row.strip_start

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        if self.data is None or self.image is None:
            painter.setPen(QColor("#9ea5b1"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Building low-resolution navigator…",
            )
            return

        for row_index, row in enumerate(self.rows):
            top = self._row_top(row_index)
            target = QRectF(
                self.MARGIN,
                top,
                row.width,
                self.data.thumbnail_height,
            )
            source = QRectF(
                row.strip_start,
                0,
                row.width,
                self.data.thumbnail_height,
            )
            if self.rotated_180:
                painter.save()
                painter.translate(target.center())
                painter.rotate(180.0)
                painter.translate(-target.center())
                painter.drawImage(target, self.image, source)
                painter.restore()
            else:
                painter.drawImage(target, self.image, source)
            painter.setPen(_cosmetic_pen(QColor("#454d5a")))
            painter.drawRect(target.adjusted(0.5, 0.5, -0.5, -0.5))

            label = f"{row.first_frame + 1}–{row.stop_frame}"
            label_width = painter.fontMetrics().horizontalAdvance(label) + 10
            label_rect = QRectF(
                self.MARGIN + 3,
                top + 3,
                label_width,
                painter.fontMetrics().height() + 4,
            )
            painter.fillRect(label_rect, QColor(0, 0, 0, 155))
            painter.setPen(QColor("#f4f5f7"))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

        selection_start = self._strip_x_from_global(self.window_x_left)
        selection_end = self._strip_x_from_global(self.window_x_right)
        if selection_end < selection_start:
            selection_start, selection_end = selection_end, selection_start
        for row_index, row in enumerate(self.rows):
            left = max(selection_start, row.strip_start)
            right = min(selection_end, row.strip_end)
            if right <= left:
                continue
            top = self._row_top(row_index)
            widget_left = self._widget_x_from_strip(row, left)
            widget_right = self._widget_x_from_strip(row, right)
            selection = QRectF(
                min(widget_left, widget_right),
                top,
                abs(widget_right - widget_left),
                self.data.thumbnail_height,
            )
            painter.fillRect(selection, QColor(62, 169, 255, 55))
            pen = _cosmetic_pen(QColor("#55c7ff"), 2.0)
            painter.setPen(pen)
            painter.drawRect(selection.adjusted(1.0, 1.0, -1.0, -1.0))

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.data is None or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = event.position()
        for row_index, row in enumerate(self.rows):
            top = self._row_top(row_index)
            if not (
                top <= point.y() < top + self.data.thumbnail_height
                and self.MARGIN <= point.x() < self.MARGIN + row.width
            ):
                continue
            if self.rotated_180:
                strip_x = row.strip_end - (point.x() - self.MARGIN)
            else:
                strip_x = row.strip_start + point.x() - self.MARGIN
            self.position_selected.emit(self._global_x_from_strip(strip_x))
            event.accept()
            return
        super().mousePressEvent(event)


class FramePixmapCache:
    def __init__(
        self,
        capture: CaptureSequence,
        capacity: int = 24,
    ):
        self.capture = capture
        self.capacity = max(1, capacity)
        self._pixmaps: OrderedDict[int, QPixmap] = OrderedDict()

    def get(self, index: int) -> QPixmap:
        if index in self._pixmaps:
            self._pixmaps.move_to_end(index)
            return self._pixmaps[index]
        frame = self.capture.frames[index]
        rgb = auto_tone_map_to_uint8(
            decode_raw_frame(frame, self.capture.frame_rows)
        )
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        self._pixmaps[index] = pixmap
        while len(self._pixmaps) > self.capacity:
            self._pixmaps.popitem(last=False)
        return pixmap


class SpeedConstraintWindow(QMainWindow):
    def __init__(
        self,
        capture: CaptureSequence,
        document: AnnotationDocument,
        json_path: Path,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        cache_frames: int = 24,
    ):
        super().__init__()
        self.capture = capture
        self.document = document
        self.json_path = json_path
        self.window_size = min(max(1, window_size), len(capture.frames))
        self.window_start = 0
        self._window_x_left = 0.0
        self.rotated_180 = capture_defaults_to_rotated_180(capture.directory)
        self._dirty = False
        self._syncing_selection = False
        self.active_group_id = (
            document.width_groups[0].id if document.width_groups else None
        )
        self.scene = QGraphicsScene(self)
        self.view = AnnotationView(self.scene, capture.image_height)
        self.pixmap_cache = FramePixmapCache(capture, cache_frames)
        self.frame_items: list[QGraphicsItem] = []
        self.circle_items: dict[str, EllipseGraphicsItem] = {}
        self.segment_items: dict[str, WidthGraphicsItem] = {}
        self.cut_items: dict[str, CutGraphicsItem] = {}
        self.navigator_loader: NavigatorLoader | None = None

        self.setWindowTitle(f"Nectar speed constraints — {capture.directory.name}")
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(1600, max(480, int(available.width() * 0.9))),
                min(1000, max(360, int(available.height() * 0.9))),
            )
        else:
            self.resize(1280, 800)
        self.setMinimumSize(480, 360)
        self._build_toolbar()
        self._build_central_widget()
        self._set_rotated_180(self.rotated_180)
        self._connect_shortcuts()
        self.view.annotation_created.connect(self._create_annotation)
        self.view.cursor_moved.connect(self._update_cursor_status)
        self.scene.selectionChanged.connect(self._scene_selection_changed)

        self._rebuild_annotation_items()
        self._populate_lists()
        self._load_window(initial=True)
        self.set_mode("select")

        warning = document.source_warning(capture)
        if warning:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.warning(self, "Raw frames changed", warning),
            )
        QTimer.singleShot(0, self.view.fit_window)
        QTimer.singleShot(0, self._start_navigator_loading)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Navigation", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.previous_action = QAction("◀ Previous", self)
        self.previous_action.setToolTip("Move the frame window left (Left arrow)")
        self.previous_action.triggered.connect(self.previous_window)
        toolbar.addAction(self.previous_action)

        self.window_label = QLabel()
        self.window_label.setMinimumWidth(230)
        self.window_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar.addWidget(self.window_label)

        self.next_action = QAction("Next ▶", self)
        self.next_action.setToolTip("Move the frame window right (Right arrow)")
        self.next_action.triggered.connect(self.next_window)
        toolbar.addAction(self.next_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Frames "))
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(1, len(self.capture.frames))
        self.window_size_spin.setValue(self.window_size)
        self.window_size_spin.setToolTip("Number of consecutive raw chunks to display")
        self.window_size_spin.valueChanged.connect(self._set_window_size)
        toolbar.addWidget(self.window_size_spin)
        toolbar.addSeparator()

        self.fit_action = QAction("Fit", self)
        self.fit_action.setCheckable(True)
        self.fit_action.setChecked(self.view.fit_mode)
        self.fit_action.triggered.connect(self.view.set_fit_mode)
        self.view.fit_mode_changed.connect(self.fit_action.setChecked)
        toolbar.addAction(self.fit_action)
        actual_action = QAction("100%", self)
        actual_action.triggered.connect(self.view.actual_size)
        toolbar.addAction(actual_action)
        zoom_out = QAction("−", self)
        zoom_out.triggered.connect(lambda: self.view.zoom(1 / 1.2))
        toolbar.addAction(zoom_out)
        zoom_in = QAction("+", self)
        zoom_in.triggered.connect(lambda: self.view.zoom(1.2))
        toolbar.addAction(zoom_in)
        toolbar.addSeparator()
        self.rotate_checkbox = QCheckBox("Rotate 180°")
        self.rotate_checkbox.setChecked(self.rotated_180)
        self.rotate_checkbox.setToolTip(
            "Rotate the detailed image and navigator without changing annotation coordinates"
        )
        self.rotate_checkbox.toggled.connect(self._set_rotated_180)
        toolbar.addWidget(self.rotate_checkbox)
        toolbar.addSeparator()
        self.panel_action = QAction("Annotations", self)
        self.panel_action.setCheckable(True)
        self.panel_action.setChecked(True)
        self.panel_action.setShortcut(QKeySequence("F9"))
        self.panel_action.setToolTip("Show or hide the annotation panel (F9)")
        self.panel_action.toggled.connect(self._toggle_panel)
        toolbar.addAction(self.panel_action)
        self.navigator_action = QAction("Navigator", self)
        self.navigator_action.setCheckable(True)
        self.navigator_action.setChecked(True)
        self.navigator_action.setShortcut(QKeySequence("F8"))
        self.navigator_action.setToolTip("Show or hide the bottom navigator (F8)")
        self.navigator_action.toggled.connect(self._toggle_navigator)
        toolbar.addAction(self.navigator_action)

    def _build_central_widget(self) -> None:
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.view)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.side_scroll.setMinimumWidth(270)
        self.side_scroll.setWidget(self._build_side_panel())
        self.splitter.addWidget(self.side_scroll)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setCollapsible(1, True)
        panel_width = min(380, max(290, self.width() // 4))
        self.splitter.setSizes([max(350, self.width() - panel_width), panel_width])
        self.navigator = CaptureNavigator(self.capture)
        self.navigator.position_selected.connect(self._navigator_position_selected)
        self.navigator.layout_changed.connect(self._navigator_layout_changed)
        self.navigator_scroll = QScrollArea()
        self.navigator_scroll.setWidgetResizable(True)
        self.navigator_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.navigator_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.navigator_scroll.setWidget(self.navigator)

        navigator_header = QHBoxLayout()
        navigator_header.setContentsMargins(8, 4, 8, 2)
        self.navigator_status = QLabel(
            f"Navigator: preparing {len(self.capture.frames)} chunks…"
        )
        self.navigator_status.setObjectName("muted")
        navigator_header.addWidget(self.navigator_status, 1)
        self.navigator_progress = QProgressBar()
        self.navigator_progress.setRange(0, len(self.capture.frames))
        self.navigator_progress.setValue(0)
        self.navigator_progress.setMaximumWidth(320)
        navigator_header.addWidget(self.navigator_progress)

        self.navigator_panel = QWidget()
        navigator_layout = QVBoxLayout(self.navigator_panel)
        navigator_layout.setContentsMargins(0, 0, 0, 0)
        navigator_layout.setSpacing(0)
        navigator_layout.addLayout(navigator_header)
        navigator_layout.addWidget(self.navigator_scroll, 1)

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.addWidget(self.splitter)
        self.vertical_splitter.addWidget(self.navigator_panel)
        self.vertical_splitter.setStretchFactor(0, 1)
        self.vertical_splitter.setStretchFactor(1, 0)
        self.vertical_splitter.setCollapsible(1, True)
        navigator_height = min(280, max(150, self.height() // 4))
        self.vertical_splitter.setSizes(
            [max(250, self.height() - navigator_height), navigator_height]
        )
        self.setCentralWidget(self.vertical_splitter)

    def _toggle_panel(self, visible: bool) -> None:
        if hasattr(self, "side_scroll"):
            self.side_scroll.setVisible(visible)

    def _toggle_navigator(self, visible: bool) -> None:
        if hasattr(self, "navigator_panel"):
            self.navigator_panel.setVisible(visible)

    def _set_rotated_180(self, enabled: bool) -> None:
        self.rotated_180 = bool(enabled)
        self.view.set_rotated_180(self.rotated_180)
        if hasattr(self, "navigator"):
            self.navigator.set_rotated_180(self.rotated_180)

    def _start_navigator_loading(self) -> None:
        if self.navigator_loader is not None:
            return
        loader = NavigatorLoader(self.capture, self)
        loader.progress_changed.connect(self._navigator_loading_progress)
        loader.navigator_ready.connect(self._navigator_loaded)
        loader.loading_failed.connect(self._navigator_loading_failed)
        self.navigator_loader = loader
        loader.start()

    def _navigator_loading_progress(self, completed: int, total: int) -> None:
        self.navigator_progress.setRange(0, total)
        self.navigator_progress.setValue(completed)
        self.navigator_status.setText(
            f"Navigator: loading chunk {completed} / {total} one at a time…"
        )

    def _navigator_loaded(self, data: NavigatorData) -> None:
        self.navigator.set_data(data)
        _start, _stop, x_left, x_right = self.capture.window_bounds(
            self.window_start, self.window_size
        )
        self.navigator.set_window_range(float(x_left), float(x_right))
        self.navigator_progress.hide()
        self.navigator_status.setText(
            f"Navigator: {len(self.capture.frames)} chunks · "
            f"{data.thumbnail_height}px thumbnails · "
            f"{data.memory_bytes / (1024 * 1024):.1f} MiB retained · "
            "click to center"
        )
        QTimer.singleShot(0, self._ensure_navigator_window_visible)

    def _navigator_loading_failed(self, message: str) -> None:
        self.navigator_progress.hide()
        self.navigator_status.setText(f"Navigator unavailable: {message}")

    def _navigator_position_selected(self, global_x: float) -> None:
        frame_index = self.capture.frame_index_at_x(global_x)
        self.window_start = min(
            max(0, frame_index - self.window_size // 2),
            len(self.capture.frames) - self.window_size,
        )
        self._load_window(center_x=global_x)

    def _navigator_layout_changed(self) -> None:
        QTimer.singleShot(0, self._ensure_navigator_window_visible)

    def _ensure_navigator_window_visible(self) -> None:
        _start, _stop, x_left, x_right = self.capture.window_bounds(
            self.window_start, self.window_size
        )
        position = self.navigator.widget_position_for_global_x((x_left + x_right) / 2)
        if position is not None:
            self.navigator_scroll.ensureVisible(
                round(position.x()),
                round(position.y()),
                30,
                30,
            )

    def _tool_button(self, text: str, mode: str, shortcut: str) -> QPushButton:
        button = QPushButton(f"{text}  [{shortcut}]")
        button.setCheckable(True)
        button.clicked.connect(lambda _checked=False: self.set_mode(mode))
        button.setProperty("mode", mode)
        self.tool_button_group.addButton(button)
        return button

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(250)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Speed constraints")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "Draw on the raw Bayer-binned image. Select an annotation to move it; "
            "green corner handles resize circles."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)

        tools = QGroupBox("Tool")
        tools_layout = QGridLayout(tools)
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(True)
        self.tool_buttons = {
            "select": self._tool_button("Select", "select", "Esc"),
            "circle": self._tool_button("Circle", "circle", "C"),
            "width": self._tool_button("Width", "width", "W"),
            "cut": self._tool_button("Cut", "cut", "K"),
        }
        for index, button in enumerate(self.tool_buttons.values()):
            tools_layout.addWidget(button, index // 2, index % 2)
        layout.addWidget(tools)

        circles_box = QGroupBox("Circles")
        circles_layout = QVBoxLayout(circles_box)
        self.circle_list = QListWidget()
        self.circle_list.setMinimumHeight(105)
        self.circle_list.currentItemChanged.connect(self._circle_list_selected)
        circles_layout.addWidget(self.circle_list)
        circle_buttons = QHBoxLayout()
        draw_circle = QPushButton("Draw circle")
        draw_circle.clicked.connect(lambda: self.set_mode("circle"))
        delete_circle = QPushButton("Delete")
        delete_circle.clicked.connect(self._delete_circle_from_list)
        circle_buttons.addWidget(draw_circle)
        circle_buttons.addWidget(delete_circle)
        circles_layout.addLayout(circle_buttons)
        layout.addWidget(circles_box)

        width_box = QGroupBox("Equal-width groups")
        width_layout = QVBoxLayout(width_box)
        width_help = QLabel(
            "Every full-height slice inside one group represents the same real width."
        )
        width_help.setWordWrap(True)
        width_help.setObjectName("muted")
        width_layout.addWidget(width_help)
        self.width_tree = QTreeWidget()
        self.width_tree.setHeaderHidden(True)
        self.width_tree.setMinimumHeight(160)
        self.width_tree.currentItemChanged.connect(self._width_tree_selected)
        self.width_tree.itemDoubleClicked.connect(self._width_tree_double_clicked)
        width_layout.addWidget(self.width_tree)
        group_buttons = QHBoxLayout()
        add_group = QPushButton("New group")
        add_group.clicked.connect(self._new_width_group)
        rename_group = QPushButton("Rename")
        rename_group.clicked.connect(self._rename_width_group)
        group_buttons.addWidget(add_group)
        group_buttons.addWidget(rename_group)
        width_layout.addLayout(group_buttons)
        segment_buttons = QHBoxLayout()
        draw_segment = QPushButton("Draw segment")
        draw_segment.clicked.connect(lambda: self.set_mode("width"))
        delete_width = QPushButton("Delete selected")
        delete_width.clicked.connect(self._delete_width_selection)
        segment_buttons.addWidget(draw_segment)
        segment_buttons.addWidget(delete_width)
        width_layout.addLayout(segment_buttons)
        layout.addWidget(width_box)

        cuts_box = QGroupBox("Sequence bounds and cuts")
        cuts_layout = QVBoxLayout(cuts_box)
        cuts_help = QLabel(
            "Positions are sorted by x: the first is Start, the last is End, and "
            "all positions between them are Cuts."
        )
        cuts_help.setWordWrap(True)
        cuts_help.setObjectName("muted")
        cuts_layout.addWidget(cuts_help)
        self.cut_list = QListWidget()
        self.cut_list.setMinimumHeight(105)
        self.cut_list.currentItemChanged.connect(self._cut_list_selected)
        cuts_layout.addWidget(self.cut_list)
        cut_buttons = QHBoxLayout()
        place_cut = QPushButton("Place position")
        place_cut.clicked.connect(lambda: self.set_mode("cut"))
        delete_cut = QPushButton("Delete")
        delete_cut.clicked.connect(self._delete_cut_from_list)
        cut_buttons.addWidget(place_cut)
        cut_buttons.addWidget(delete_cut)
        cuts_layout.addLayout(cut_buttons)
        layout.addWidget(cuts_box)

        layout.addStretch(1)
        self.save_status = QLabel(f"Autosaves to\n{self.json_path}")
        self.save_status.setWordWrap(True)
        self.save_status.setObjectName("muted")
        layout.addWidget(self.save_status)
        save_button = QPushButton("Save now  [Ctrl+S]")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self.save)
        layout.addWidget(save_button)
        return panel

    def _connect_shortcuts(self) -> None:
        self.shortcuts: list[QShortcut] = []

        def add(sequence: str | QKeySequence, callback: Callable[[], None]) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

        add(QKeySequence(Qt.Key.Key_Left), self.previous_window)
        add(QKeySequence(Qt.Key.Key_Right), self.next_window)
        add("Escape", lambda: self.set_mode("select"))
        add("C", lambda: self.set_mode("circle"))
        add("W", lambda: self.set_mode("width"))
        add("K", lambda: self.set_mode("cut"))
        add(QKeySequence.StandardKey.Save, self.save)
        add(QKeySequence.StandardKey.Delete, self.delete_selected_annotation)
        add("F", self.view.fit_window)
        add("1", self.view.actual_size)

    def set_mode(self, mode: str) -> None:
        if mode == "width" and self.active_group_id is None:
            group = self._create_width_group(prompt=False)
            if group is None:
                mode = "select"
        self.view.set_mode(mode)
        self.tool_buttons[mode].setChecked(True)
        if mode == "width" and self.active_group_id:
            group = self.document.width_group(self.active_group_id)
            suffix = f" — adding to {group.name}" if group else ""
        else:
            suffix = ""
        self.statusBar().showMessage(f"Tool: {mode}{suffix}")

    def _load_window(
        self, *, initial: bool = False, center_x: float | None = None
    ) -> None:
        old_center = self.view.mapToScene(self.view.viewport().rect().center())
        old_left = self._window_x_left
        start, stop, x_left, x_right = self.capture.window_bounds(
            self.window_start, self.window_size
        )
        self.window_start = start
        self._window_x_left = float(x_left)

        for item in self.frame_items:
            self.scene.removeItem(item)
        self.frame_items.clear()

        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            for index in range(start, stop):
                frame = self.capture.frames[index]
                pixmap_item = QGraphicsPixmapItem(self.pixmap_cache.get(index))
                pixmap_item.setPos(frame.x_start, 0.0)
                pixmap_item.setZValue(-100)
                self.scene.addItem(pixmap_item)
                self.frame_items.append(pixmap_item)

                if index > start:
                    boundary = QGraphicsLineItem(
                        frame.x_start,
                        0.0,
                        frame.x_start,
                        self.capture.image_height,
                    )
                    boundary.setPen(_cosmetic_pen(QColor(255, 255, 255, 55)))
                    boundary.setZValue(-50)
                    self.scene.addItem(boundary)
                    self.frame_items.append(boundary)
        finally:
            QApplication.restoreOverrideCursor()

        rect = QRectF(
            float(x_left),
            0.0,
            float(x_right - x_left),
            float(self.capture.image_height),
        )
        self.scene.setSceneRect(rect)
        self._update_annotation_visibility(float(x_left), float(x_right))
        if self.view.fit_mode:
            self.view.fit_window()
        elif not initial:
            if center_x is None:
                center_x = x_left + (old_center.x() - old_left)
            self.view.centerOn(
                min(max(center_x, rect.left()), rect.right()),
                min(max(old_center.y(), rect.top()), rect.bottom()),
            )

        first_name = self.capture.frames[start].path.stem
        last_name = self.capture.frames[stop - 1].path.stem
        names = first_name if first_name == last_name else f"{first_name} – {last_name}"
        self.window_label.setText(
            f"{start + 1}–{stop} / {len(self.capture.frames)}   {names}"
        )
        self.previous_action.setEnabled(start > 0)
        self.next_action.setEnabled(stop < len(self.capture.frames))
        self.navigator.set_window_range(float(x_left), float(x_right))
        if not initial and self.navigator.data is not None:
            QTimer.singleShot(0, self._ensure_navigator_window_visible)

    def _update_annotation_visibility(self, x_left: float, x_right: float) -> None:
        for circle in self.document.circles:
            item = self.circle_items.get(circle.id)
            if item is not None:
                item.setVisible(
                    circle.x + circle.width >= x_left and circle.x <= x_right
                )
        for group in self.document.width_groups:
            for segment in group.segments:
                item = self.segment_items.get(segment.id)
                if item is not None:
                    item.setVisible(segment.x2 >= x_left and segment.x1 <= x_right)
        for cut in self.document.cut_positions:
            item = self.cut_items.get(cut.id)
            if item is not None:
                item.setVisible(x_left <= cut.x <= x_right)

    def previous_window(self) -> None:
        if self.window_start > 0:
            self.window_start -= 1
            self._load_window()

    def next_window(self) -> None:
        if self.window_start + self.window_size < len(self.capture.frames):
            self.window_start += 1
            self._load_window()

    def _set_window_size(self, value: int) -> None:
        self.window_size = value
        self._load_window()

    def _ensure_x_visible(self, x: float, y: float | None = None) -> None:
        frame_index = self.capture.frame_index_at_x(x)
        if not (
            self.window_start <= frame_index < self.window_start + self.window_size
        ):
            self.window_start = min(
                max(0, frame_index - self.window_size // 2),
                len(self.capture.frames) - self.window_size,
            )
            self._load_window(center_x=x)
        if not self.view.fit_mode:
            self.view.centerOn(x, self.capture.image_height / 2 if y is None else y)

    def _rebuild_annotation_items(self) -> None:
        for item in (
            list(self.circle_items.values())
            + list(self.segment_items.values())
            + list(self.cut_items.values())
        ):
            self.scene.removeItem(item)
        self.circle_items.clear()
        self.segment_items.clear()
        self.cut_items.clear()

        bounds = QRectF(
            0.0,
            0.0,
            float(self.capture.total_width),
            float(self.capture.image_height),
        )
        for circle in self.document.circles:
            item = EllipseGraphicsItem(circle, bounds, self._circle_geometry_changed)
            self.scene.addItem(item)
            self.circle_items[circle.id] = item

        for group_index, group in enumerate(self.document.width_groups):
            color = QColor(GROUP_COLORS[group_index % len(GROUP_COLORS)])
            for segment in group.segments:
                item = WidthGraphicsItem(
                    segment,
                    self.capture.image_height,
                    self.capture.total_width,
                    color,
                    self._segment_geometry_changed,
                )
                self.scene.addItem(item)
                self.segment_items[segment.id] = item

        for cut in self.document.cut_positions:
            item = CutGraphicsItem(
                cut,
                self.capture.image_height,
                self.capture.total_width,
                self._cut_geometry_changed,
            )
            self.scene.addItem(item)
            self.cut_items[cut.id] = item
        _start, _stop, x_left, x_right = self.capture.window_bounds(
            self.window_start, self.window_size
        )
        self._update_annotation_visibility(float(x_left), float(x_right))

    def _populate_lists(self, selected_key: tuple[str, str] | None = None) -> None:
        self._syncing_selection = True
        try:
            self.circle_list.clear()
            for index, circle in enumerate(self.document.circles, 1):
                item = QListWidgetItem(
                    f"Circle {index}   x={circle.x:.1f}, y={circle.y:.1f}   "
                    f"{circle.width:.1f}×{circle.height:.1f}"
                )
                item.setData(Qt.ItemDataRole.UserRole, circle.id)
                self.circle_list.addItem(item)
                if selected_key == ("circle", circle.id):
                    self.circle_list.setCurrentItem(item)

            self.width_tree.clear()
            for group_index, group in enumerate(self.document.width_groups):
                group_item = QTreeWidgetItem(
                    [f"{group.name}  ({len(group.segments)} segments)"]
                )
                group_item.setData(0, Qt.ItemDataRole.UserRole, ("group", group.id))
                color = QColor(GROUP_COLORS[group_index % len(GROUP_COLORS)])
                group_item.setForeground(0, QBrush(color))
                self.width_tree.addTopLevelItem(group_item)
                for segment_index, segment in enumerate(group.segments, 1):
                    child = QTreeWidgetItem(
                        [
                            (
                                f"Segment {segment_index}   "
                                f"x={segment.x1:.1f}–{segment.x2:.1f}   "
                                f"width={segment.x2 - segment.x1:.1f}"
                            )
                        ]
                    )
                    child.setData(
                        0, Qt.ItemDataRole.UserRole, ("segment", segment.id, group.id)
                    )
                    group_item.addChild(child)
                    if selected_key == ("segment", segment.id):
                        self.width_tree.setCurrentItem(child)
                group_item.setExpanded(True)
                if selected_key == ("group", group.id) or (
                    selected_key is None and group.id == self.active_group_id
                ):
                    self.width_tree.setCurrentItem(group_item)

            self.cut_list.clear()
            ordered_cuts = sorted(
                self.document.cut_positions, key=lambda cut: (cut.x, cut.id)
            )
            for index, cut in enumerate(ordered_cuts):
                if len(ordered_cuts) == 1:
                    role = "Start / end"
                elif index == 0:
                    role = "Start"
                elif index == len(ordered_cuts) - 1:
                    role = "End"
                else:
                    role = f"Cut {index}"
                item = QListWidgetItem(f"{role}   x={cut.x:.1f}")
                item.setData(Qt.ItemDataRole.UserRole, cut.id)
                self.cut_list.addItem(item)
                if selected_key == ("cut", cut.id):
                    self.cut_list.setCurrentItem(item)
        finally:
            self._syncing_selection = False

    def _create_annotation(self, mode: str, start: QPointF, end: QPointF) -> None:
        if mode == "circle":
            rect = QRectF(start, end).normalized()
            identifier = _unique_id(
                "circle", {circle.id for circle in self.document.circles}
            )
            self.document.circles.append(
                CircleConstraint(
                    identifier, rect.x(), rect.y(), rect.width(), rect.height()
                )
            )
            selected = ("circle", identifier)
        elif mode == "width":
            group = (
                self.document.width_group(self.active_group_id)
                if self.active_group_id
                else None
            )
            if group is None:
                group = self._create_width_group(prompt=False)
            if group is None:
                return
            x1, x2 = sorted((start.x(), end.x()))
            existing = {
                segment.id
                for existing_group in self.document.width_groups
                for segment in existing_group.segments
            }
            identifier = _unique_id("segment", existing)
            group.segments.append(WidthSegment(identifier, x1, x2))
            selected = ("segment", identifier)
        else:
            identifier = _unique_id(
                "position", {cut.id for cut in self.document.cut_positions}
            )
            self.document.cut_positions.append(CutPosition(identifier, start.x()))
            self.document.cut_positions.sort(key=lambda cut: (cut.x, cut.id))
            selected = ("cut", identifier)

        self._rebuild_annotation_items()
        self._populate_lists(selected)
        self._select_scene_item(selected, switch_to_select=False)
        self._mark_changed()

    def _circle_geometry_changed(self, identifier: str, rect: QRectF) -> None:
        circle = next(
            (circle for circle in self.document.circles if circle.id == identifier),
            None,
        )
        if circle is None:
            return
        circle.x, circle.y = rect.x(), rect.y()
        circle.width, circle.height = rect.width(), rect.height()
        self._populate_lists(("circle", identifier))
        self._mark_changed()

    def _segment_geometry_changed(self, identifier: str, x1: float, x2: float) -> None:
        for group in self.document.width_groups:
            segment = next(
                (segment for segment in group.segments if segment.id == identifier),
                None,
            )
            if segment is not None:
                segment.x1, segment.x2 = min(x1, x2), max(x1, x2)
                self._populate_lists(("segment", identifier))
                self._mark_changed()
                return

    def _cut_geometry_changed(self, identifier: str, x: float) -> None:
        cut = next(
            (cut for cut in self.document.cut_positions if cut.id == identifier), None
        )
        if cut is None:
            return
        cut.x = x
        self.document.cut_positions.sort(key=lambda position: (position.x, position.id))
        self._populate_lists(("cut", identifier))
        self._mark_changed()

    def _scene_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        items = [
            item
            for item in self.scene.selectedItems()
            if hasattr(item, "annotation_key")
        ]
        if not items:
            return
        key = items[0].annotation_key
        if key[0] == "segment":
            for group in self.document.width_groups:
                if any(segment.id == key[1] for segment in group.segments):
                    self.active_group_id = group.id
                    break
        self._populate_lists(key)

    def _select_scene_item(
        self, key: tuple[str, str], *, switch_to_select: bool = True
    ) -> None:
        item: QGraphicsItem | None
        if key[0] == "circle":
            item = self.circle_items.get(key[1])
        elif key[0] == "segment":
            item = self.segment_items.get(key[1])
        else:
            item = self.cut_items.get(key[1])
        if item is None:
            return
        self._syncing_selection = True
        try:
            self.scene.clearSelection()
            item.setSelected(True)
        finally:
            self._syncing_selection = False
        if switch_to_select:
            self.set_mode("select")

    def _circle_list_selected(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if self._syncing_selection or current is None:
            return
        identifier = current.data(Qt.ItemDataRole.UserRole)
        circle = next(
            (circle for circle in self.document.circles if circle.id == identifier),
            None,
        )
        if circle:
            self._select_scene_item(("circle", identifier))
            self._ensure_x_visible(circle.center.x(), circle.center.y())

    def _width_tree_selected(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        if self._syncing_selection or current is None:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data[0] == "group":
            self.active_group_id = data[1]
            self._syncing_selection = True
            try:
                self.scene.clearSelection()
            finally:
                self._syncing_selection = False
            self.statusBar().showMessage(f"Active equal-width group: {current.text(0)}")
            return
        identifier, group_id = data[1], data[2]
        self.active_group_id = group_id
        segment = None
        group = self.document.width_group(group_id)
        if group:
            segment = next(
                (entry for entry in group.segments if entry.id == identifier), None
            )
        if segment:
            self._select_scene_item(("segment", identifier))
            self._ensure_x_visible(segment.center_x)

    def _width_tree_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "group":
            self._rename_width_group()

    def _cut_list_selected(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if self._syncing_selection or current is None:
            return
        identifier = current.data(Qt.ItemDataRole.UserRole)
        cut = next(
            (cut for cut in self.document.cut_positions if cut.id == identifier), None
        )
        if cut:
            self._select_scene_item(("cut", identifier))
            self._ensure_x_visible(cut.x)

    def _create_width_group(self, *, prompt: bool) -> WidthGroup | None:
        default_name = f"Width group {len(self.document.width_groups) + 1}"
        if prompt:
            name, accepted = QInputDialog.getText(
                self, "New equal-width group", "Group name:", text=default_name
            )
            if not accepted:
                return None
            name = name.strip() or default_name
        else:
            name = default_name
        identifier = _unique_id(
            "width_group", {group.id for group in self.document.width_groups}
        )
        group = WidthGroup(identifier, name)
        self.document.width_groups.append(group)
        self.active_group_id = identifier
        self._rebuild_annotation_items()
        self._populate_lists(("group", identifier))
        self._mark_changed()
        return group

    def _new_width_group(self) -> None:
        group = self._create_width_group(prompt=True)
        if group:
            self.set_mode("width")

    def _selected_width_group(self) -> WidthGroup | None:
        current = self.width_tree.currentItem()
        if current:
            data = current.data(0, Qt.ItemDataRole.UserRole)
            if data:
                group_id = data[1] if data[0] == "group" else data[2]
                return self.document.width_group(group_id)
        if self.active_group_id:
            return self.document.width_group(self.active_group_id)
        return None

    def _rename_width_group(self) -> None:
        group = self._selected_width_group()
        if group is None:
            return
        name, accepted = QInputDialog.getText(
            self, "Rename equal-width group", "Group name:", text=group.name
        )
        if accepted and name.strip() and name.strip() != group.name:
            group.name = name.strip()
            self._populate_lists(("group", group.id))
            self._mark_changed()

    def _delete_circle_from_list(self) -> None:
        item = self.circle_list.currentItem()
        if item:
            self._delete_key(("circle", item.data(Qt.ItemDataRole.UserRole)))

    def _delete_cut_from_list(self) -> None:
        item = self.cut_list.currentItem()
        if item:
            self._delete_key(("cut", item.data(Qt.ItemDataRole.UserRole)))

    def _delete_width_selection(self) -> None:
        item = self.width_tree.currentItem()
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data[0] == "segment":
            self._delete_key(("segment", data[1]))
        else:
            self._delete_key(("group", data[1]))

    def delete_selected_annotation(self) -> None:
        selected = [
            item
            for item in self.scene.selectedItems()
            if hasattr(item, "annotation_key")
        ]
        if selected:
            self._delete_key(selected[0].annotation_key)
            return
        if self.circle_list.hasFocus():
            self._delete_circle_from_list()
        elif self.width_tree.hasFocus():
            self._delete_width_selection()
        elif self.cut_list.hasFocus():
            self._delete_cut_from_list()

    def _delete_key(self, key: tuple[str, str]) -> None:
        kind, identifier = key
        if kind == "circle":
            self.document.circles = [
                circle for circle in self.document.circles if circle.id != identifier
            ]
        elif kind == "segment":
            for group in self.document.width_groups:
                group.segments = [
                    segment for segment in group.segments if segment.id != identifier
                ]
        elif kind == "cut":
            self.document.cut_positions = [
                cut for cut in self.document.cut_positions if cut.id != identifier
            ]
        else:
            group = self.document.width_group(identifier)
            if group is None:
                return
            if group.segments:
                answer = QMessageBox.question(
                    self,
                    "Delete equal-width group?",
                    f"Delete {group.name!r} and its {len(group.segments)} segments?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self.document.width_groups = [
                entry for entry in self.document.width_groups if entry.id != identifier
            ]
            self.active_group_id = (
                self.document.width_groups[0].id if self.document.width_groups else None
            )
        self._rebuild_annotation_items()
        self._populate_lists()
        self._mark_changed()

    def _mark_changed(self) -> None:
        _start, _stop, x_left, x_right = self.capture.window_bounds(
            self.window_start, self.window_size
        )
        self._update_annotation_visibility(float(x_left), float(x_right))
        self._dirty = True
        self.save()

    def save(self) -> bool:
        try:
            self.document.save(self.json_path, self.capture)
        except OSError as exc:
            self.save_status.setText(f"Save failed: {exc}")
            QMessageBox.critical(self, "Could not save constraints", str(exc))
            return False
        self._dirty = False
        counts = (
            len(self.document.circles),
            sum(len(group.segments) for group in self.document.width_groups),
            len(self.document.cut_positions),
        )
        self.save_status.setText(
            f"Saved automatically\n{self.json_path.name}\n"
            f"{counts[0]} circles · {counts[1]} widths · {counts[2]} positions"
        )
        self.statusBar().showMessage(f"Saved {self.json_path}", 2500)
        return True

    def _update_cursor_status(self, point: QPointF) -> None:
        frame_index = self.capture.frame_index_at_x(point.x())
        frame = self.capture.frames[frame_index]
        local_x = point.x() - frame.x_start
        self.statusBar().showMessage(
            f"x={point.x():.1f}, y={point.y():.1f}   "
            f"{frame.path.name} local x={local_x:.1f}"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._dirty and not self.save():
            answer = QMessageBox.question(
                self,
                "Close without saving?",
                "The latest constraints could not be saved. Close and lose those changes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.navigator_loader is not None and self.navigator_loader.isRunning():
            self.navigator_loader.requestInterruption()
            self.navigator_loader.wait()
        event.accept()


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle(QStyleFactory.create("Fusion"))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#17191f"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8eaf0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#101216"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1e222a"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#252a34"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f4f5f7"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e8eaf0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#252a34"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e8eaf0"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ff5c5c"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#68b5ff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2878bd"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6f7580")
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#6f7580")
    )
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget { font-size: 13px; }
        QMainWindow, QWidget { background: #17191f; color: #e8eaf0; }
        QToolBar { background: #1e222a; border: 0; border-bottom: 1px solid #343a46; spacing: 5px; padding: 5px; }
        QToolBar QToolButton { padding: 6px 10px; }
        QGroupBox { border: 1px solid #343a46; border-radius: 6px; margin-top: 10px; padding-top: 8px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QPushButton, QToolButton, QSpinBox { background: #252a34; border: 1px solid #3b4350; border-radius: 4px; padding: 6px 8px; }
        QPushButton:hover, QToolButton:hover { background: #303744; border-color: #586474; }
        QPushButton:checked { background: #245b83; border-color: #55a7df; }
        QPushButton#primaryButton { background: #17683d; border-color: #27935a; font-weight: 600; }
        QListWidget, QTreeWidget { background: #101216; border: 1px solid #303641; border-radius: 4px; outline: none; }
        QListWidget::item, QTreeWidget::item { padding: 4px; }
        QListWidget::item:selected, QTreeWidget::item:selected { background: #245b83; color: white; }
        QSplitter::handle { background: #343a46; width: 1px; }
        QStatusBar { background: #1e222a; border-top: 1px solid #343a46; }
        QLabel#panelTitle { font-size: 20px; font-weight: 700; }
        QLabel#muted { color: #9ea5b1; font-weight: 400; }
        QScrollBar:vertical, QScrollBar:horizontal { background: #15181d; border: 0; }
        QScrollBar::handle { background: #454d5a; border-radius: 5px; min-width: 24px; min-height: 24px; }
        QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
        """
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate circle, equal-width, and sequence-cut constraints on raw "
            "Alkeria line-scan frames."
        )
    )
    parser.add_argument(
        "capture_dir",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="directory containing .bin frames (default: current directory)",
    )
    parser.add_argument(
        "--json-name",
        default=DEFAULT_JSON_NAME,
        help=f"JSON filename inside the capture directory (default: {DEFAULT_JSON_NAME})",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"number of consecutive chunks to show (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--frame-rows",
        type=int,
        default=DEFAULT_FRAME_ROWS,
        help=f"raw sensor row count (default: {DEFAULT_FRAME_ROWS})",
    )
    parser.add_argument(
        "--cache-frames",
        type=int,
        default=24,
        help="decoded frame pixmaps retained in memory (default: 24)",
    )
    args = parser.parse_args(argv)
    json_name = Path(args.json_name)
    if json_name.name != args.json_name or json_name.suffix.lower() != ".json":
        parser.error("--json-name must be a .json filename without a directory")
    if args.window_size < 1:
        parser.error("--window-size must be positive")
    if args.cache_frames < 1:
        parser.error("--cache-frames must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        capture = CaptureSequence.discover(args.capture_dir, frame_rows=args.frame_rows)
        json_path = capture.directory / args.json_name
        document = AnnotationDocument.load(json_path)
    except (TypeError, ValueError) as exc:
        print(f"speed_constraints: {exc}", file=sys.stderr)
        return 2

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Nectar speed constraints")
    app.setOrganizationName("Nectar")
    apply_dark_theme(app)
    window = SpeedConstraintWindow(
        capture,
        document,
        json_path,
        window_size=args.window_size,
        cache_frames=args.cache_frames,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
