from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from numpy.typing import NDArray

from boxing.spatial_index import _build_offsets, _geometry_to_use_cylinder
from boxing.utils import argcountsort, count1D


def _resolve_bucket_size(widths: NDArray, bucket_size: int | None) -> int:
    if bucket_size is not None:
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be > 0, got {bucket_size}")
        return int(bucket_size)
    return max(1, int(np.ceil(np.median(widths))))


def _construct_sorted_boxes_from_centers(
    xs: NDArray,
    ys: NDArray,
    xscales: NDArray,
    yscales: NDArray,
    xmult: float,
    ymult: float,
    intensities: NDArray,
    precursor_idxs: NDArray,
    x_bucket_size: int,
    y_bucket_size: int,
) -> tuple[NDArray, int, int]:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    xscales = np.asarray(xscales, dtype=np.float64)
    yscales = np.asarray(yscales, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.int64)
    precursor_idxs = np.asarray(precursor_idxs, dtype=np.int64)

    rx = float(xmult) * xscales
    ry = float(ymult) * yscales
    if (rx < 0).any() or (ry < 0).any():
        raise ValueError("xmult/ymult times scales must be >= 0")

    boxes = np.empty((len(xs), 6), dtype=np.int64)
    boxes[:, 0] = np.floor(xs - rx).clip(0).astype(np.int64)
    boxes[:, 1] = np.ceil(xs + rx).astype(np.int64)
    boxes[:, 2] = np.floor(ys - ry).clip(0).astype(np.int64)
    boxes[:, 3] = np.ceil(ys + ry).astype(np.int64)
    boxes[:, 4] = precursor_idxs
    boxes[:, 5] = intensities

    first_x_all = (boxes[:, 0] // int(x_bucket_size)).astype(np.int64)
    n_x_buckets = int(boxes[:, 1].max()) // int(x_bucket_size) + 1
    n_y_buckets = int(boxes[:, 3].max()) // int(y_bucket_size) + 1

    keys = first_x_all.astype(np.intp)
    x_counts = np.zeros(n_x_buckets, dtype=np.uint32)
    count1D(keys, counts=x_counts)
    box_order = argcountsort(keys, counts=x_counts).astype(np.int32)
    return np.ascontiguousarray(boxes[box_order]), n_x_buckets, n_y_buckets


@numba.njit(boundscheck=True)
def _count_nbs(
    boxes: NDArray,
    x_bucket_size: np.int64,
    y_bucket_size: np.int64,
    n_x_buckets: np.int64,
    n_y_buckets: np.int64,
) -> NDArray:
    K = len(boxes)
    cnts = np.zeros((n_x_buckets, n_y_buckets), dtype=np.int64)

    for i in range(K):
        x_lo = boxes[i, 0]
        x_hi = boxes[i, 1]
        y_lo = boxes[i, 2]
        y_hi = boxes[i, 3]

        first_x = x_lo // x_bucket_size
        last_x = (x_hi - np.int64(1)) // x_bucket_size
        first_y = y_lo // y_bucket_size
        last_y = (y_hi - np.int64(1)) // y_bucket_size

        if first_x < np.int64(0):
            first_x = np.int64(0)
        if last_x >= n_x_buckets:
            last_x = n_x_buckets - np.int64(1)
        if first_y < np.int64(0):
            first_y = np.int64(0)
        if last_y >= n_y_buckets:
            last_y = n_y_buckets - np.int64(1)

        for x_idx in range(first_x, last_x + 1):
            for y_idx in range(first_y, last_y + 1):
                cnts[x_idx, y_idx] += 1

    return cnts


@numba.njit(boundscheck=True)
def _fill_nbs(
    boxes: NDArray,
    x_bucket_size: np.int64,
    y_bucket_size: np.int64,
    n_x_buckets: np.int64,
    n_y_buckets: np.int64,
    x_starts: NDArray,
    xy_offsets: NDArray,
    cursors: NDArray,
    nbs: NDArray,
) -> None:
    for i in range(len(boxes)):
        x_lo = boxes[i, 0]
        x_hi = boxes[i, 1]
        y_lo = boxes[i, 2]
        y_hi = boxes[i, 3]

        first_x = x_lo // x_bucket_size
        last_x = (x_hi - np.int64(1)) // x_bucket_size
        first_y = y_lo // y_bucket_size
        last_y = (y_hi - np.int64(1)) // y_bucket_size

        if first_x < np.int64(0):
            first_x = np.int64(0)
        if last_x >= n_x_buckets:
            last_x = n_x_buckets - np.int64(1)
        if first_y < np.int64(0):
            first_y = np.int64(0)
        if last_y >= n_y_buckets:
            last_y = n_y_buckets - np.int64(1)

        for x_idx in range(first_x, last_x + 1):
            for y_idx in range(first_y, last_y + 1):
                pos = (
                    x_starts[x_idx]
                    + xy_offsets[x_idx, y_idx]
                    + cursors[x_idx, y_idx]
                )
                nbs[pos] = np.int32(i)
                cursors[x_idx, y_idx] += 1


@dataclass
class Grid2DSpatialIndex:
    cnts: NDArray
    x_starts: NDArray
    xy_offsets: NDArray
    nbs: NDArray
    x_bucket_size: int
    y_bucket_size: int

    @classmethod
    def from_centers(
        cls,
        xs: NDArray,
        ys: NDArray,
        xscales: NDArray,
        yscales: NDArray,
        intensities: NDArray,
        precursor_idxs: NDArray | None = None,
        xmult: float = 1.0,
        ymult: float = 1.0,
        x_bucket_size: int | None = None,
        y_bucket_size: int | None = None,
    ) -> Grid2DSpatialIndex:
        N = len(xs)
        assert N == len(ys)
        assert N == len(xscales)
        assert N == len(yscales)
        assert N == len(intensities)

        if precursor_idxs is None:
            precursor_idxs = np.arange(N, dtype=np.int64)
        assert N == len(precursor_idxs)

        x_widths = 2.0 * float(xmult) * np.asarray(xscales, dtype=np.float64)
        y_widths = 2.0 * float(ymult) * np.asarray(yscales, dtype=np.float64)
        x_bucket_size = _resolve_bucket_size(x_widths, x_bucket_size)
        y_bucket_size = _resolve_bucket_size(y_widths, y_bucket_size)

        boxes, n_x_buckets, n_y_buckets = _construct_sorted_boxes_from_centers(
            xs,
            ys,
            xscales,
            yscales,
            xmult,
            ymult,
            intensities,
            precursor_idxs,
            x_bucket_size,
            y_bucket_size,
        )

        cnts = _count_nbs(
            boxes,
            np.int64(x_bucket_size),
            np.int64(y_bucket_size),
            np.int64(n_x_buckets),
            np.int64(n_y_buckets),
        )
        x_starts, xy_offsets = _build_offsets(cnts)
        nbs = np.empty(int(x_starts[-1]), dtype=np.int32)
        cursors = np.zeros((n_x_buckets, n_y_buckets), dtype=np.int64)
        _fill_nbs(
            boxes,
            np.int64(x_bucket_size),
            np.int64(y_bucket_size),
            np.int64(n_x_buckets),
            np.int64(n_y_buckets),
            x_starts,
            xy_offsets,
            cursors,
            nbs,
        )
        return cls(
            cnts=cnts,
            x_starts=x_starts,
            xy_offsets=xy_offsets,
            nbs=nbs,
            x_bucket_size=x_bucket_size,
            y_bucket_size=y_bucket_size,
        )

    def query(
        self,
        centers: NDArray,
        scales: NDArray,
        intensities: NDArray,
        precursor_idxs: NDArray | None = None,
        *,
        x_outer_mult: float,
        y_outer_mult: float,
        z_outer_rad: float,
        x_inner_mult: float,
        y_inner_mult: float,
        z_inner_rad: float,
        top_k: int,
        geometry: str = "box",
        inner_xy_mults: tuple[float, float] | None = None,
        inner_z_rad: float | None = None,
    ):
        _geometry_to_use_cylinder(geometry)
        raise NotImplementedError("Grid2DSpatialIndex.query not implemented yet")
