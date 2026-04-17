"""
2D spatial bucket index for axis-aligned boxes over generic (xx, yy) axes.

The caller chooses which physical coordinates map to xx and yy — e.g.
(frame, scan), (urt, scan), etc.

Layout
------
::

    row_starts:    int64[BX + 1]       absolute start of each xx-bucket in flat_members
    cell_offsets:  int64[BX, BY + 1]   per-row yy cumsum offsets (row-local, not global)
    flat_members:  int32[M]            concatenated box indices per cell

Cell (bx, by) access::

    start = row_starts[bx] + cell_offsets[bx, by]
    end   = row_starts[bx] + cell_offsets[bx, by + 1]
    members = flat_members[start:end]

Design notes
------------
- Half-open intervals: box spans [lo, hi); last bucket = (hi - 1) // width.
- Zero-width boxes (lo == hi on any axis) are rejected by validate_boxes_2d.
- Bucket IDs are clamped to [0, n_*_buckets - 1].
- Bucket widths <= 0 raise ValueError.
- Two-pass CSR build: count → prefix-sum → fill.
- Core kernels are @numba.njit(boundscheck=True) and take flat contiguous arrays only.
"""
from __future__ import annotations

import math

import numpy as np
import numba
from numpy.typing import NDArray
from numba_progress import ProgressBar

from boxing.utils import count1D, argcountsort


# ---------------------------------------------------------------------------
# Validation and helpers (pure Python / NumPy)
# ---------------------------------------------------------------------------


def validate_boxes_2d(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Validate box arrays and return contiguous uint32 copies.

    xx_lo/xx_hi, yy_lo/yy_hi : half-open [lo, hi) bounds for each box along
    the two axes.  Any array dtype is accepted; negative lo values are clipped
    to 0.  Zero-width boxes (lo >= hi) are rejected.

    >>> import numpy as np
    >>> xl = np.array([0, 10], dtype=np.uint32)
    >>> xh = np.array([5, 20], dtype=np.uint32)
    >>> yl = np.array([1,  3], dtype=np.uint32)
    >>> yh = np.array([4,  8], dtype=np.uint32)
    >>> out = validate_boxes_2d(xl, xh, yl, yh)
    >>> [a.dtype for a in out]
    [dtype('uint32'), dtype('uint32'), dtype('uint32'), dtype('uint32')]
    """
    # Validate in int64 to avoid uint32 wraparound masking genuine errors.
    xx_lo_i64 = np.asarray(xx_lo, dtype=np.int64)
    xx_hi_i64 = np.asarray(xx_hi, dtype=np.int64)
    yy_lo_i64 = np.asarray(yy_lo, dtype=np.int64)
    yy_hi_i64 = np.asarray(yy_hi, dtype=np.int64)
    n = len(xx_lo_i64)
    if not (len(xx_hi_i64) == len(yy_lo_i64) == len(yy_hi_i64) == n):
        raise ValueError("All four arrays must have equal length.")
    if np.any(xx_hi_i64 <= xx_lo_i64):
        raise ValueError(
            "Zero-width or inverted boxes on xx axis are not allowed "
            "(require xx_hi > xx_lo)."
        )
    if np.any(yy_hi_i64 <= yy_lo_i64):
        raise ValueError(
            "Zero-width or inverted boxes on yy axis are not allowed "
            "(require yy_hi > yy_lo)."
        )
    # Negative coordinates are clipped to 0: buckets start at 0 so any
    # negative lo simply means the box starts before the grid origin.
    xx_lo_u32 = np.ascontiguousarray(np.clip(xx_lo_i64, 0, None), dtype=np.uint32)
    xx_hi_u32 = np.ascontiguousarray(xx_hi_i64, dtype=np.uint32)
    yy_lo_u32 = np.ascontiguousarray(np.clip(yy_lo_i64, 0, None), dtype=np.uint32)
    yy_hi_u32 = np.ascontiguousarray(yy_hi_i64, dtype=np.uint32)
    return xx_lo_u32, xx_hi_u32, yy_lo_u32, yy_hi_u32


def box_widths_2d(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
) -> tuple[NDArray, NDArray]:
    """Return per-box widths along xx and yy axes as int64.

    xx_lo/xx_hi, yy_lo/yy_hi : raw box bounds (any dtype).
    Casts to int64 before subtraction to avoid uint32 underflow.

    >>> import numpy as np
    >>> xw, yw = box_widths_2d(
    ...     np.array([10], dtype=np.uint32), np.array([15], dtype=np.uint32),
    ...     np.array([3],  dtype=np.uint32), np.array([7],  dtype=np.uint32))
    >>> int(xw[0]), int(yw[0])
    (5, 4)
    """
    xx_widths = xx_hi.astype(np.int64) - xx_lo.astype(np.int64)
    yy_widths = yy_hi.astype(np.int64) - yy_lo.astype(np.int64)
    return xx_widths, yy_widths


def get_multiplied_median_bucket_widths(
    xx_widths: NDArray,
    yy_widths: NDArray,
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
) -> tuple[int, int]:
    """Suggest bucket widths as factor * median box width, minimum 1.

    xx_widths, yy_widths : per-box widths from box_widths_2d.
    xx_factor, yy_factor : multiplier applied to the median width per axis
        (default 2.0 — one bucket ≈ two box-widths wide).

    >>> import numpy as np
    >>> get_multiplied_median_bucket_widths(np.array([10, 20, 30]), np.array([5, 10, 15]))
    (40, 20)
    """
    bw_xx = max(1, int(xx_factor * float(np.median(xx_widths))))
    bw_yy = max(1, int(yy_factor * float(np.median(yy_widths))))
    return bw_xx, bw_yy


# ---------------------------------------------------------------------------
# Count pass — Numba kernel
# ---------------------------------------------------------------------------


@numba.njit(boundscheck=True)
def _count_cell_memberships_numba(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    xx_bucket_width: np.int64,
    yy_bucket_width: np.int64,
    n_xx_buckets: np.int64,
    n_yy_buckets: np.int64,
    counts_2d: NDArray,
) -> None:
    """Increment counts_2d[bx, by] for every (bx, by) cell each box overlaps.

    Inputs must be contiguous uint32 arrays. counts_2d is int64[BX, BY],
    modified in-place.

    Bucket assignment (half-open intervals):
        first_bucket = lo // width
        last_bucket  = (hi - 1) // width   (safe since hi > lo always)

    Bucket IDs are clamped to [0, n_*_buckets - 1].
    """
    n_boxes = len(xx_lo)
    for i in range(n_boxes):
        first_xx_bucket = np.int64(xx_lo[i]) // xx_bucket_width
        last_xx_bucket = (np.int64(xx_hi[i]) - np.int64(1)) // xx_bucket_width
        first_yy_bucket = np.int64(yy_lo[i]) // yy_bucket_width
        last_yy_bucket = (np.int64(yy_hi[i]) - np.int64(1)) // yy_bucket_width

        if first_xx_bucket < np.int64(0):
            first_xx_bucket = np.int64(0)
        if last_xx_bucket >= n_xx_buckets:
            last_xx_bucket = n_xx_buckets - np.int64(1)
        if first_yy_bucket < np.int64(0):
            first_yy_bucket = np.int64(0)
        if last_yy_bucket >= n_yy_buckets:
            last_yy_bucket = n_yy_buckets - np.int64(1)

        for xx_bucket_idx in range(first_xx_bucket, last_xx_bucket + 1):
            for yy_bucket_idx in range(first_yy_bucket, last_yy_bucket + 1):
                counts_2d[xx_bucket_idx, yy_bucket_idx] += 1


def count_cell_memberships(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    xx_bucket_width: int,
    yy_bucket_width: int,
    n_xx_buckets: int,
    n_yy_buckets: int,
) -> NDArray:
    """Return int64 count matrix of shape (n_xx_buckets, n_yy_buckets).

    xx_lo/xx_hi, yy_lo/yy_hi : box bounds (any dtype).
    xx_bucket_width, yy_bucket_width : bucket size along each axis (int, > 0).
    n_xx_buckets, n_yy_buckets : grid dimensions; out-of-range IDs are clamped.

    counts[bx, by] = number of boxes that overlap cell (bx, by).

    >>> import numpy as np
    >>> xl = np.array([0], dtype=np.uint32)
    >>> xh = np.array([3], dtype=np.uint32)
    >>> yl = np.array([0], dtype=np.uint32)
    >>> yh = np.array([3], dtype=np.uint32)
    >>> count_cell_memberships(xl, xh, yl, yh, 2, 2, 3, 3)
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 0]])
    """
    xx_lo_u32, xx_hi_u32, yy_lo_u32, yy_hi_u32 = validate_boxes_2d(xx_lo, xx_hi, yy_lo, yy_hi)
    counts = np.zeros((n_xx_buckets, n_yy_buckets), dtype=np.int64)
    _count_cell_memberships_numba(
        xx_lo_u32, xx_hi_u32, yy_lo_u32, yy_hi_u32,
        np.int64(xx_bucket_width), np.int64(yy_bucket_width),
        np.int64(n_xx_buckets), np.int64(n_yy_buckets),
        counts,
    )
    return counts


# ---------------------------------------------------------------------------
# Index construction (pure NumPy — not in hot path)
# ---------------------------------------------------------------------------


def _build_offsets(
    counts_2d: NDArray,
) -> tuple[NDArray, NDArray]:
    """Convert a (BX, BY) count matrix into row_starts and cell_offsets.

    cell_offsets[bx, by] is the row-local cumsum offset — the number of
    members in cells (bx, 0) .. (bx, by-1) combined.  cell_offsets has
    shape (BX, BY+1) with cell_offsets[:, 0] == 0.

    row_starts[bx] is the absolute position of xx-bucket bx's first
    member in flat_members.  row_starts has shape (BX+1,).
    """
    n_xx_buckets, n_yy_buckets = counts_2d.shape
    cell_offsets = np.empty((n_xx_buckets, n_yy_buckets + 1), dtype=np.int64)
    cell_offsets[:, 0] = 0
    np.cumsum(counts_2d, axis=1, out=cell_offsets[:, 1:])
    row_member_counts = cell_offsets[:, -1]
    row_starts = np.empty(n_xx_buckets + 1, dtype=np.int64)
    row_starts[0] = 0
    np.cumsum(row_member_counts, out=row_starts[1:])
    return row_starts, cell_offsets


# ---------------------------------------------------------------------------
# Fill pass — Numba kernel
# ---------------------------------------------------------------------------


@numba.njit(boundscheck=True)
def _fill_memberships_numba(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    xx_bucket_width: np.int64,
    yy_bucket_width: np.int64,
    n_xx_buckets: np.int64,
    n_yy_buckets: np.int64,
    row_starts: NDArray,
    cell_offsets: NDArray,
    cursors: NDArray,
    flat_members: NDArray,
) -> None:
    """Write box index i into flat_members for every cell it overlaps.

    cursors is an int64[BX, BY] scratch array, zero-initialised before call.
    For each (bx, by) the write position is:
        pos = row_starts[bx] + cell_offsets[bx, by] + cursors[bx, by]
    cursors[bx, by] is post-incremented after each write.
    """
    n_boxes = len(xx_lo)
    for i in range(n_boxes):
        first_xx_bucket = np.int64(xx_lo[i]) // xx_bucket_width
        last_xx_bucket = (np.int64(xx_hi[i]) - np.int64(1)) // xx_bucket_width
        first_yy_bucket = np.int64(yy_lo[i]) // yy_bucket_width
        last_yy_bucket = (np.int64(yy_hi[i]) - np.int64(1)) // yy_bucket_width

        if first_xx_bucket < np.int64(0):
            first_xx_bucket = np.int64(0)
        if last_xx_bucket >= n_xx_buckets:
            last_xx_bucket = n_xx_buckets - np.int64(1)
        if first_yy_bucket < np.int64(0):
            first_yy_bucket = np.int64(0)
        if last_yy_bucket >= n_yy_buckets:
            last_yy_bucket = n_yy_buckets - np.int64(1)

        for xx_bucket_idx in range(first_xx_bucket, last_xx_bucket + 1):
            for yy_bucket_idx in range(first_yy_bucket, last_yy_bucket + 1):
                pos = (row_starts[xx_bucket_idx]
                       + cell_offsets[xx_bucket_idx, yy_bucket_idx]
                       + cursors[xx_bucket_idx, yy_bucket_idx])
                flat_members[pos] = np.int32(i)
                cursors[xx_bucket_idx, yy_bucket_idx] += 1


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_spatial_index_2d(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    xx_bucket_width: int,
    yy_bucket_width: int,
    n_xx_buckets: int,
    n_yy_buckets: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build a 2D CSR-like spatial index for axis-aligned boxes.

    Parameters
    ----------
    xx_lo, xx_hi, yy_lo, yy_hi : array-like
        Box bounds, half-open [lo, hi).  Converted to uint32 internally.
    xx_bucket_width, yy_bucket_width : int
        Size of each bucket along the respective axis.
    n_xx_buckets, n_yy_buckets : int
        Grid dimensions.  Out-of-range bucket IDs are clamped.

    Returns
    -------
    row_starts : int64[BX + 1]
    cell_offsets : int64[BX, BY + 1]
    flat_members : int32[M]

    Cell (bx, by) members::

        start = row_starts[bx] + cell_offsets[bx, by]
        end   = row_starts[bx] + cell_offsets[bx, by + 1]
        members = flat_members[start:end]

    Raises
    ------
    ValueError
        If bucket_width <= 0 or any box has lo >= hi.

    >>> import numpy as np
    >>> xl = np.array([0, 4], dtype=np.uint32)
    >>> xh = np.array([3, 7], dtype=np.uint32)
    >>> yl = np.array([0, 4], dtype=np.uint32)
    >>> yh = np.array([3, 7], dtype=np.uint32)
    >>> rs, co, fm = build_spatial_index_2d(xl, xh, yl, yh, 4, 4, 2, 2)
    >>> fm[rs[0]+co[0,0] : rs[0]+co[0,1]].tolist()
    [0]
    >>> fm[rs[1]+co[1,1] : rs[1]+co[1,2]].tolist()
    [1]
    """
    if xx_bucket_width <= 0:
        raise ValueError(f"xx_bucket_width must be > 0, got {xx_bucket_width}")
    if yy_bucket_width <= 0:
        raise ValueError(f"yy_bucket_width must be > 0, got {yy_bucket_width}")

    xx_lo_u32, xx_hi_u32, yy_lo_u32, yy_hi_u32 = validate_boxes_2d(xx_lo, xx_hi, yy_lo, yy_hi)

    counts = np.zeros((n_xx_buckets, n_yy_buckets), dtype=np.int64)
    _count_cell_memberships_numba(
        xx_lo_u32, xx_hi_u32, yy_lo_u32, yy_hi_u32,
        np.int64(xx_bucket_width), np.int64(yy_bucket_width),
        np.int64(n_xx_buckets), np.int64(n_yy_buckets),
        counts,
    )

    row_starts, cell_offsets = _build_offsets(counts)

    total_members = int(row_starts[-1])
    flat_members = np.empty(total_members, dtype=np.int32)
    cursors = np.zeros((n_xx_buckets, n_yy_buckets), dtype=np.int64)
    _fill_memberships_numba(
        xx_lo_u32, xx_hi_u32, yy_lo_u32, yy_hi_u32,
        np.int64(xx_bucket_width), np.int64(yy_bucket_width),
        np.int64(n_xx_buckets), np.int64(n_yy_buckets),
        row_starts, cell_offsets, cursors, flat_members,
    )

    return row_starts, cell_offsets, flat_members


# ---------------------------------------------------------------------------
# Cell access helpers (Numba-compatible)
# ---------------------------------------------------------------------------


@numba.njit
def get_cell_range(
    row_starts: NDArray,
    cell_offsets: NDArray,
    xx_bucket_idx: int,
    yy_bucket_idx: int,
) -> tuple[int, int]:
    """Return (start, end) indices into flat_members for cell (xx_bucket_idx, yy_bucket_idx).

    row_starts, cell_offsets : output arrays from build_spatial_index_2d.
    xx_bucket_idx, yy_bucket_idx : integer bucket coordinates of the cell.

    Both start and end are absolute positions in flat_members::

        start = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx]
        end   = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx + 1]
    """
    start = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx]
    end = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx + 1]
    return start, end


@numba.njit
def get_cell_members(
    row_starts: NDArray,
    cell_offsets: NDArray,
    flat_members: NDArray,
    xx_bucket_idx: int,
    yy_bucket_idx: int,
) -> NDArray:
    """Return the flat_members slice for cell (xx_bucket_idx, yy_bucket_idx).

    row_starts, cell_offsets, flat_members : output arrays from build_spatial_index_2d.
    xx_bucket_idx, yy_bucket_idx : integer bucket coordinates of the cell.

    Equivalent to flat_members[start:end] where start, end come from
    get_cell_range(row_starts, cell_offsets, xx_bucket_idx, yy_bucket_idx).
    """
    start, end = get_cell_range(row_starts, cell_offsets, xx_bucket_idx, yy_bucket_idx)
    return flat_members[start:end]


# ---------------------------------------------------------------------------
# Per-box intersection counting
# ---------------------------------------------------------------------------


def _build_xx_index(
    first_xx_all: NDArray,
    n_xx: int,
) -> tuple[NDArray, NDArray]:
    """CSR index: xx-bucket → contiguous slice of box_order.

    first_xx_all : int64[N], first xx-bucket index of each box.
    n_xx         : total number of xx-buckets (including trailing empty ones).

    Returns
    -------
    xx_starts : int64[n_xx + 1]  — CSR row pointers into box_order
    box_order : int32[N]         — boxes sorted by first_xx_all (stable, O(n+k))
    """
    keys = first_xx_all.astype(np.intp)
    xx_counts = np.zeros(n_xx, dtype=np.uint32)
    count1D(keys, counts=xx_counts)
    box_order = argcountsort(keys, counts=xx_counts).astype(np.int32)
    xx_starts = np.zeros(n_xx + 1, dtype=np.int64)
    np.cumsum(xx_counts, out=xx_starts[1:])
    return xx_starts, box_order


@numba.njit(parallel=True)
def _count_intersections_first_coordinate_left_side_parallel(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    first_xx_all: NDArray,
    first_yy_all: NDArray,
    xx_starts: NDArray,
    box_order: NDArray,
    row_starts: NDArray,
    cell_offsets: NDArray,
    flat_members: NDArray,
    xx_bucket_width: np.int64,
    yy_bucket_width: np.int64,
    progress: ProgressBar | None = None,
) -> NDArray:
    """Count 2D box-box intersections; prange over xx-buckets.

    All array arguments are in sorted order (sorted by first_xx_bucket).
    box_order[idx] gives the original (unsorted) box index for sorted position idx.

    Returns counts[N] in sorted order; caller must unsort via counts[box_order] = counts_sorted.
    """
    n_boxes = len(xx_lo)
    n_xx_buckets = len(row_starts) - 1
    n_yy_buckets = cell_offsets.shape[1] - 1
    counts = np.zeros(n_boxes, dtype=np.int32)

    for bx in numba.prange(n_xx_buckets):
        n_in_bx = xx_starts[bx + 1] - xx_starts[bx]
        for idx in range(xx_starts[bx], xx_starts[bx + 1]):
            i = np.int64(box_order[idx])
            first_xx_bucket = np.int64(bx)
            last_xx_bucket  = np.int64(math.ceil(xx_hi[i] / xx_bucket_width)) - np.int64(1)
            first_yy_bucket = first_yy_all[i]
            last_yy_bucket  = np.int64(math.ceil(yy_hi[i] / yy_bucket_width)) - np.int64(1)

            last_xx_bucket  = min(last_xx_bucket,  np.int64(n_xx_buckets - 1))
            first_yy_bucket = max(first_yy_bucket, np.int64(0))
            last_yy_bucket  = min(last_yy_bucket,  np.int64(n_yy_buckets - 1))

            single_cell = (first_xx_bucket == last_xx_bucket and
                           first_yy_bucket == last_yy_bucket)

            local_count = np.int32(0)
            for xx_bucket_idx in range(first_xx_bucket, last_xx_bucket + 1):
                for yy_bucket_idx in range(first_yy_bucket, last_yy_bucket + 1):
                    start = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx]
                    end   = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx + 1]
                    for member_pos in range(start, end):
                        j = np.int64(flat_members[member_pos])
                        if j == i:
                            continue
                        if not single_cell:
                            canonical_bx = max(first_xx_bucket, first_xx_all[j])
                            canonical_by = max(first_yy_bucket, first_yy_all[j])
                            if xx_bucket_idx != canonical_bx or yy_bucket_idx != canonical_by:
                                continue
                        if (xx_lo[i] < xx_hi[j] and xx_lo[j] < xx_hi[i] and
                                yy_lo[i] < yy_hi[j] and yy_lo[j] < yy_hi[i]):
                            local_count += 1

            counts[i] = local_count

        if progress is not None:
            progress.update(n_in_bx)

    return counts


@numba.njit(parallel=True)
def _count_intersections_first_coordinate_left_side_parallel_zz(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    zz_lo: NDArray,
    zz_hi: NDArray,
    first_xx_all: NDArray,
    first_yy_all: NDArray,
    xx_starts: NDArray,
    box_order: NDArray,
    row_starts: NDArray,
    cell_offsets: NDArray,
    flat_members: NDArray,
    xx_bucket_width: np.int64,
    yy_bucket_width: np.int64,
    progress: ProgressBar | None = None,
) -> NDArray:
    """Count 2D box-box intersections with additional zz-axis overlap filter.

    Identical to _count_intersections_first_coordinate_left_side_parallel but a pair (i, j) is only
    counted when zz_lo[i] < zz_hi[j] and zz_lo[j] < zz_hi[i] as well.
    zz_lo, zz_hi : int64[N] in sorted order (same sort as xx/yy arrays).
    """
    n_boxes = len(xx_lo)
    n_xx_buckets = len(row_starts) - 1
    n_yy_buckets = cell_offsets.shape[1] - 1
    counts = np.zeros(n_boxes, dtype=np.int32)

    for bx in numba.prange(n_xx_buckets):
        n_in_bx = xx_starts[bx + 1] - xx_starts[bx]
        for idx in range(xx_starts[bx], xx_starts[bx + 1]):
            i = np.int64(box_order[idx])
            first_xx_bucket = np.int64(bx)
            last_xx_bucket  = np.int64(math.ceil(xx_hi[i] / xx_bucket_width)) - np.int64(1)
            first_yy_bucket = first_yy_all[i]
            last_yy_bucket  = np.int64(math.ceil(yy_hi[i] / yy_bucket_width)) - np.int64(1)

            last_xx_bucket  = min(last_xx_bucket,  np.int64(n_xx_buckets - 1))
            first_yy_bucket = max(first_yy_bucket, np.int64(0))
            last_yy_bucket  = min(last_yy_bucket,  np.int64(n_yy_buckets - 1))

            single_cell = (first_xx_bucket == last_xx_bucket and
                           first_yy_bucket == last_yy_bucket)

            local_count = np.int32(0)
            for xx_bucket_idx in range(first_xx_bucket, last_xx_bucket + 1):
                for yy_bucket_idx in range(first_yy_bucket, last_yy_bucket + 1):
                    start = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx]
                    end   = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx + 1]
                    for member_pos in range(start, end):
                        j = np.int64(flat_members[member_pos])
                        if j == i:
                            continue
                        if not single_cell:
                            canonical_bx = max(first_xx_bucket, first_xx_all[j])
                            canonical_by = max(first_yy_bucket, first_yy_all[j])
                            if xx_bucket_idx != canonical_bx or yy_bucket_idx != canonical_by:
                                continue
                        if (xx_lo[i] < xx_hi[j] and xx_lo[j] < xx_hi[i] and
                                yy_lo[i] < yy_hi[j] and yy_lo[j] < yy_hi[i] and
                                zz_lo[i] < zz_hi[j] and zz_lo[j] < zz_hi[i]):
                            local_count += 1

            counts[i] = local_count

        if progress is not None:
            progress.update(n_in_bx)

    return counts


def _setup_first_coordinate_left_side_sort(
    xx_lo_raw: NDArray,
    xx_hi_raw: NDArray,
    yy_lo_raw: NDArray,
    yy_hi_raw: NDArray,
    xx_factor: float,
    yy_factor: float,
):
    """Build the presorted spatial index; return all arrays needed by the kernels.

    Accepts both integer and float box bounds.  The spatial index structure is
    built from conservatively rounded integer bounds (floor lo, ceil hi) so
    every bucket touched by a float box is included.  The original float arrays
    are returned to the kernels so the intersection predicate uses exact values.
    """
    xx_lo_f = np.asarray(xx_lo_raw, dtype=np.float64)
    xx_hi_f = np.asarray(xx_hi_raw, dtype=np.float64)
    yy_lo_f = np.asarray(yy_lo_raw, dtype=np.float64)
    yy_hi_f = np.asarray(yy_hi_raw, dtype=np.float64)

    xx_widths = xx_hi_f - xx_lo_f
    yy_widths = yy_hi_f - yy_lo_f
    bw_xx, bw_yy = get_multiplied_median_bucket_widths(xx_widths, yy_widths, xx_factor, yy_factor)
    n_xx = int(xx_hi_f.max()) // bw_xx + 1
    n_yy = int(yy_hi_f.max()) // bw_yy + 1

    # first_xx/yy_all: floor division; clamp to 0 so negative lo boxes start at bucket 0.
    first_xx_all = np.maximum(np.floor(xx_lo_f / bw_xx), 0).astype(np.int64)
    first_yy_all = np.maximum(np.floor(yy_lo_f / bw_yy), 0).astype(np.int64)

    xx_starts, box_order = _build_xx_index(first_xx_all, n_xx)

    # Sort original float arrays by first_xx bucket.
    xx_s  = xx_lo_f[box_order];        xxh_s = xx_hi_f[box_order]
    yy_s  = yy_lo_f[box_order];        yyh_s = yy_hi_f[box_order]
    fxa_s = first_xx_all[box_order];   fya_s = first_yy_all[box_order]

    # Build the spatial index structure using conservatively rounded integer bounds
    # (floor lo, ceil hi) so every cell the float box might touch is registered.
    xx_lo_idx = np.maximum(np.floor(xx_lo_f), 0).astype(np.int64).astype(np.uint32)
    xx_hi_idx = np.ceil(xx_hi_f).astype(np.int64).astype(np.uint32)
    yy_lo_idx = np.maximum(np.floor(yy_lo_f), 0).astype(np.int64).astype(np.uint32)
    yy_hi_idx = np.ceil(yy_hi_f).astype(np.int64).astype(np.uint32)
    row_starts, cell_offsets, flat_members = build_spatial_index_2d(
        xx_lo_idx[box_order], xx_hi_idx[box_order],
        yy_lo_idx[box_order], yy_hi_idx[box_order],
        bw_xx, bw_yy, n_xx, n_yy,
    )

    box_order_id = np.arange(len(xx_lo_f), dtype=np.int32)
    return (
        xx_s, xxh_s, yy_s, yyh_s,
        fxa_s, fya_s,
        xx_starts, box_order_id,
        row_starts, cell_offsets, flat_members,
        np.int64(bw_xx), np.int64(bw_yy),
        box_order,
    )


def count_intersections_2d(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
    progress=None,
) -> NDArray:
    """Count per-box 2D intersections using a presorted spatial index.

    xx_lo, xx_hi, yy_lo, yy_hi : half-open [lo, hi) box bounds (any integer dtype).
    xx_factor, yy_factor : bucket-width multiplier (passed to get_multiplied_median_bucket_widths).

    Returns int32[N] — number of other boxes that overlap each box in (xx, yy).

    Uses the fb-sort strategy: boxes are presorted by first xx-bucket so that
    the prange over xx-buckets gives sequential i-side access and cache-hot
    cell_offsets rows.  Each pair (i, j) is counted exactly once via canonical-
    cell deduplication.
    """
    out = _setup_first_coordinate_left_side_sort(
        np.asarray(xx_lo), np.asarray(xx_hi),
        np.asarray(yy_lo), np.asarray(yy_hi),
        xx_factor, yy_factor,
    )
    *kernel_args, box_order = out
    counts_sorted = _count_intersections_first_coordinate_left_side_parallel(*kernel_args, progress)
    counts = np.empty_like(counts_sorted)
    counts[box_order] = counts_sorted
    return counts


def count_intersections_2d_zz(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    zz_lo: NDArray,
    zz_hi: NDArray,
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
    progress=None,
) -> NDArray:
    """Count per-box intersections using a 2D spatial index with a zz-axis filter.

    Same as count_intersections_2d but a pair (i, j) is only counted when the
    zz intervals also overlap: zz_lo[i] < zz_hi[j] and zz_lo[j] < zz_hi[i].

    xx_lo, xx_hi, yy_lo, yy_hi : half-open [lo, hi) 2D box bounds (any integer dtype).
    zz_lo, zz_hi               : half-open [lo, hi) bounds along the third axis (int64).
    xx_factor, yy_factor       : bucket-width multiplier for the 2D index.

    Returns int32[N].
    """
    xx_lo_raw = np.asarray(xx_lo)
    xx_hi_raw = np.asarray(xx_hi)
    yy_lo_raw = np.asarray(yy_lo)
    yy_hi_raw = np.asarray(yy_hi)
    out = _setup_first_coordinate_left_side_sort(xx_lo_raw, xx_hi_raw, yy_lo_raw, yy_hi_raw, xx_factor, yy_factor)
    *kernel_args, box_order = out

    zz_lo_s = np.asarray(zz_lo, dtype=np.int64)[box_order]
    zz_hi_s = np.asarray(zz_hi, dtype=np.int64)[box_order]

    # inject zz arrays after yy_hi (4th positional arg)
    xx_s, xxh_s, yy_s, yyh_s, *rest = kernel_args
    counts_sorted = _count_intersections_first_coordinate_left_side_parallel_zz(
        xx_s, xxh_s, yy_s, yyh_s, zz_lo_s, zz_hi_s, *rest, progress
    )
    counts = np.empty_like(counts_sorted)
    counts[box_order] = counts_sorted
    return counts


# ---------------------------------------------------------------------------
# Per-box neighbor listing (CSR adjacency)
# ---------------------------------------------------------------------------


@numba.njit(parallel=True)
def _fill_neighbors_first_coordinate_left_side_parallel_zz(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    zz_lo: NDArray,
    zz_hi: NDArray,
    first_xx_all: NDArray,
    first_yy_all: NDArray,
    xx_starts: NDArray,
    box_order: NDArray,   # identity permutation (box_order_id)
    sort_perm: NDArray,   # actual sort permutation: sort_perm[sorted_j] = orig_j
    row_starts: NDArray,
    cell_offsets: NDArray,
    flat_members: NDArray,
    xx_bucket_width: np.int64,
    yy_bucket_width: np.int64,
    offsets: NDArray,     # int64[N+1] CSR offsets in sorted-i space
    adj: NDArray,         # int64[M]   output: original-j neighbor indices
) -> None:
    """Fill CSR adjacency array for 2D+zz box intersections.

    Parallel twin of _fill_neighbors_first_coordinate_left_side_parallel_zz — same prange-over-bx
    structure as the count kernel.  Each sorted position idx is owned by
    exactly one bx thread → writes to adj[offsets[idx]:...] are race-free.

    sort_perm[sorted_j] = original_j — used to convert flat_members indices
    (sorted space) to original box indices stored in adj.
    offsets : int64[N+1] cumsum of counts_sorted (output of count pass).
    adj     : pre-allocated int64[M]; filled here in-place.
    """
    n_xx_buckets = len(row_starts) - 1
    n_yy_buckets = cell_offsets.shape[1] - 1

    for bx in numba.prange(n_xx_buckets):
        for idx in range(xx_starts[bx], xx_starts[bx + 1]):
            i = np.int64(box_order[idx])   # = idx with identity box_order
            first_xx_bucket = np.int64(bx)
            last_xx_bucket  = np.int64(math.ceil(xx_hi[i] / xx_bucket_width)) - np.int64(1)
            first_yy_bucket = first_yy_all[i]
            last_yy_bucket  = np.int64(math.ceil(yy_hi[i] / yy_bucket_width)) - np.int64(1)

            last_xx_bucket  = min(last_xx_bucket,  np.int64(n_xx_buckets - 1))
            first_yy_bucket = max(first_yy_bucket, np.int64(0))
            last_yy_bucket  = min(last_yy_bucket,  np.int64(n_yy_buckets - 1))

            single_cell = (first_xx_bucket == last_xx_bucket and
                           first_yy_bucket == last_yy_bucket)

            cursor = np.int64(0)
            for xx_bucket_idx in range(first_xx_bucket, last_xx_bucket + 1):
                for yy_bucket_idx in range(first_yy_bucket, last_yy_bucket + 1):
                    start = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx]
                    end   = row_starts[xx_bucket_idx] + cell_offsets[xx_bucket_idx, yy_bucket_idx + 1]
                    for member_pos in range(start, end):
                        j = np.int64(flat_members[member_pos])
                        if j == i:
                            continue
                        if not single_cell:
                            canonical_bx = max(first_xx_bucket, first_xx_all[j])
                            canonical_by = max(first_yy_bucket, first_yy_all[j])
                            if xx_bucket_idx != canonical_bx or yy_bucket_idx != canonical_by:
                                continue
                        if (xx_lo[i] < xx_hi[j] and xx_lo[j] < xx_hi[i] and
                                yy_lo[i] < yy_hi[j] and yy_lo[j] < yy_hi[i] and
                                zz_lo[i] < zz_hi[j] and zz_lo[j] < zz_hi[i]):
                            adj[offsets[i] + cursor] = np.int64(sort_perm[j])
                            cursor += 1


@numba.njit
def _unsort_adj(
    adj_sorted: NDArray,
    offsets_sorted: NDArray,
    offsets_orig: NDArray,
    box_order: NDArray,
    N: int,
    adj_orig: NDArray,
) -> None:
    """Scatter adj_sorted (sorted-index-space CSR) into adj_orig (original-index-space CSR).

    Iterates over sorted positions; for each sorted_i, copies
    adj_sorted[offsets_sorted[sorted_i]:offsets_sorted[sorted_i+1]] into
    adj_orig[offsets_orig[orig_i]:...] where orig_i = box_order[sorted_i].
    """
    for sorted_i in range(N):
        orig_i = np.int64(box_order[sorted_i])
        src = offsets_sorted[sorted_i]
        dst = offsets_orig[orig_i]
        n   = offsets_sorted[sorted_i + 1] - src
        for k in range(n):
            adj_orig[dst + k] = adj_sorted[src + k]


def find_neighbors_2d_zz(
    xx_lo: NDArray,
    xx_hi: NDArray,
    yy_lo: NDArray,
    yy_hi: NDArray,
    zz_lo: NDArray,
    zz_hi: NDArray,
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
) -> tuple[NDArray, NDArray]:
    """Return CSR adjacency (offsets, neighbors) of intersecting box pairs.

    A pair (i, j) is included when all three axes overlap:
      xx_lo[i] < xx_hi[j]  and  xx_lo[j] < xx_hi[i]  (half-open)
      yy_lo[i] < yy_hi[j]  and  yy_lo[j] < yy_hi[i]
      zz_lo[i] < zz_hi[j]  and  zz_lo[j] < zz_hi[i]

    xx_lo, xx_hi, yy_lo, yy_hi : 2D box bounds — any numeric dtype (int or float).
        The spatial index is built from conservatively rounded integer bounds;
        the intersection predicate uses the original values exactly.
    zz_lo, zz_hi               : third-axis bounds — same treatment.
    xx_factor, yy_factor       : passed to get_multiplied_median_bucket_widths.

    Returns
    -------
    offsets   : int64[N + 1]  — CSR row pointers in original box order
    neighbors : int64[M]      — flat neighbor indices (original box order)

    The adjacency is symmetric: j in neighbors[offsets[i]:offsets[i+1]]
    implies i in neighbors[offsets[j]:offsets[j+1]].
    """
    xx_lo_f = np.asarray(xx_lo, dtype=np.float64)
    xx_hi_f = np.asarray(xx_hi, dtype=np.float64)
    yy_lo_f = np.asarray(yy_lo, dtype=np.float64)
    yy_hi_f = np.asarray(yy_hi, dtype=np.float64)
    zz_lo_f = np.asarray(zz_lo, dtype=np.float64)
    zz_hi_f = np.asarray(zz_hi, dtype=np.float64)

    N = len(xx_lo_f)
    out = _setup_first_coordinate_left_side_sort(xx_lo_f, xx_hi_f, yy_lo_f, yy_hi_f, xx_factor, yy_factor)
    *kernel_args, box_order = out

    xx_s, xxh_s, yy_s, yyh_s, fxa_s, fya_s, xx_starts, box_order_id, \
        row_starts, cell_offsets, flat_members, bw_xx, bw_yy = kernel_args

    zz_lo_s = zz_lo_f[box_order]
    zz_hi_s = zz_hi_f[box_order]

    # --- count pass ---
    counts_sorted = _count_intersections_first_coordinate_left_side_parallel_zz(
        xx_s, xxh_s, yy_s, yyh_s, zz_lo_s, zz_hi_s,
        fxa_s, fya_s, xx_starts, box_order_id,
        row_starts, cell_offsets, flat_members,
        bw_xx, bw_yy,
    )
    offsets_sorted = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts_sorted, out=offsets_sorted[1:])
    M = int(offsets_sorted[-1])

    # --- fill pass ---
    adj_sorted = np.empty(M, dtype=np.int64)
    _fill_neighbors_first_coordinate_left_side_parallel_zz(
        xx_s, xxh_s, yy_s, yyh_s, zz_lo_s, zz_hi_s,
        fxa_s, fya_s, xx_starts, box_order_id,
        box_order,                                      # sort_perm: sorted_j → orig_j
        row_starts, cell_offsets, flat_members,
        bw_xx, bw_yy,
        offsets_sorted, adj_sorted,
    )

    # --- unsort to original index space ---
    counts_orig = np.empty(N, dtype=np.int32)
    counts_orig[box_order] = counts_sorted
    offsets_orig = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts_orig, out=offsets_orig[1:])

    adj_orig = np.empty(M, dtype=np.int64)
    _unsort_adj(adj_sorted, offsets_sorted, offsets_orig, box_order, N, adj_orig)

    return offsets_orig, adj_orig
