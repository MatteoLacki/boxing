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
# Helpers (pure Python / NumPy)
# ---------------------------------------------------------------------------


def box_widths_2d(boxes: NDArray) -> tuple[NDArray, NDArray]:
    """Return per-box widths along xx and yy axes as int64.

    boxes : (N, 4) with columns [xx_lo, xx_hi, yy_lo, yy_hi].
    Casts to int64 before subtraction to avoid uint32 underflow.

    >>> import numpy as np
    >>> xw, yw = box_widths_2d(np.array([[10, 15, 3, 7]], dtype=np.uint32))
    >>> int(xw[0]), int(yw[0])
    (5, 4)
    """
    boxes = np.asarray(boxes)
    xx_widths = boxes[:, 1].astype(np.int64) - boxes[:, 0].astype(np.int64)
    yy_widths = boxes[:, 3].astype(np.int64) - boxes[:, 2].astype(np.int64)
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
    boxes: NDArray,
    xx_bucket_width: np.int64,
    yy_bucket_width: np.int64,
    n_xx_buckets: np.int64,
    n_yy_buckets: np.int64,
    counts_2d: NDArray,
) -> None:
    """Increment counts_2d[bx, by] for every (bx, by) cell each box overlaps.

    boxes : int32[N, 4] with columns [xx_lo, xx_hi, yy_lo, yy_hi].
    counts_2d : int64[BX, BY], modified in-place.

    Bucket assignment (half-open intervals):
        first_bucket = lo // width
        last_bucket  = (hi - 1) // width   (safe since hi > lo always)

    Bucket IDs are clamped to [0, n_*_buckets - 1].
    """
    for i in range(len(boxes)):
        xb = np.int64(boxes[i, 0]); xe = np.int64(boxes[i, 1])
        yb = np.int64(boxes[i, 2]); ye = np.int64(boxes[i, 3])
        first_xx_bucket = xb // xx_bucket_width
        last_xx_bucket  = (xe - np.int64(1)) // xx_bucket_width
        first_yy_bucket = yb // yy_bucket_width
        last_yy_bucket  = (ye - np.int64(1)) // yy_bucket_width

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
    boxes: NDArray,
    xx_bucket_width: int,
    yy_bucket_width: int,
    n_xx_buckets: int,
    n_yy_buckets: int,
) -> NDArray:
    """Return int64 count matrix of shape (n_xx_buckets, n_yy_buckets).

    boxes : (N, 4) with columns [xx_lo, xx_hi, yy_lo, yy_hi] (any dtype).
    xx_bucket_width, yy_bucket_width : bucket size along each axis (int, > 0).
    n_xx_buckets, n_yy_buckets : grid dimensions; out-of-range IDs are clamped.

    counts[bx, by] = number of boxes that overlap cell (bx, by).

    >>> import numpy as np
    >>> count_cell_memberships(np.array([[0, 3, 0, 3]], dtype=np.uint32), 2, 2, 3, 3)
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 0]])
    """
    boxes = np.ascontiguousarray(boxes, dtype=np.int32)
    counts = np.zeros((n_xx_buckets, n_yy_buckets), dtype=np.int64)
    _count_cell_memberships_numba(
        boxes,
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
    boxes: NDArray,
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

    boxes : int32[N, 4] with columns [xx_lo, xx_hi, yy_lo, yy_hi].
    cursors : int64[BX, BY] scratch array, zero-initialised before call.
    For each (bx, by) the write position is:
        pos = row_starts[bx] + cell_offsets[bx, by] + cursors[bx, by]
    cursors[bx, by] is post-incremented after each write.
    """
    for i in range(len(boxes)):
        xb = np.int64(boxes[i, 0]); xe = np.int64(boxes[i, 1])
        yb = np.int64(boxes[i, 2]); ye = np.int64(boxes[i, 3])
        first_xx_bucket = xb // xx_bucket_width
        last_xx_bucket  = (xe - np.int64(1)) // xx_bucket_width
        first_yy_bucket = yb // yy_bucket_width
        last_yy_bucket  = (ye - np.int64(1)) // yy_bucket_width

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
    boxes: NDArray,
    xx_bucket_width: int,
    yy_bucket_width: int,
    n_xx_buckets: int,
    n_yy_buckets: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build a 2D CSR-like spatial index for axis-aligned boxes.

    Parameters
    ----------
    boxes : (N, 4) array-like with columns [xx_lo, xx_hi, yy_lo, yy_hi].
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
        If bucket_width <= 0.

    >>> import numpy as np
    >>> boxes = np.array([[0, 3, 0, 3], [4, 7, 4, 7]], dtype=np.uint32)
    >>> rs, co, fm = build_spatial_index_2d(boxes, 4, 4, 2, 2)
    >>> fm[rs[0]+co[0,0] : rs[0]+co[0,1]].tolist()
    [0]
    >>> fm[rs[1]+co[1,1] : rs[1]+co[1,2]].tolist()
    [1]
    """
    if xx_bucket_width <= 0:
        raise ValueError(f"xx_bucket_width must be > 0, got {xx_bucket_width}")
    if yy_bucket_width <= 0:
        raise ValueError(f"yy_bucket_width must be > 0, got {yy_bucket_width}")

    boxes = np.ascontiguousarray(boxes, dtype=np.int32)

    counts = np.zeros((n_xx_buckets, n_yy_buckets), dtype=np.int64)
    _count_cell_memberships_numba(
        boxes,
        np.int64(xx_bucket_width), np.int64(yy_bucket_width),
        np.int64(n_xx_buckets), np.int64(n_yy_buckets),
        counts,
    )

    row_starts, cell_offsets = _build_offsets(counts)

    total_members = int(row_starts[-1])
    flat_members = np.empty(total_members, dtype=np.int32)
    cursors = np.zeros((n_xx_buckets, n_yy_buckets), dtype=np.int64)
    _fill_memberships_numba(
        boxes,
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


@numba.njit
def _count_processor(i: np.int64, j: np.int64, counts: NDArray) -> None:
    """Increment counts[i] for each valid pair."""
    counts[i] += np.int32(1)


@numba.njit
def _fill_neighbors_processor(
    i: np.int64,
    j: np.int64,
    adj: NDArray,
    offsets: NDArray,
    cursors: NDArray,
    sort_perm: NDArray,
) -> None:
    """Write sort_perm[j] (original-index j) into adj at the cursor for row i."""
    adj[offsets[i] + cursors[i]] = np.int64(sort_perm[j])
    cursors[i] += np.int64(1)


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
def visit_box_intersections_2d_zz(
    boxes: NDArray,
    first_xx_all: NDArray,
    first_yy_all: NDArray,
    xx_starts: NDArray,
    box_order: NDArray,
    row_starts: NDArray,
    cell_offsets: NDArray,
    flat_members: NDArray,
    xx_bucket_width: np.int64,
    yy_bucket_width: np.int64,
    processor,
    processor_args=(),
    progress: ProgressBar | None = None,
    ellipsoid_radius: float = math.inf,
) -> None:
    """Visit every intersecting (xx, yy, zz) box pair and call processor(i, j, *processor_args).

    boxes : float64[N, 6] sorted by first xx-bucket, columns [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi].
    A pair is visited only when all three axes overlap (half-open intervals).
    ellipsoid_radius : if finite, applies a normalised distance filter in
        the indexed xx/yy plane.  The zz axis is still required to overlap,
        but it is not part of the ellipsoid distance.
    """
    MAX_R2 = ellipsoid_radius * ellipsoid_radius
    n_xx_buckets = len(row_starts) - 1
    n_yy_buckets = cell_offsets.shape[1] - 1

    for bx in numba.prange(n_xx_buckets):
        n_in_bx = xx_starts[bx + 1] - xx_starts[bx]
        for idx in range(xx_starts[bx], xx_starts[bx + 1]):
            i = np.int64(box_order[idx])
            xb = boxes[i, 0]; xe = boxes[i, 1]
            yb = boxes[i, 2]; ye = boxes[i, 3]
            zb = boxes[i, 4]; ze = boxes[i, 5]

            first_xx_bucket = np.int64(bx)
            last_xx_bucket  = np.int64(math.ceil(xe / xx_bucket_width)) - np.int64(1)
            first_yy_bucket = first_yy_all[i]
            last_yy_bucket  = np.int64(math.ceil(ye / yy_bucket_width)) - np.int64(1)

            last_xx_bucket  = min(last_xx_bucket,  np.int64(n_xx_buckets - 1))
            first_yy_bucket = max(first_yy_bucket, np.int64(0))
            last_yy_bucket  = min(last_yy_bucket,  np.int64(n_yy_buckets - 1))

            single_cell = (first_xx_bucket == last_xx_bucket and
                           first_yy_bucket == last_yy_bucket)

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
                        xbj = boxes[j, 0]; xej = boxes[j, 1]
                        ybj = boxes[j, 2]; yej = boxes[j, 3]
                        zbj = boxes[j, 4]; zej = boxes[j, 5]
                        if (xb < xej and xbj < xe and
                                yb < yej and ybj < ye and
                                zb < zej and zbj < ze):
                            passes = True
                            if MAX_R2 < math.inf:
                                d_xx = (xb + xe - xbj - xej) / (xe - xb + xej - xbj)
                                d_yy = (yb + ye - ybj - yej) / (ye - yb + yej - ybj)
                                passes = d_xx * d_xx + d_yy * d_yy <= MAX_R2
                            if passes:
                                processor(i, j, *processor_args)

        if progress is not None:
            progress.update(n_in_bx)


def _setup_first_coordinate_left_side_sort(
    boxes: NDArray,
    xx_factor: float,
    yy_factor: float,
):
    """Build the presorted spatial index; return all arrays needed by the kernels.

    boxes : (N, 6) array [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi], any dtype.
    The spatial index is built from conservatively rounded integer bounds (floor lo,
    ceil hi) so every bucket touched by a float box is included.  The returned
    boxes_s is a C-contiguous float64 copy sorted by first xx-bucket.
    """
    boxes_f = np.asarray(boxes, dtype=np.float64)
    xx_lo_f = boxes_f[:, 0]
    xx_hi_f = boxes_f[:, 1]
    yy_lo_f = boxes_f[:, 2]
    yy_hi_f = boxes_f[:, 3]

    xx_widths = xx_hi_f - xx_lo_f
    yy_widths = yy_hi_f - yy_lo_f
    bw_xx, bw_yy = get_multiplied_median_bucket_widths(xx_widths, yy_widths, xx_factor, yy_factor)
    n_xx = int(xx_hi_f.max()) // bw_xx + 1
    n_yy = int(yy_hi_f.max()) // bw_yy + 1

    # first_xx/yy_all: floor division; clamp to 0 so negative lo boxes start at bucket 0.
    first_xx_all = np.maximum(np.floor(xx_lo_f / bw_xx), 0).astype(np.int64)
    first_yy_all = np.maximum(np.floor(yy_lo_f / bw_yy), 0).astype(np.int64)

    xx_starts, box_order = _build_xx_index(first_xx_all, n_xx)

    # Fancy-index produces a new C-contiguous (N, 6) float64 array sorted by first_xx.
    boxes_s = boxes_f[box_order]
    fxa_s = first_xx_all[box_order]
    fya_s = first_yy_all[box_order]

    # Build the spatial index from conservatively rounded integer bounds.
    xx_lo_idx = np.floor(xx_lo_f).clip(0).astype(np.int32)
    xx_hi_idx = np.ceil(xx_hi_f).astype(np.int32)
    yy_lo_idx = np.floor(yy_lo_f).clip(0).astype(np.int32)
    yy_hi_idx = np.ceil(yy_hi_f).astype(np.int32)
    boxes_idx = np.column_stack([
        xx_lo_idx[box_order], xx_hi_idx[box_order],
        yy_lo_idx[box_order], yy_hi_idx[box_order],
    ])
    row_starts, cell_offsets, flat_members = build_spatial_index_2d(
        boxes_idx, bw_xx, bw_yy, n_xx, n_yy,
    )

    box_order_id = np.arange(len(boxes_f), dtype=np.int32)
    return (
        boxes_s,
        fxa_s, fya_s,
        xx_starts, box_order_id,
        row_starts, cell_offsets, flat_members,
        np.int64(bw_xx), np.int64(bw_yy),
        box_order,
    )


def count_intersections_2d_zz(
    boxes: NDArray,
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
    progress=None,
    ellipsoid_radius: float = math.inf,
) -> NDArray:
    """Count per-box intersections using a 2D spatial index with a zz-axis filter.

    boxes : float or int array of shape (N, 6) with columns
        [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi].
        A pair (i, j) is counted only when all three axes overlap.
    xx_factor, yy_factor : bucket-width multiplier for the 2D index.
    ellipsoid_radius : if finite, restrict pairs by normalised centre-to-centre
        distance in the xx/yy plane.  The zz axis must still overlap, but is
        not part of the distance.  Default math.inf = full box.

    Returns int32[N].
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    out = _setup_first_coordinate_left_side_sort(boxes, xx_factor, yy_factor)
    boxes_s, fxa_s, fya_s, xx_starts, box_order_id, \
        row_starts, cell_offsets, flat_members, bw_xx, bw_yy, box_order = out

    n_boxes = len(box_order)
    counts_sorted = np.zeros(n_boxes, dtype=np.int32)
    visit_box_intersections_2d_zz(
        boxes_s, fxa_s, fya_s, xx_starts, box_order_id,
        row_starts, cell_offsets, flat_members, bw_xx, bw_yy,
        _count_processor, (counts_sorted,), progress, ellipsoid_radius,
    )
    counts = np.empty_like(counts_sorted)
    counts[box_order] = counts_sorted
    return counts


# ---------------------------------------------------------------------------
# Per-box neighbor listing (CSR adjacency)
# ---------------------------------------------------------------------------


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
    boxes: NDArray,
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
    ellipsoid_radius: float = math.inf,
) -> tuple[NDArray, NDArray]:
    """Return CSR adjacency (offsets, neighbors) of intersecting box pairs.

    boxes : array of shape (N, 6) with columns [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi].
        A pair (i, j) is included when all three axes overlap (half-open intervals).
        The spatial index is built from conservatively rounded integer bounds;
        the intersection predicate uses the original values exactly.
    xx_factor, yy_factor : passed to get_multiplied_median_bucket_widths.
    ellipsoid_radius : if finite, restrict pairs by normalised centre-to-centre
        distance in the xx/yy plane.  The zz axis must still overlap, but is
        not part of the distance.  Default math.inf = full box.

    Returns
    -------
    offsets   : int64[N + 1]  — CSR row pointers in original box order
    neighbors : int64[M]      — flat neighbor indices (original box order)

    The adjacency is symmetric: j in neighbors[offsets[i]:offsets[i+1]]
    implies i in neighbors[offsets[j]:offsets[j+1]].
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    N = len(boxes)
    out = _setup_first_coordinate_left_side_sort(boxes, xx_factor, yy_factor)
    boxes_s, fxa_s, fya_s, xx_starts, box_order_id, \
        row_starts, cell_offsets, flat_members, bw_xx, bw_yy, box_order = out

    _zz_kernel_args = (boxes_s, fxa_s, fya_s, xx_starts, box_order_id,
                       row_starts, cell_offsets, flat_members, bw_xx, bw_yy)

    # --- count pass ---
    counts_sorted = np.zeros(N, dtype=np.int32)
    visit_box_intersections_2d_zz(
        *_zz_kernel_args, _count_processor, (counts_sorted,), None, ellipsoid_radius,
    )
    offsets_sorted = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts_sorted, out=offsets_sorted[1:])
    M = int(offsets_sorted[-1])

    # --- fill pass ---
    adj_sorted = np.empty(M, dtype=np.int64)
    cursors = np.zeros(N, dtype=np.int64)
    visit_box_intersections_2d_zz(
        *_zz_kernel_args,
        _fill_neighbors_processor,
        (adj_sorted, offsets_sorted, cursors, box_order),  # sort_perm: sorted_j → orig_j
        None, ellipsoid_radius,
    )

    # --- unsort to original index space ---
    counts_orig = np.empty(N, dtype=np.int32)
    counts_orig[box_order] = counts_sorted
    offsets_orig = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts_orig, out=offsets_orig[1:])

    adj_orig = np.empty(M, dtype=np.int64)
    _unsort_adj(adj_sorted, offsets_sorted, offsets_orig, box_order, N, adj_orig)

    return offsets_orig, adj_orig


# ---------------------------------------------------------------------------
# Per-box top-k intense neighbor listing
# ---------------------------------------------------------------------------


@numba.njit
def _top_k_neighbors_processor(
    i: np.int64,
    j: np.int64,
    neighbor_ids: NDArray,
    neighbor_ints: NDArray,
    intensities: NDArray,
    sort_perm: NDArray,
    prec_idxs_s: NDArray,
) -> None:
    """Insert neighbor j into the top-k table for box i (min-intensity eviction).

    neighbor_ids, neighbor_ints : int32/int64 [N, top_k] indexed by original box i.
    intensities  : int64[N] in sorted order.
    sort_perm    : box_order (sorted → original box index) — used to address rows
                   in neighbor_ids/neighbor_ints.
    prec_idxs_s  : int32[N] sorted by box_order; maps sorted j → precursor index.
                   When no precursor remapping is needed, pass box_order itself so
                   prec_idxs_s[j] == original box index.

    Zero is the empty-slot sentinel in neighbor_ints — neighbors with intensity
    zero are never inserted.  Safe for ion-count data where zero means no signal.
    """
    orig_i = np.int64(sort_perm[i])
    new_intensity = intensities[j]
    if new_intensity == np.int64(0):
        return
    new_id = np.int32(prec_idxs_s[j])
    top_k = neighbor_ids.shape[1]

    min_intensity = neighbor_ints[orig_i, 0]
    min_slot = np.int64(0)
    for slot in range(1, top_k):
        v = neighbor_ints[orig_i, slot]
        if v < min_intensity:
            min_intensity = v
            min_slot = np.int64(slot)

    if new_intensity > min_intensity:
        neighbor_ids[orig_i, min_slot] = new_id
        neighbor_ints[orig_i, min_slot] = new_intensity


@numba.njit
def _top_k_neighbors_shell_processor(
    i: np.int64,
    j: np.int64,
    neighbor_ids: NDArray,
    neighbor_ints: NDArray,
    intensities: NDArray,
    sort_perm: NDArray,
    prec_idxs_s: NDArray,
    boxes_s: NDArray,
    inner_boxes_s: NDArray,
) -> None:
    """Like _top_k_neighbors_processor but skips j when j intersects inner box of i.

    boxes_s      : float64[N, 6] sorted outer boxes (same space as i, j indices).
    inner_boxes_s: float64[N, 6] sorted inner boxes, aligned to boxes_s via box_order.
    j is rejected (not added to top-k) when its outer box overlaps inner_boxes_s[i]
    on all three axes — i.e. j lies inside the inner shell of i.
    """
    if (inner_boxes_s[i, 0] < boxes_s[j, 1] and boxes_s[j, 0] < inner_boxes_s[i, 1] and
            inner_boxes_s[i, 2] < boxes_s[j, 3] and boxes_s[j, 2] < inner_boxes_s[i, 3] and
            inner_boxes_s[i, 4] < boxes_s[j, 5] and boxes_s[j, 4] < inner_boxes_s[i, 5]):
        return
    orig_i = np.int64(sort_perm[i])
    new_intensity = intensities[j]
    if new_intensity == np.int64(0):
        return
    new_id = np.int32(prec_idxs_s[j])
    top_k = neighbor_ids.shape[1]

    min_intensity = neighbor_ints[orig_i, 0]
    min_slot = np.int64(0)
    for slot in range(1, top_k):
        v = neighbor_ints[orig_i, slot]
        if v < min_intensity:
            min_intensity = v
            min_slot = np.int64(slot)

    if new_intensity > min_intensity:
        neighbor_ids[orig_i, min_slot] = new_id
        neighbor_ints[orig_i, min_slot] = new_intensity


def find_top_k_neighbors_2d_zz(
    boxes: NDArray,
    intensities: NDArray,
    top_k: int,
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
    progress=None,
    ellipsoid_radius: float = math.inf,
    precursor_idxs: NDArray | None = None,
    inner_boxes: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Return the top-k most intense neighbors per box, using a 2D index + zz filter.

    boxes : array of shape (N, 6) with columns [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi].
        A pair (i, j) is a candidate when all three axes overlap.
    intensities : int64-compatible array of length N — intensity of each box in
        original order.  Zero-intensity boxes are never recorded as neighbors.
    ellipsoid_radius : if finite, restrict pairs by normalised centre-to-centre
        distance in the xx/yy plane.  The zz axis must still overlap, but is
        not part of the distance.  Default math.inf = full box.
    precursor_idxs : int32-compatible array of length N or None.
        Maps box index → precursor index.  When provided, neighbor_ids entries
        contain precursor indices rather than box indices.  When None, defaults
        to np.arange(N) so neighbor_ids contains box indices (original behavior).

    Returns
    -------
    neighbor_ids  : int32[N, top_k]  — precursor (or box) ids of the top-k
        neighbors; unused slots contain -1.
    neighbor_ints : int64[N, top_k]  — corresponding intensities; unused slots
        contain 0.

    Both arrays are in original box order (row i = box i as passed in).
    Result is not sorted within each row; sort by neighbor_ints[i] descending
    if order matters.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    N = len(boxes)
    out = _setup_first_coordinate_left_side_sort(boxes, xx_factor, yy_factor)
    boxes_s, fxa_s, fya_s, xx_starts, box_order_id, \
        row_starts, cell_offsets, flat_members, bw_xx, bw_yy, box_order = out

    intensities_s = np.asarray(intensities, dtype=np.int64)[box_order]

    if precursor_idxs is None:
        precursor_idxs = np.arange(N, dtype=np.int32)
        prec_to_row = None
    else:
        precursor_idxs = np.asarray(precursor_idxs, dtype=np.int32)
        prec_to_row = np.full(int(precursor_idxs.max()) + 1, -1, dtype=np.int32)
        prec_to_row[precursor_idxs] = np.arange(N, dtype=np.int32)
    precursor_idxs_s = precursor_idxs[box_order]

    neighbor_ids  = np.full((N, top_k), -1, dtype=np.int32)
    neighbor_ints = np.zeros((N, top_k), dtype=np.int64)

    if inner_boxes is None:
        proc = _top_k_neighbors_processor
        proc_args = (neighbor_ids, neighbor_ints, intensities_s, box_order, precursor_idxs_s)
    else:
        inner_boxes_s = np.asarray(inner_boxes, dtype=np.float64)[box_order]
        proc = _top_k_neighbors_shell_processor
        proc_args = (
            neighbor_ids, neighbor_ints, intensities_s, box_order, precursor_idxs_s,
            boxes_s, inner_boxes_s,
        )

    visit_box_intersections_2d_zz(
        boxes_s, fxa_s, fya_s, xx_starts, box_order_id,
        row_starts, cell_offsets, flat_members, bw_xx, bw_yy,
        proc, proc_args, progress, ellipsoid_radius,
    )

    return neighbor_ids, neighbor_ints, prec_to_row


def dense_neighbors_to_csr(
    neighbor_ids: NDArray,
    neighbor_ints: NDArray | None = None,
    prec_to_row: NDArray | None = None,
    out_path=None,
):
    """Convert dense (N, K) top-k neighbor arrays to CSR format, skipping empty slots.

    neighbor_ids  : int32[N, K]  — neighbor (or precursor) indices; -1 = empty slot.
    neighbor_ints : int64[N, K] or None — corresponding intensities.
    prec_to_row   : int32 array returned by find_top_k_neighbors_2d_zz when
        precursor_idxs is provided.  prec_to_row[prec_idx] = row in neighbor_ids
        for that precursor (-1 if absent).  When provided, offsets has size
        len(prec_to_row)+1 and is directly indexed by precursor_idx:
            offsets[prec_idx]:offsets[prec_idx+1]  →  neighbors of prec_idx.
        When None, offsets[i]:offsets[i+1] gives the neighbors of box i.
    out_path      : path-like or None.  When provided the path must not exist;
        it is created and two mmappet datasets are written inside:
            neighbors.mmappet  — columns prec_idx (int32) [+ intensity (int64)]
            index.mmappet      — column offset (int64)
        The returned flat_ids / flat_ints are the memory-mapped arrays from
        those files (already filled), not in-RAM copies.

    Returns
    -------
    offsets  : int64[len(prec_to_row)+1 or N+1]
    flat_ids : int32[M]
    flat_ints: int64[M]  — only when neighbor_ints given
    """
    # ---- compute source rows and offsets ----------------------------------------
    if prec_to_row is not None:
        present = np.where(prec_to_row >= 0)[0]   # prec_idx values, ascending
        rows = prec_to_row[present]
        src_ids  = neighbor_ids[rows]              # (P, K) in prec_idx order
        src_ints = neighbor_ints[rows] if neighbor_ints is not None else None
        valid = src_ids >= 0
        counts = valid.sum(axis=1).astype(np.int64)
        offsets = np.zeros(len(prec_to_row) + 1, dtype=np.int64)
        offsets[present + 1] = counts
        np.cumsum(offsets, out=offsets)
    else:
        src_ids  = neighbor_ids
        src_ints = neighbor_ints
        valid = neighbor_ids >= 0
        counts = valid.sum(axis=1).astype(np.int64)
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

    M = int(offsets[-1])

    # ---- allocate output buffers ------------------------------------------------
    if out_path is not None:
        import mmappet
        from pathlib import Path as _Path
        out_path = _Path(out_path)
        if out_path.exists():
            raise ValueError(f"out_path already exists: {out_path}")
        out_path.mkdir(parents=True)

        neighbors_scheme = mmappet.get_schema(prec_idx=neighbor_ids.dtype)
        if neighbor_ints is not None:
            neighbors_scheme = mmappet.get_schema(
                prec_idx=neighbor_ids.dtype, intensity=neighbor_ints.dtype,
            )
        neighbors_ds = mmappet.open_new_dataset_dct(
            out_path / "neighbors.mmappet", scheme=neighbors_scheme, nrows=M,
        )
        flat_ids = neighbors_ds["prec_idx"]
        flat_ints = neighbors_ds["intensity"] if neighbor_ints is not None else None

        index_ds = mmappet.open_new_dataset_dct(
            out_path / "index.mmappet",
            scheme=mmappet.get_schema(offset=np.int64),
            nrows=len(offsets),
        )
        index_ds["offset"][:] = offsets
    else:
        flat_ids = np.empty(M, dtype=neighbor_ids.dtype)
        flat_ints = np.empty(M, dtype=neighbor_ints.dtype) if neighbor_ints is not None else None

    # ---- fill flat arrays -------------------------------------------------------
    flat_ids[:] = src_ids[valid]
    if neighbor_ints is not None:
        flat_ints[:] = src_ints[valid]
        return offsets, flat_ids, flat_ints
    return offsets, flat_ids
