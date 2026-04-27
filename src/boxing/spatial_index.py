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

from dataclasses import dataclass, field

import numpy as np
import numba
from numpy.typing import NDArray
from numba_progress import ProgressBar

from boxing.utils import count1D, argcountsort


# ---------------------------------------------------------------------------
# Helpers (pure Python / NumPy)
# ---------------------------------------------------------------------------


def get_multiplied_median_bucket_widths(
    xx_widths: NDArray,
    yy_widths: NDArray,
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
) -> tuple[int, int]:
    """Suggest bucket widths as factor * median box width, minimum 1.

    xx_widths, yy_widths : per-box widths.
    xx_factor, yy_factor : multiplier applied to the median width per axis
        (default 2.0 — one bucket ≈ two box-widths wide).

    >>> import numpy as np
    >>> get_multiplied_median_bucket_widths(np.array([10, 20, 30]), np.array([5, 10, 15]))
    (40, 20)
    """
    bw_xx = max(1, int(xx_factor * float(np.median(xx_widths))))
    bw_yy = max(1, int(yy_factor * float(np.median(yy_widths))))
    return bw_xx, bw_yy


def scales_to_width(scales: NDArray, factor: float = 2.0) -> int:
    """Bucket width = factor * 2 * median(scales), minimum 1."""
    median_width = 2 * np.median(scales)
    return max(1, int(factor * median_width))


@numba.njit
def inplace_start_pos(xx):
    """Exclusive prefix sum of xx in-place. Returns the total."""
    current = xx[0]
    xx[0] = 0
    for i in range(1, len(xx)):
        nxt = xx[i]
        xx[i] = current
        current += nxt
    return current


@numba.njit
def lo_cell_idx(center, radius, inverse_width, lowest_cell_idx=0):
    return max(int((center - radius) * inverse_width), lowest_cell_idx)


@numba.njit
def hi_cell_idx(center, radius, inverse_width, highest_cell_idx):
    return min(int((center + radius) * inverse_width) + 1, highest_cell_idx)


@numba.njit
def _count_proc_3d(n, i, j, l, imin, jmin, lmin, out):
    out[i, j, l] += 1


@numba.njit
def _fill_proc_3d(n, i, j, l, imin, jmin, lmin, idx, flat, cursors):
    pos = idx[i, j, l] + cursors[i, j, l]
    flat[pos] = n
    cursors[i, j, l] += 1


def _stream_cells_3d(
    xcenters,
    ycenters,
    mz,
    xscales,
    yscales,
    mz_radius_da,
    mz_offset,
    xmult,
    ymult,
    xwidth,
    ywidth,
    zwidth,
    cells,
    order,
    batch_size,
    processor,
    progress,
    *processor_args,
):
    inv_xw = 1.0 / xwidth
    inv_yw = 1.0 / ywidth
    inv_zw = 1.0 / zwidth
    xhi = cells.shape[0]
    yhi = cells.shape[1]
    zhi = cells.shape[2] - 1
    n_boxes = len(order)
    n_batches = (n_boxes + batch_size - 1) // batch_size
    for batch in numba.prange(n_batches):
        start = batch * batch_size
        end = min(start + batch_size, n_boxes)
        for box_idx in range(start, end):
            n = order[box_idx]
            xc = xcenters[n]
            yc = ycenters[n]
            zc = mz[n]
            xs = xscales[n]
            ys = yscales[n]
            imin = lo_cell_idx(xc, xmult * xs, inv_xw)
            jmin = lo_cell_idx(yc, ymult * ys, inv_yw)
            lmin = max(int((zc - mz_radius_da - mz_offset) * inv_zw), 0)
            itop = hi_cell_idx(xc, xmult * xs, inv_xw, xhi)
            jtop = hi_cell_idx(yc, ymult * ys, inv_yw, yhi)
            ltop = min(int((zc + mz_radius_da - mz_offset) * inv_zw) + 1, zhi)
            for i in range(imin, itop):
                for j in range(jmin, jtop):
                    for l in range(lmin, ltop):
                        processor(n, i, j, l, imin, jmin, lmin, *processor_args)
        if progress is not None:
            progress.update(end - start)


stream_cells_3d = numba.njit(_stream_cells_3d)
stream_cells_3d_parallel = numba.njit(parallel=True)(_stream_cells_3d)


@numba.njit
def _visit_proc_3d(
    n,
    i,
    j,
    l,
    imin,
    jmin,
    lmin,
    cells,
    nb_box_idxs,
    idx_xcenters,
    idx_ycenters,
    idx_mz,
    idx_xscales,
    idx_yscales,
    idx_mz_radius_da,
    idx_mz_offset,
    xmult,
    ymult,
    xwidth,
    ywidth,
    zwidth,
    processor,
    *processor_args,
):
    inv_xw = 1.0 / xwidth
    inv_yw = 1.0 / ywidth
    inv_zw = 1.0 / zwidth
    for pos in range(cells[i, j, l], cells[i, j, l + 1]):
        k = nb_box_idxs[pos]
        nb_lx = lo_cell_idx(idx_xcenters[k], xmult * idx_xscales[k], inv_xw)
        nb_ly = lo_cell_idx(idx_ycenters[k], ymult * idx_yscales[k], inv_yw)
        nb_lz = max(int((idx_mz[k] - idx_mz_radius_da - idx_mz_offset) * inv_zw), 0)
        if i != max(imin, nb_lx) or j != max(jmin, nb_ly) or l != max(lmin, nb_lz):
            continue
        processor(n, k, *processor_args)


@dataclass
class Boxes3D:
    """3D frame x scan x m/z boxes used by the C++ precursor-neighbor index."""

    xcenters: NDArray
    ycenters: NDArray
    mz: NDArray
    xscales: NDArray
    yscales: NDArray
    mz_radius_da: float
    xmult: float
    ymult: float
    order: NDArray = field(init=False)
    mz_offset: float = field(init=False)

    def __post_init__(self):
        xlo = self.xcenters - self.xmult * self.xscales
        ylo = self.ycenters - self.ymult * self.yscales
        mz_lo = self.mz - self.mz_radius_da
        self.mz_offset = float(mz_lo.min())
        self.order = np.lexsort((mz_lo, ylo, xlo)).astype(np.int64)


@dataclass
class GridIndex3D:
    cells: NDArray
    nb_box_idxs: NDArray
    boxes: Boxes3D
    xwidth: int
    ywidth: int
    zwidth: float

    @classmethod
    def from_boxes(cls, boxes: Boxes3D, batch_size: int = 64, progress=None):
        xwidth = scales_to_width(boxes.xscales)
        ywidth = scales_to_width(boxes.yscales)
        zwidth = 2.0 * boxes.mz_radius_da

        def cnt_xy(max_val, width):
            return (int(max_val) + 1) // width + 1

        bx = cnt_xy(int(boxes.xcenters.max()), xwidth)
        by = cnt_xy(int(boxes.ycenters.max()), ywidth)
        mz_hi_range = float((boxes.mz + boxes.mz_radius_da).max()) - boxes.mz_offset
        bz = int(mz_hi_range / zwidth) + 2

        cells = np.zeros((bx, by, bz + 1), dtype=np.int64)
        stream_cells_3d(
            boxes.xcenters,
            boxes.ycenters,
            boxes.mz,
            boxes.xscales,
            boxes.yscales,
            boxes.mz_radius_da,
            boxes.mz_offset,
            boxes.xmult,
            boxes.ymult,
            xwidth,
            ywidth,
            zwidth,
            cells,
            boxes.order,
            batch_size,
            _count_proc_3d,
            progress,
            cells,
        )
        total = inplace_start_pos(cells.ravel())
        nb_box_idxs = np.empty(total, dtype=np.int64)
        cursors = np.zeros_like(cells)
        stream_cells_3d(
            boxes.xcenters,
            boxes.ycenters,
            boxes.mz,
            boxes.xscales,
            boxes.yscales,
            boxes.mz_radius_da,
            boxes.mz_offset,
            boxes.xmult,
            boxes.ymult,
            xwidth,
            ywidth,
            zwidth,
            cells,
            boxes.order,
            batch_size,
            _fill_proc_3d,
            progress,
            cells,
            nb_box_idxs,
            cursors,
        )
        return cls(cells, nb_box_idxs, boxes, xwidth, ywidth, zwidth)

    def query(self, query_boxes: Boxes3D, processor, *processor_args, batch_size: int = 64, progress=None):
        """Call processor(n, k, *args) for every indexed 3D box pair exactly once."""
        stream_cells_3d_parallel(
            query_boxes.xcenters,
            query_boxes.ycenters,
            query_boxes.mz,
            query_boxes.xscales,
            query_boxes.yscales,
            query_boxes.mz_radius_da,
            query_boxes.mz_offset,
            self.boxes.xmult,
            self.boxes.ymult,
            self.xwidth,
            self.ywidth,
            self.zwidth,
            self.cells,
            query_boxes.order,
            batch_size,
            _visit_proc_3d,
            progress,
            self.cells,
            self.nb_box_idxs,
            self.boxes.xcenters,
            self.boxes.ycenters,
            self.boxes.mz,
            self.boxes.xscales,
            self.boxes.yscales,
            self.boxes.mz_radius_da,
            self.boxes.mz_offset,
            self.boxes.xmult,
            self.boxes.ymult,
            self.xwidth,
            self.ywidth,
            self.zwidth,
            processor,
            *processor_args,
        )


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
        xb = np.int64(boxes[i, 0])
        xe = np.int64(boxes[i, 1])
        yb = np.int64(boxes[i, 2])
        ye = np.int64(boxes[i, 3])
        first_xx_bucket = xb // xx_bucket_width
        last_xx_bucket = (xe - np.int64(1)) // xx_bucket_width
        first_yy_bucket = yb // yy_bucket_width
        last_yy_bucket = (ye - np.int64(1)) // yy_bucket_width

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
        xb = np.int64(boxes[i, 0])
        xe = np.int64(boxes[i, 1])
        yb = np.int64(boxes[i, 2])
        ye = np.int64(boxes[i, 3])
        first_xx_bucket = xb // xx_bucket_width
        last_xx_bucket = (xe - np.int64(1)) // xx_bucket_width
        first_yy_bucket = yb // yy_bucket_width
        last_yy_bucket = (ye - np.int64(1)) // yy_bucket_width

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
                pos = (
                    row_starts[xx_bucket_idx]
                    + cell_offsets[xx_bucket_idx, yy_bucket_idx]
                    + cursors[xx_bucket_idx, yy_bucket_idx]
                )
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
        np.int64(xx_bucket_width),
        np.int64(yy_bucket_width),
        np.int64(n_xx_buckets),
        np.int64(n_yy_buckets),
        counts,
    )

    row_starts, cell_offsets = _build_offsets(counts)

    total_members = int(row_starts[-1])
    flat_members = np.empty(total_members, dtype=np.int32)
    cursors = np.zeros((n_xx_buckets, n_yy_buckets), dtype=np.int64)
    _fill_memberships_numba(
        boxes,
        np.int64(xx_bucket_width),
        np.int64(yy_bucket_width),
        np.int64(n_xx_buckets),
        np.int64(n_yy_buckets),
        row_starts,
        cell_offsets,
        cursors,
        flat_members,
    )

    return row_starts, cell_offsets, flat_members


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


def _make_centered_xy_boxes_i64(
    centers: NDArray,
    scales: NDArray,
    xy_mults: tuple[float, float],
) -> NDArray:
    """Return conservative integer half-open x/y supports as int64."""
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers must have shape (N, 3), got {centers.shape}")
    if scales.shape != (len(centers), 2):
        raise ValueError(f"scales must have shape (N, 2), got {scales.shape}")
    if len(xy_mults) != 2:
        raise ValueError("xy_mults must contain exactly two values")

    rx = float(xy_mults[0]) * scales[:, 0]
    ry = float(xy_mults[1]) * scales[:, 1]
    if (rx < 0).any() or (ry < 0).any():
        raise ValueError("xy_mults * scales must be >= 0")

    x_lo = np.floor(centers[:, 0] - rx).clip(0).astype(np.int64)
    x_hi = np.ceil(centers[:, 0] + rx).astype(np.int64)
    y_lo = np.floor(centers[:, 1] - ry).clip(0).astype(np.int64)
    y_hi = np.ceil(centers[:, 1] + ry).astype(np.int64)
    return np.column_stack([x_lo, x_hi, y_lo, y_hi])


def _make_centered_zz_bounds(
    centers: NDArray,
    mz_radius_da: float,
) -> NDArray:
    """Return float64 half-open z supports."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers must have shape (N, 3), got {centers.shape}")
    if mz_radius_da < 0:
        raise ValueError("mz_radius_da must be >= 0")
    return np.column_stack(
        [
            centers[:, 2] - float(mz_radius_da),
            centers[:, 2] + float(mz_radius_da),
        ]
    )


@numba.njit(parallel=True)
def visit_box_intersections_2d_zz(
    boxes: NDArray,
    zz_bounds: NDArray,
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
    use_cylinder: bool = False,
    cylinder_radius: float = 1.0,
) -> None:
    """Visit every intersecting centered support pair and call processor(i, j, *processor_args).

    boxes is int64[N, 6] with columns [x_lo, x_hi, y_lo, y_hi, intensity, precursor_idx].
    zz_bounds is float64[N, 2] with columns [z_lo, z_hi], aligned to xy_rows.
    use_cylinder : when true, applies a normalised 2D cylinder cross-section
        filter in the indexed xx/yy plane.  The zz axis is still required to
        overlap, but it is not part of the cylinder distance.
    """
    MAX_R2 = cylinder_radius * cylinder_radius
    n_xx_buckets = len(row_starts) - 1
    n_yy_buckets = cell_offsets.shape[1] - 1

    for bx in numba.prange(n_xx_buckets):
        n_in_bx = xx_starts[bx + 1] - xx_starts[bx]
        for idx in range(xx_starts[bx], xx_starts[bx + 1]):
            i = np.int64(box_order[idx])
            xb = boxes[i, 0]
            xe = boxes[i, 1]
            yb = boxes[i, 2]
            ye = boxes[i, 3]
            zb = zz_bounds[i, 0]
            ze = zz_bounds[i, 1]

            first_xx_bucket = np.int64(bx)
            last_xx_bucket = (xe - np.int64(1)) // xx_bucket_width
            first_yy_bucket = first_yy_all[i]
            last_yy_bucket = (ye - np.int64(1)) // yy_bucket_width

            last_xx_bucket = min(last_xx_bucket, np.int64(n_xx_buckets - 1))
            first_yy_bucket = max(first_yy_bucket, np.int64(0))
            last_yy_bucket = min(last_yy_bucket, np.int64(n_yy_buckets - 1))

            single_cell = (
                first_xx_bucket == last_xx_bucket and first_yy_bucket == last_yy_bucket
            )

            for xx_bucket_idx in range(first_xx_bucket, last_xx_bucket + 1):
                for yy_bucket_idx in range(first_yy_bucket, last_yy_bucket + 1):
                    start = (
                        row_starts[xx_bucket_idx]
                        + cell_offsets[xx_bucket_idx, yy_bucket_idx]
                    )
                    end = (
                        row_starts[xx_bucket_idx]
                        + cell_offsets[xx_bucket_idx, yy_bucket_idx + 1]
                    )
                    for member_pos in range(start, end):
                        j = np.int64(flat_members[member_pos])
                        if j == i:
                            continue
                        if not single_cell:
                            canonical_bx = max(first_xx_bucket, first_xx_all[j])
                            canonical_by = max(first_yy_bucket, first_yy_all[j])
                            if (
                                xx_bucket_idx != canonical_bx
                                or yy_bucket_idx != canonical_by
                            ):
                                continue
                        xbj = boxes[j, 0]
                        xej = boxes[j, 1]
                        ybj = boxes[j, 2]
                        yej = boxes[j, 3]
                        zbj = zz_bounds[j, 0]
                        zej = zz_bounds[j, 1]
                        if (
                            xb < xej
                            and xbj < xe
                            and yb < yej
                            and ybj < ye
                            and zb < zej
                            and zbj < ze
                        ):
                            passes = True
                            if use_cylinder:
                                d_xx = (xb + xe - xbj - xej) / (xe - xb + xej - xbj)
                                d_yy = (yb + ye - ybj - yej) / (ye - yb + yej - ybj)
                                passes = d_xx * d_xx + d_yy * d_yy <= MAX_R2
                            if passes:
                                processor(i, j, *processor_args)

        if progress is not None:
            progress.update(n_in_bx)


def _setup_first_coordinate_left_side_sort(
    centers: NDArray,
    scales: NDArray,
    intensities: NDArray,
    precursor_idxs: NDArray,
    xy_mults: tuple[float, float],
    mz_radius_da: float,
    xx_factor: float,
    yy_factor: float,
):
    """Build the presorted x/y spatial index and aligned sorted payload arrays."""
    xy_boxes = _make_centered_xy_boxes_i64(centers, scales, xy_mults)
    zz_bounds = _make_centered_zz_bounds(centers, mz_radius_da)
    intensities = np.asarray(intensities, dtype=np.int64)
    precursor_idxs = np.asarray(precursor_idxs, dtype=np.int64)

    xx_lo = xy_boxes[:, 0]
    xx_hi = xy_boxes[:, 1]
    yy_lo = xy_boxes[:, 2]
    yy_hi = xy_boxes[:, 3]

    xx_widths = xx_hi - xx_lo
    yy_widths = yy_hi - yy_lo
    bw_xx, bw_yy = get_multiplied_median_bucket_widths(
        xx_widths, yy_widths, xx_factor, yy_factor
    )
    n_xx = int(xx_hi.max()) // bw_xx + 1
    n_yy = int(yy_hi.max()) // bw_yy + 1

    first_xx_all = (xx_lo // bw_xx).astype(np.int64)
    first_yy_all = (yy_lo // bw_yy).astype(np.int64)

    xx_starts, box_order = _build_xx_index(first_xx_all, n_xx)

    boxes = np.empty((len(xy_boxes), 6), dtype=np.int64)
    boxes[:, :4] = xy_boxes
    boxes[:, 4] = intensities
    boxes[:, 5] = precursor_idxs
    boxes_s = np.ascontiguousarray(boxes[box_order])
    zz_bounds_s = np.ascontiguousarray(zz_bounds[box_order])
    fxa_s = np.ascontiguousarray(first_xx_all[box_order])
    fya_s = np.ascontiguousarray(first_yy_all[box_order])

    boxes_idx = np.ascontiguousarray(boxes_s[:, :4].astype(np.int32))
    row_starts, cell_offsets, flat_members = build_spatial_index_2d(
        boxes_idx,
        bw_xx,
        bw_yy,
        n_xx,
        n_yy,
    )

    box_order_id = np.arange(len(boxes), dtype=np.int32)
    return (
        boxes_s,
        zz_bounds_s,
        fxa_s,
        fya_s,
        xx_starts,
        box_order_id,
        row_starts,
        cell_offsets,
        flat_members,
        np.int64(bw_xx),
        np.int64(bw_yy),
        box_order,
    )


def _geometry_to_use_cylinder(geometry: str) -> bool:
    if geometry == "box":
        return False
    if geometry == "cylinder":
        return True
    raise ValueError(f"geometry must be 'box' or 'cylinder', got {geometry!r}")


def _make_centered_boxes(
    centers: NDArray,
    scales: NDArray,
    xy_mults: tuple[float, float],
    mz_radius_da: float,
) -> NDArray:
    """Return endpoint boxes from centered 2D scale multipliers and mz radius.

    centers : float array (N, 3), columns [xx, yy, mz].
    scales  : float array (N, 2), columns [xx_scale, yy_scale].
    xy_mults: half-width multipliers for the first two dimensions.
    """
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers must have shape (N, 3), got {centers.shape}")
    if scales.shape != (len(centers), 2):
        raise ValueError(f"scales must have shape (N, 2), got {scales.shape}")
    if len(xy_mults) != 2:
        raise ValueError("xy_mults must contain exactly two values")
    rx = float(xy_mults[0]) * scales[:, 0]
    ry = float(xy_mults[1]) * scales[:, 1]
    if (rx < 0).any() or (ry < 0).any():
        raise ValueError("xy_mults * scales must be >= 0")
    if mz_radius_da < 0:
        raise ValueError("mz_radius_da must be >= 0")
    return np.column_stack(
        [
            centers[:, 0] - rx,
            centers[:, 0] + rx,
            centers[:, 1] - ry,
            centers[:, 1] + ry,
            centers[:, 2] - float(mz_radius_da),
            centers[:, 2] + float(mz_radius_da),
        ]
    )


# ---------------------------------------------------------------------------
# Per-box top-k intense neighbor listing
# ---------------------------------------------------------------------------


@numba.njit
def _top_k_neighbors_processor(
    i: np.int64,
    j: np.int64,
    neighbor_ids: NDArray,
    neighbor_ints: NDArray,
    boxes: NDArray,
    sort_perm: NDArray,
) -> None:
    """Insert neighbor j into the top-k table for box i (min-intensity eviction).

    neighbor_ids, neighbor_ints : int32/int64 [N, top_k] indexed by original box i.
    boxes        : int64[N, 6] in sorted order.
    sort_perm    : box_order (sorted → original box index) — used to address rows
                   in neighbor_ids/neighbor_ints.
    Zero is the empty-slot sentinel in neighbor_ints — neighbors with intensity
    zero are never inserted.  Safe for ion-count data where zero means no signal.
    """
    orig_i = np.int64(sort_perm[i])
    new_intensity = boxes[j, 4]
    if new_intensity == np.int64(0):
        return
    new_id = np.int32(boxes[j, 5])
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
    boxes: NDArray,
    sort_perm: NDArray,
    outer_zz_bounds: NDArray,
    inner_boxes: NDArray,
    inner_zz_bounds: NDArray,
    use_cylinder: bool,
    cylinder_radius: float,
) -> None:
    """Like _top_k_neighbors_processor but skips j when j intersects inner support of i.

    j is rejected when its outer support intersects i's inner support.
    """
    ixb = inner_boxes[i, 0]
    ixe = inner_boxes[i, 1]
    iyb = inner_boxes[i, 2]
    iye = inner_boxes[i, 3]
    izb = inner_zz_bounds[i, 0]
    ize = inner_zz_bounds[i, 1]

    jxb = boxes[j, 0]
    jxe = boxes[j, 1]
    jyb = boxes[j, 2]
    jye = boxes[j, 3]
    jzb = outer_zz_bounds[j, 0]
    jze = outer_zz_bounds[j, 1]

    if ixb < jxe and jxb < ixe and iyb < jye and jyb < iye and izb < jze and jzb < ize:
        reject = True
        if use_cylinder:
            d_xx = (ixb + ixe - jxb - jxe) / (ixe - ixb + jxe - jxb)
            d_yy = (iyb + iye - jyb - jye) / (iye - iyb + jye - jyb)
            reject = d_xx * d_xx + d_yy * d_yy <= cylinder_radius * cylinder_radius
        if reject:
            return
    orig_i = np.int64(sort_perm[i])
    new_intensity = boxes[j, 4]
    if new_intensity == np.int64(0):
        return
    new_id = np.int32(boxes[j, 5])
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
    centers: NDArray,
    scales: NDArray,
    mz_radius_da: float,
    intensities: NDArray,
    top_k: int,
    xy_mults: tuple[float, float] = (1.0, 1.0),
    xx_factor: float = 2.0,
    yy_factor: float = 2.0,
    progress=None,
    geometry: str = "box",
    cylinder_radius: float = 1.0,
    precursor_idxs: NDArray | None = None,
    inner_xy_mults: tuple[float, float] | None = None,
    inner_mz_radius_da: float | None = None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    use_cylinder = _geometry_to_use_cylinder(geometry)
    centers = np.asarray(centers, dtype=np.float64)
    N = len(centers)

    if precursor_idxs is None:
        precursor_idxs = np.arange(N, dtype=np.uint32)
        prec_to_row = None
    else:
        precursor_idxs = np.asarray(precursor_idxs, dtype=np.int64)
        prec_to_row = np.full(int(precursor_idxs.max()) + 1, -1, dtype=np.int32)
        prec_to_row[precursor_idxs] = np.arange(N, dtype=np.int32)

    out = _setup_first_coordinate_left_side_sort(
        centers,
        scales,
        intensities,
        precursor_idxs,
        xy_mults,
        mz_radius_da,
        xx_factor,
        yy_factor,
    )
    (
        boxes_s,
        zz_bounds_s,
        fxa_s,
        fya_s,
        xx_starts,
        box_order_id,
        row_starts,
        cell_offsets,
        flat_members,
        bw_xx,
        bw_yy,
        box_order,
    ) = out

    neighbor_ids = np.full((N, top_k), -1, dtype=np.int32)
    neighbor_ints = np.zeros((N, top_k), dtype=np.int64)

    use_inner_filter = False
    if inner_xy_mults is not None and (
        float(inner_xy_mults[0]) > 0 or float(inner_xy_mults[1]) > 0
    ):
        if inner_mz_radius_da is None:
            inner_mz_radius_da = mz_radius_da
        use_inner_filter = True

    if not use_inner_filter:
        proc = _top_k_neighbors_processor
        proc_args = (neighbor_ids, neighbor_ints, boxes_s, box_order)
    else:
        inner_xy_boxes = _make_centered_xy_boxes_i64(centers, scales, inner_xy_mults)[
            box_order
        ]
        inner_boxes = np.empty((N, 6), dtype=np.int64)
        inner_boxes[:, :4] = inner_xy_boxes
        inner_boxes[:, 4:] = 0
        inner_zz_bounds = _make_centered_zz_bounds(centers, inner_mz_radius_da)[
            box_order
        ]
        proc = _top_k_neighbors_shell_processor
        proc_args = (
            neighbor_ids,
            neighbor_ints,
            boxes_s,
            box_order,
            zz_bounds_s,
            inner_boxes,
            inner_zz_bounds,
            use_cylinder,
            cylinder_radius,
        )

    visit_box_intersections_2d_zz(
        boxes_s,
        zz_bounds_s,
        fxa_s,
        fya_s,
        xx_starts,
        box_order_id,
        row_starts,
        cell_offsets,
        flat_members,
        bw_xx,
        bw_yy,
        proc,
        proc_args,
        progress,
        use_cylinder,
        cylinder_radius,
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
        present = np.where(prec_to_row >= 0)[0]  # prec_idx values, ascending
        rows = prec_to_row[present]
        src_ids = neighbor_ids[rows]  # (P, K) in prec_idx order
        src_ints = neighbor_ints[rows] if neighbor_ints is not None else None
        valid = src_ids >= 0
        counts = valid.sum(axis=1).astype(np.int64)
        offsets = np.zeros(len(prec_to_row) + 1, dtype=np.int64)
        offsets[present + 1] = counts
        np.cumsum(offsets, out=offsets)
    else:
        src_ids = neighbor_ids
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
                prec_idx=neighbor_ids.dtype,
                intensity=neighbor_ints.dtype,
            )
        neighbors_ds = mmappet.open_new_dataset_dct(
            out_path / "neighbors.mmappet",
            scheme=neighbors_scheme,
            nrows=M,
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
        flat_ints = (
            np.empty(M, dtype=neighbor_ints.dtype)
            if neighbor_ints is not None
            else None
        )

    # ---- fill flat arrays -------------------------------------------------------
    flat_ids[:] = src_ids[valid]
    if neighbor_ints is not None:
        flat_ints[:] = src_ints[valid]
        return offsets, flat_ids, flat_ints
    return offsets, flat_ids
