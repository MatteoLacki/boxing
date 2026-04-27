"""


> why we need those flat_first_{x,y}?                                                   

  When query box n overlaps cells (i0..i1) × (j0..j1), and neighbor k (stored in nb_box_idxs) also
  spans multiple cells, the pair (n, k) will be found once per shared cell — i.e. multiple times.

  The canonical-cell rule fires exactly once per pair: only process (n, k) in cell (i, j) if:

  i == max(first_x_of_n, first_x_of_k)
  j == max(first_y_of_n, first_y_of_k)

  first_x_of_n and first_y_of_n the query computes on the fly (it's just imin/jmin). But first_x_of_k
  and first_y_of_k — the first bucket of neighbor k — must be looked up. That's what flat_first_x[pos] /
   flat_first_y[pos] provide.




          col0    col1    col2    col3                                                                  
        +-------+-------+-------+-------+
   row0 |  [0]  |  [0]  |  [0]  |       |                                                               
        +-------+-------+-------+-------+                                                               
   row1 |  [0]  | [0,1] | [0,1] |  [1]  |                                                               
        +-------+-------+-------+-------+                                                               
   row2 |       |  [1]  |  [1]  |  [1]  |
        +-------+-------+-------+-------+                                                               
   
    Box 0: rows 0-1, cols 0-2                                                                           
    Box 1: rows 1-2, cols 1-3
    Overlap: (row1,col1) and (row1,col2)                                                                
           
  flat (row-major, each group = one cell):                                                              
   
   pos:   0    1    2    3    4    5    6    7    8    9   10   11                                      
        +----+----+----+----+----+----+----+----+----+----+----+----+
        | 0  | 0  | 0  | 0  | 0  | 1  | 0  | 1  | 1  | 1  | 1  | 1  |                                   
        +----+----+----+----+----+----+----+----+----+----+----+----+                                   
  cell:  (0,0)(0,1)(0,2)(1,0)  (1,1)   (1,2)  (1,3)(2,1)(2,2)(2,3)                                      
                                |_____|  |_____|                                                        
                                 2 members each                                                         
                                                                                                        
  cells (start offsets):                                                                                
   row0: [  0,  1,  2,  3 ]                                                                             
   row1: [  3,  4,  6,  8 ]                                                                             
   row2: [  9,  9, 10, 11 ]


"""
%load_ext autoreload
%autoreload 2

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import tomllib
from typing import Literal
import dictodot

import numba
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from numba_progress import ProgressBar
from numpy.typing import NDArray
from pandas_ops.io import read_df

from boxing.cli.precursor_neighbors import _load_config, _read_isolation_mz_radius_da
from boxing.spatial_index import dense_neighbors_to_csr, find_top_k_neighbors_2d_zz
from boxing.spatial_index import _build_offsets, _geometry_to_use_cylinder
from boxing.utils import argcountsort, count1D
from opentimspy import OpenTIMS

import matplotlib.pyplot as plt


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 5)
pd.set_option("display.max_rows", 25)


class Geometry(IntEnum):
    BOX = 0
    CYLINDER = 1
    ELLIPSOID = 2


def scales_to_width(scales: NDArray, factor: float = 2.0):
    """Bucket width = factor * 2 * median(scales), minimum 1."""
    median_width = 2 * np.median(scales)
    return max(1, int(factor * median_width))



@numba.njit
def minmax(xx: NDArray):
    """Return (min, max) of xx in a single pass."""
    _min = xx[0]
    _max = xx[0]
    for x in xx:
        _min = min(_min, x)
        _max = max(_max, x)
    return _min, _max


@numba.njit
def inplace_start_pos(xx):
    """Exclusive prefix sum of xx in-place. Returns total (sum of all elements)."""
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




def _stream_cells(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, cells, order, batch_size, processor, progress, *processor_args):
    """Iterate boxes in batches (spatially sorted via order), calling processor(n, i, j, imin, jmin, *processor_args).
    Parallel: each batch is a prange unit; sequential inner loop gives cache locality within a batch."""
    inv_xw = 1 / xwidth
    inv_yw = 1 / ywidth
    xhi = cells.shape[0]
    yhi = cells.shape[1] - 1
    n_boxes = len(order)
    n_batches = (n_boxes + batch_size - 1) // batch_size
    for batch in numba.prange(n_batches):
        start = batch * batch_size
        end = min(start + batch_size, n_boxes)
        for idx in range(start, end):
            n = order[idx]
            xc = xcenters[n]; yc = ycenters[n]
            xs = xscales[n];  ys = yscales[n]
            imin = lo_cell_idx(xc, xmult*xs, inv_xw)
            jmin = lo_cell_idx(yc, ymult*ys, inv_yw)
            itop = hi_cell_idx(xc, xmult*xs, inv_xw, xhi)
            jtop = hi_cell_idx(yc, ymult*ys, inv_yw, yhi)
            for i in range(imin, itop):
                for j in range(jmin, jtop):
                    processor(n, i, j, imin, jmin, *processor_args)
        if progress is not None:
            progress.update(end - start)

stream_cells = numba.njit(_stream_cells)
stream_cells_parallel = numba.njit(parallel=True)(_stream_cells)


@numba.njit
def _count_proc(n, i, j, imin, jmin, out):
    out[i, j] += 1


@numba.njit
def _fill_proc(n, i, j, imin, jmin, idx, flat, cursors):
    pos = idx[i, j] + cursors[i, j]
    flat[pos] = n
    cursors[i, j] += 1


@numba.njit
def _visit_proc(n, i, j, imin, jmin, cells, nb_box_idxs,
                idx_xcenters, idx_ycenters, idx_xscales, idx_yscales,
                xmult, ymult, xwidth, ywidth, processor, *processor_args):
    """Canonical-cell dedup then call inner processor(n, k, *processor_args) for each true neighbor k."""
    inv_xw = 1 / xwidth
    inv_yw = 1 / ywidth
    for pos in range(cells[i, j], cells[i, j + 1]):
        k = nb_box_idxs[pos]
        nb_lowest_x = lo_cell_idx(idx_xcenters[k], xmult*idx_xscales[k], inv_xw)
        nb_lowest_y = lo_cell_idx(idx_ycenters[k], ymult*idx_yscales[k], inv_yw)
        if i != max(imin, nb_lowest_x) or j != max(jmin, nb_lowest_y):
            continue
        processor(n, k, *processor_args)


@numba.njit
def count_2d_intersections(n, k, counts, xcenters, ycenters, xscales, yscales, xmult, ymult):
    """Increment counts[n] if boxes n and k truly overlap in 2D. Race-free: writes only to counts[n]."""
    xlo_n = xcenters[n] - xmult * xscales[n]; xhi_n = xcenters[n] + xmult * xscales[n]
    ylo_n = ycenters[n] - ymult * yscales[n]; yhi_n = ycenters[n] + ymult * yscales[n]
    xlo_k = xcenters[k] - xmult * xscales[k]; xhi_k = xcenters[k] + xmult * xscales[k]
    ylo_k = ycenters[k] - ymult * yscales[k]; yhi_k = ycenters[k] + ymult * yscales[k]
    if xlo_n < xhi_k and xlo_k < xhi_n and ylo_n < yhi_k and ylo_k < yhi_n:
        counts[n] += 1


@numba.njit
def _count_3d_box(n, k, counts, xcenters, ycenters, xscales, yscales, xmult, ymult, mz, mz_radius_da):
    if abs(mz[n] - mz[k]) >= 2.0 * mz_radius_da:
        return
    xlo_n = xcenters[n] - xmult * xscales[n]; xhi_n = xcenters[n] + xmult * xscales[n]
    ylo_n = ycenters[n] - ymult * yscales[n]; yhi_n = ycenters[n] + ymult * yscales[n]
    xlo_k = xcenters[k] - xmult * xscales[k]; xhi_k = xcenters[k] + xmult * xscales[k]
    ylo_k = ycenters[k] - ymult * yscales[k]; yhi_k = ycenters[k] + ymult * yscales[k]
    if not (xlo_n < xhi_k and xlo_k < xhi_n and ylo_n < yhi_k and ylo_k < yhi_n):
        return
    counts[n] += 1


@numba.njit
def _count_3d_cylinder(n, k, counts, xcenters, ycenters, xscales, yscales, xmult, ymult, mz, mz_radius_da):
    if abs(mz[n] - mz[k]) >= 2.0 * mz_radius_da:
        return
    xlo_n = xcenters[n] - xmult * xscales[n]; xhi_n = xcenters[n] + xmult * xscales[n]
    ylo_n = ycenters[n] - ymult * yscales[n]; yhi_n = ycenters[n] + ymult * yscales[n]
    xlo_k = xcenters[k] - xmult * xscales[k]; xhi_k = xcenters[k] + xmult * xscales[k]
    ylo_k = ycenters[k] - ymult * yscales[k]; yhi_k = ycenters[k] + ymult * yscales[k]
    if not (xlo_n < xhi_k and xlo_k < xhi_n and ylo_n < yhi_k and ylo_k < yhi_n):
        return
    rx = xmult * (xscales[n] + xscales[k]) * 0.5
    ry = ymult * (yscales[n] + yscales[k]) * 0.5
    dx = xcenters[n] - xcenters[k]
    dy = ycenters[n] - ycenters[k]
    if dx*dx/(rx*rx) + dy*dy/(ry*ry) > 1.0:
        return
    counts[n] += 1


@numba.njit
def _count_3d_ellipsoid(n, k, counts, xcenters, ycenters, xscales, yscales, xmult, ymult, mz, mz_radius_da):
    if abs(mz[n] - mz[k]) >= 2.0 * mz_radius_da:
        return
    xlo_n = xcenters[n] - xmult * xscales[n]; xhi_n = xcenters[n] + xmult * xscales[n]
    ylo_n = ycenters[n] - ymult * yscales[n]; yhi_n = ycenters[n] + ymult * yscales[n]
    xlo_k = xcenters[k] - xmult * xscales[k]; xhi_k = xcenters[k] + xmult * xscales[k]
    ylo_k = ycenters[k] - ymult * yscales[k]; yhi_k = ycenters[k] + ymult * yscales[k]
    if not (xlo_n < xhi_k and xlo_k < xhi_n and ylo_n < yhi_k and ylo_k < yhi_n):
        return
    rx = xmult * (xscales[n] + xscales[k]) * 0.5
    ry = ymult * (yscales[n] + yscales[k]) * 0.5
    dx = xcenters[n] - xcenters[k]
    dy = ycenters[n] - ycenters[k]
    if dx*dx/(rx*rx) + dy*dy/(ry*ry) > 1.0:
        return
    dz = mz[n] - mz[k]
    if dz*dz/(mz_radius_da*mz_radius_da) > 1.0:
        return
    counts[n] += 1


_COUNT_3D = {
    Geometry.BOX: _count_3d_box,
    Geometry.CYLINDER: _count_3d_cylinder,
    Geometry.ELLIPSOID: _count_3d_ellipsoid,
}


def count_3d_intersections(geometry: Geometry):
    """Return the branch-free specialized processor for the given geometry."""
    return _COUNT_3D[geometry]


def count_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, out, order, batch_size, progress=None):
    """Increment out[i,j] for every grid cell each box overlaps. First pass of CSR build."""
    stream_cells(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, out, order, batch_size, _count_proc, progress, out)


def fill_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, idx, flat, order, batch_size, progress=None):
    """Write array index into flat for every cell it overlaps. Second pass of CSR build."""
    cursors = np.zeros_like(idx)
    stream_cells(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, idx, order, batch_size, _fill_proc, progress, idx, flat, cursors)


@dataclass
class Boxes:
    """2D box parametrized by center +- scale * multiplier."""
    xcenters: NDArray
    ycenters: NDArray
    xscales: NDArray
    yscales: NDArray
    xmult: float
    ymult: float
    order: NDArray = field(init=False)

    def __post_init__(self):
        xlo = self.xcenters - self.xmult * self.xscales
        ylo = self.ycenters - self.ymult * self.yscales
        self.order = np.lexsort((ylo, xlo)).astype(np.int64)


@dataclass
class GridIndex:
    cells: NDArray
    nb_box_idxs: NDArray   # array indices (0..N-1) per cell membership
    boxes: Boxes
    xwidth: int
    ywidth: int

    @classmethod
    def from_boxes(cls, boxes: Boxes, batch_size: int = 64, progress=None):
        xwidth = scales_to_width(boxes.xscales)
        ywidth = scales_to_width(boxes.yscales)
        cells_cnt = lambda max_val, width: (int(max_val) + 1) // width + 1
        cells = np.zeros(
            (cells_cnt(boxes.xcenters.max(), xwidth), cells_cnt(boxes.ycenters.max(), ywidth) + 2),
            dtype=np.int64,
        )
        count_box_content(boxes.xcenters, boxes.ycenters, boxes.xscales, boxes.yscales, boxes.xmult, boxes.ymult, xwidth, ywidth, cells, boxes.order, batch_size, progress)
        total = inplace_start_pos(cells.ravel())
        nb_box_idxs = np.empty(total, dtype=np.int64)
        fill_box_content(boxes.xcenters, boxes.ycenters, boxes.xscales, boxes.yscales, boxes.xmult, boxes.ymult, xwidth, ywidth, cells, nb_box_idxs, boxes.order, batch_size, progress)
        return cls(cells, nb_box_idxs, boxes, xwidth, ywidth)

    def query(self, query_boxes: Boxes, processor, *processor_args, batch_size: int = 64, progress=None):
        """Call processor(n, k, *processor_args) for every (query n, indexed k) intersecting pair.
        Each pair visited exactly once via canonical-cell rule. Parallel: processor must write only to output[n]."""
        stream_cells_parallel(
            query_boxes.xcenters, query_boxes.ycenters, query_boxes.xscales, query_boxes.yscales,
            self.boxes.xmult, self.boxes.ymult, self.xwidth, self.ywidth,
            self.cells, query_boxes.order, batch_size, _visit_proc, progress,
            self.cells, self.nb_box_idxs,
            self.boxes.xcenters, self.boxes.ycenters, self.boxes.xscales, self.boxes.yscales,
            self.boxes.xmult, self.boxes.ymult, self.xwidth, self.ywidth,
            processor, *processor_args,
        )


@numba.njit
def _count_proc_3d(n, i, j, l, imin, jmin, lmin, out):
    out[i, j, l] += 1


@numba.njit
def _fill_proc_3d(n, i, j, l, imin, jmin, lmin, idx, flat, cursors):
    pos = idx[i, j, l] + cursors[i, j, l]
    flat[pos] = n
    cursors[i, j, l] += 1


def _stream_cells_3d(
    xcenters, ycenters, mz, xscales, yscales,
    mz_radius_da, mz_offset,
    xmult, ymult,
    xwidth, ywidth, zwidth,
    cells, order, batch_size,
    processor, progress, *processor_args,
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
            xc = xcenters[n]; yc = ycenters[n]; zc = mz[n]
            xs = xscales[n]; ys = yscales[n]
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
def _visit_proc_3d(n, i, j, l, imin, jmin, lmin,
                   cells, nb_box_idxs,
                   idx_xcenters, idx_ycenters, idx_mz,
                   idx_xscales, idx_yscales,
                   idx_mz_radius_da, idx_mz_offset,
                   xmult, ymult,
                   xwidth, ywidth, zwidth,
                   processor, *processor_args):
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
    """3D box: frame × scan × mz, parametrized by center ± scale * multiplier (xy) and mz_radius_da (z)."""
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
    cells: NDArray        # int64[BX, BY, BZ+1] CSR offsets
    nb_box_idxs: NDArray  # int64[M]
    boxes: Boxes3D
    xwidth: int
    ywidth: int
    zwidth: float         # = 2 * mz_radius_da

    @classmethod
    def from_boxes(cls, boxes: Boxes3D, batch_size: int = 64, progress=None):
        xwidth = scales_to_width(boxes.xscales)
        ywidth = scales_to_width(boxes.yscales)
        zwidth = 2.0 * boxes.mz_radius_da

        def cnt_xy(max_val, w): return (int(max_val) + 1) // w + 1
        bx = cnt_xy(int(boxes.xcenters.max()), xwidth)
        by = cnt_xy(int(boxes.ycenters.max()), ywidth)
        mz_hi_range = float((boxes.mz + boxes.mz_radius_da).max()) - boxes.mz_offset
        bz = int(mz_hi_range / zwidth) + 2

        cells = np.zeros((bx, by, bz + 1), dtype=np.int64)
        stream_cells_3d(
            boxes.xcenters, boxes.ycenters, boxes.mz, boxes.xscales, boxes.yscales,
            boxes.mz_radius_da, boxes.mz_offset, boxes.xmult, boxes.ymult,
            xwidth, ywidth, zwidth,
            cells, boxes.order, batch_size, _count_proc_3d, None, cells,
        )
        total = inplace_start_pos(cells.ravel())
        nb_box_idxs = np.empty(total, dtype=np.int64)
        cursors = np.zeros_like(cells)
        stream_cells_3d(
            boxes.xcenters, boxes.ycenters, boxes.mz, boxes.xscales, boxes.yscales,
            boxes.mz_radius_da, boxes.mz_offset, boxes.xmult, boxes.ymult,
            xwidth, ywidth, zwidth,
            cells, boxes.order, batch_size, _fill_proc_3d, None, cells, nb_box_idxs, cursors,
        )
        return cls(cells, nb_box_idxs, boxes, xwidth, ywidth, zwidth)

    def query(self, query_boxes: Boxes3D, processor, *processor_args, batch_size: int = 64, progress=None):
        """Call processor(n, k, *processor_args) for every (query n, indexed k) intersecting pair.
        Each pair visited exactly once via 3D canonical-cell rule."""
        stream_cells_3d_parallel(
            query_boxes.xcenters, query_boxes.ycenters, query_boxes.mz,
            query_boxes.xscales, query_boxes.yscales,
            query_boxes.mz_radius_da, query_boxes.mz_offset,
            self.boxes.xmult, self.boxes.ymult,
            self.xwidth, self.ywidth, self.zwidth,
            self.cells, query_boxes.order, batch_size,
            _visit_proc_3d, progress,
            self.cells, self.nb_box_idxs,
            self.boxes.xcenters, self.boxes.ycenters, self.boxes.mz,
            self.boxes.xscales, self.boxes.yscales,
            self.boxes.mz_radius_da, self.boxes.mz_offset,
            self.boxes.xmult, self.boxes.ymult,
            self.xwidth, self.ywidth, self.zwidth,
            processor, *processor_args,
        )


PLOT = False
if __name__ == "__main__":
    precursors_path = Path(
        "temp/F9477/optimal2tier/pre_sage_filtered_precursor_clusters.parquet"
    )
    dataset_path = Path("data/F9477.d")
    config_path = Path("configs/precursor_neighbors/default.toml")
    out_path = Path("/home/matteo/tmp/F9477nbs.mmappet")


    cfg = _load_config(config_path)
    prec = read_df(precursors_path)

    mz_radius_da = _read_isolation_mz_radius_da(OpenTIMS(dataset_path))

    data = dictodot.df2dd(
        prec[
            [
                "frame",
                "scan",
                "mz",
                "frame_scale",
                "scan_scale",
                "precursor_idx",
                "intensity",
            ]
        ]
    )

    frame_scan_boxes = Boxes(
        xcenters=data.frame,
        ycenters=data.scan,
        xscales=data.frame_scale,
        yscales=data.scan_scale,
        xmult=cfg.frame_mult,
        ymult=cfg.scan_mult,
    )
    N = len(data.frame)
    with ProgressBar(total=N, desc="build index") as progress:
        frame_scan_grid_idx = GridIndex.from_boxes(frame_scan_boxes, progress=progress)

    counts = np.zeros(N, dtype=np.int64)
    with ProgressBar(total=N, desc="count intersections") as progress:
        frame_scan_grid_idx.query(
            frame_scan_boxes, count_2d_intersections,
            counts, frame_scan_boxes.xcenters, frame_scan_boxes.ycenters, frame_scan_boxes.xscales, frame_scan_boxes.yscales, frame_scan_boxes.xmult, frame_scan_boxes.ymult,
            progress=progress,
        )
    print("2D intersection counts:", counts[:10])

    counts3d = np.zeros(N, dtype=np.int64)
    with ProgressBar(total=N, desc="count 3D intersections") as progress:
        frame_scan_grid_idx.query(
            frame_scan_boxes, count_3d_intersections(Geometry.BOX),
            counts3d, frame_scan_boxes.xcenters, frame_scan_boxes.ycenters, frame_scan_boxes.xscales, frame_scan_boxes.yscales, frame_scan_boxes.xmult, frame_scan_boxes.ymult,
            data.mz, mz_radius_da,
            progress=progress,
        )
    print("3D intersection counts:", counts3d[:10])

    frame_scan_mz_boxes = Boxes3D(
        xcenters=data.frame, ycenters=data.scan, mz=data.mz,
        xscales=data.frame_scale, yscales=data.scan_scale,
        mz_radius_da=mz_radius_da,
        xmult=cfg.frame_mult, ymult=cfg.scan_mult,
    )
    with ProgressBar(total=N, desc="build 3D index") as progress:
        frame_scan_mz_grid_idx = GridIndex3D.from_boxes(frame_scan_mz_boxes, progress=progress)

    counts3d_v2 = np.zeros(N, dtype=np.int64)
    with ProgressBar(total=N, desc="count intersections (3D grid)") as progress:
        frame_scan_mz_grid_idx.query(
            frame_scan_mz_boxes, count_3d_intersections(Geometry.BOX),
            counts3d_v2,
            frame_scan_mz_boxes.xcenters, frame_scan_mz_boxes.ycenters,
            frame_scan_mz_boxes.xscales, frame_scan_mz_boxes.yscales,
            frame_scan_mz_boxes.xmult, frame_scan_mz_boxes.ymult,
            data.mz, mz_radius_da,
            progress=progress,
        )
    print("3D grid counts:", counts3d_v2[:10])
    print("Match 2D-grid vs 3D-grid:", np.array_equal(counts3d, counts3d_v2))

    counts3dcyl = np.zeros(N, dtype=np.int64)
    with ProgressBar(total=N, desc="count 3D intersections") as progress:
        frame_scan_grid_idx.query(
            frame_scan_boxes, count_3d_intersections(Geometry.CYLINDER),
            counts3dcyl, frame_scan_boxes.xcenters, frame_scan_boxes.ycenters, frame_scan_boxes.xscales, frame_scan_boxes.yscales, frame_scan_boxes.xmult, frame_scan_boxes.ymult,
            data.mz, mz_radius_da,
            progress=progress,
        )
    print("3D cylinder intersection counts:", counts3dcyl[:10])

    from timstofu.stats import count1D
    from timstofu.plotting import plot_counts

    # plot_counts(count1D(counts), xlog=False, ylog=False)
    # The problem

    # F9477 -
    # F9468 -
    # B6699 - longer gradient

"""
In git/boxing/__dev/deslop2.py
"""
