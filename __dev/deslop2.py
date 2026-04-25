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
from dataclasses import dataclass
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


def _stream_cells(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, cells, processor, progress, *processor_args):
    """Iterate all boxes and their overlapping cells, calling processor(n, i, j, imin, jmin, *processor_args).
    Pass progress=None to skip updates."""
    inv_xw = 1 / xwidth
    inv_yw = 1 / ywidth
    xhi = cells.shape[0]
    yhi = cells.shape[1] - 1
    for n in numba.prange(len(xcenters)):
        xc = xcenters[n]; yc = ycenters[n]
        xs = xscales[n];  ys = yscales[n]
        imin = lo_cell_idx(xc, xmult*xs, inv_xw)
        jmin = lo_cell_idx(yc, ymult*ys, inv_yw)
        itop = hi_cell_idx(xc, xmult*xs, inv_xw, xhi)
        jtop = hi_cell_idx(yc, ymult*ys, inv_yw, yhi)
        for i in range(imin, itop):
            for j in range(jmin, jtop):
                processor(n, i, j, imin, jmin, *processor_args)
        if progress is not None and n % 1000 == 0:
            progress.update(1000)

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
def count_3d_intersections(n, k, counts, xcenters, ycenters, xscales, yscales, xmult, ymult, mz, mz_radius_da):
    """count_2d_intersections + mz box overlap gate. Race-free: writes only to counts[n]."""
    if abs(mz[n] - mz[k]) >= 2.0 * mz_radius_da:
        return
    xlo_n = xcenters[n] - xmult * xscales[n]; xhi_n = xcenters[n] + xmult * xscales[n]
    ylo_n = ycenters[n] - ymult * yscales[n]; yhi_n = ycenters[n] + ymult * yscales[n]
    xlo_k = xcenters[k] - xmult * xscales[k]; xhi_k = xcenters[k] + xmult * xscales[k]
    ylo_k = ycenters[k] - ymult * yscales[k]; yhi_k = ycenters[k] + ymult * yscales[k]
    if xlo_n < xhi_k and xlo_k < xhi_n and ylo_n < yhi_k and ylo_k < yhi_n:
        counts[n] += 1


@numba.njit(boundscheck=True)
def count_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, out, progress=None):
    """Increment out[i,j] for every grid cell each box overlaps. First pass of CSR build."""
    stream_cells(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, out, _count_proc, progress, out)


@numba.njit(boundscheck=True)
def fill_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, idx, flat, progress=None):
    """Write array index into flat for every cell it overlaps. Second pass of CSR build."""
    cursors = np.zeros_like(idx)
    stream_cells(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, idx, _fill_proc, progress, idx, flat, cursors)


@dataclass
class Boxes:
    """
    2D box parametrized by center +- scale * multiplier.
    """
    xcenters: NDArray
    ycenters: NDArray
    xscales: NDArray
    yscales: NDArray
    xmult: float
    ymult: float


@dataclass
class GridIndex:
    cells: NDArray
    nb_box_idxs: NDArray   # array indices (0..N-1) per cell membership
    boxes: Boxes
    xwidth: int
    ywidth: int

    @classmethod
    def from_boxes(cls, boxes: Boxes, progress=None):
        xwidth = scales_to_width(boxes.xscales)
        ywidth = scales_to_width(boxes.yscales)
        cells_cnt = lambda max_val, width: (int(max_val) + 1) // width + 1
        cells = np.zeros(
            (cells_cnt(boxes.xcenters.max(), xwidth), cells_cnt(boxes.ycenters.max(), ywidth) + 2),
            dtype=np.int64,
        )
        count_box_content(boxes.xcenters, boxes.ycenters, boxes.xscales, boxes.yscales, boxes.xmult, boxes.ymult, xwidth, ywidth, cells, progress)
        total = inplace_start_pos(cells.ravel())
        nb_box_idxs = np.empty(total, dtype=np.int64)
        fill_box_content(boxes.xcenters, boxes.ycenters, boxes.xscales, boxes.yscales, boxes.xmult, boxes.ymult, xwidth, ywidth, cells, nb_box_idxs, progress)
        return cls(cells, nb_box_idxs, boxes, xwidth, ywidth)

    def query(self, query_boxes: Boxes, processor, *processor_args, progress=None):
        """Call processor(n, k, *processor_args) for every (query n, indexed k) intersecting pair.
        Each pair visited exactly once via canonical-cell rule. Parallel: processor must write only to output[n]."""
        stream_cells_parallel(
            query_boxes.xcenters, query_boxes.ycenters, query_boxes.xscales, query_boxes.yscales,
            self.boxes.xmult, self.boxes.ymult, self.xwidth, self.ywidth,
            self.cells, _visit_proc, progress,
            self.cells, self.nb_box_idxs,
            self.boxes.xcenters, self.boxes.ycenters, self.boxes.xscales, self.boxes.yscales,
            self.boxes.xmult, self.boxes.ymult, self.xwidth, self.ywidth,
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

    b = frame_scan_grid_idx.boxes
    counts = np.zeros(N, dtype=np.int64)
    with ProgressBar(total=N, desc="count intersections") as progress:
        frame_scan_grid_idx.query(
            b, count_2d_intersections,
            counts, b.xcenters, b.ycenters, b.xscales, b.yscales, b.xmult, b.ymult,
            progress=progress,
        )
    print("2D intersection counts:", counts[:10])

    counts3d = np.zeros(N, dtype=np.int64)
    with ProgressBar(total=N, desc="count 3D intersections") as progress:
        frame_scan_grid_idx.query(
            b, count_3d_intersections,
            counts3d, b.xcenters, b.ycenters, b.xscales, b.yscales, b.xmult, b.ymult,
            data.mz, mz_radius_da,
            progress=progress,
        )
    print("3D intersection counts:", counts3d[:10])

    from timstofu.stats import count1D
    from timstofu.plotting import plot_counts

    plot_counts(count1D(counts), xlog=False, ylog=False)
    # The problem

    # F9477 -
    # F9468 -
    # B6699 - longer gradient

"""
In git/boxing/__dev/deslop2.py I have frames and scans that are ints. 
I want to construct a 2D index for box intersection querries.
I want that by 

to that end I want to corsify the grid of     
  points that boxes occupy. Each box is define by center x and y position (data.frame adn data.scan).   
  I have already corse width {frame,scan}_box_width. I have {scan,frame}.{min,max,box_width} and want   
  to use those to define grid. The grid should allow for representing all boxes. One box i   
"""
