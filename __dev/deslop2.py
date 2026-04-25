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


@numba.njit(boundscheck=True)
def count_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, out,):
    """Increment out[i,j] for every grid cell each box overlaps. First pass of CSR build."""
    inv_xw = 1 / xwidth
    inv_yw = 1 / ywidth
    xhi = out.shape[0]
    yhi = out.shape[1]
    for idx in range(len(xcenters)):
        xc = xcenters[idx]; yc = ycenters[idx]
        xs = xscales[idx];  ys = yscales[idx]
        imin = lo_cell_idx(xc, xmult*xs, inv_xw)
        jmin = lo_cell_idx(yc, ymult*ys, inv_yw)
        itop = hi_cell_idx(xc, xmult*xs, inv_xw, xhi)
        jtop = hi_cell_idx(yc, ymult*ys, inv_yw, yhi)
        for i in range(imin, itop):
            for j in range(jmin, jtop):
                out[i,j] += 1


@numba.njit(boundscheck=True)
def fill_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, idx, flat):
    """Write array index into flat for every cell it overlaps. Second pass of CSR build."""
    inv_xw = 1 / xwidth
    inv_yw = 1 / ywidth
    xhi = idx.shape[0]
    yhi = idx.shape[1] - 1
    cursors = np.zeros_like(idx)
    for n in range(len(xcenters)):
        xc = xcenters[n]; yc = ycenters[n]
        xs = xscales[n];  ys = yscales[n]
        imin = lo_cell_idx(xc, xmult*xs, inv_xw)
        jmin = lo_cell_idx(yc, ymult*ys, inv_yw)
        itop = hi_cell_idx(xc, xmult*xs, inv_xw, xhi)
        jtop = hi_cell_idx(yc, ymult*ys, inv_yw, yhi)
        for i in range(imin, itop):
            for j in range(jmin, jtop):
                pos = idx[i, j] + cursors[i, j]
                flat[pos] = n
                cursors[i, j] += 1


@numba.njit
def _visit_intersections(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth,
                         cells, nb_box_idxs,
                         idx_xcenters, idx_ycenters, idx_xscales, idx_yscales,
                         processor, processor_args):
    """For each query box n, call processor(n, k, *processor_args) for every indexed box k it intersects.
    Each pair is visited exactly once via the canonical-cell rule: max(lo_n, lo_k) per axis.
    First buckets of indexed boxes are computed on the fly from idx_x/ycenters and idx_x/yscales."""
    inv_xw = 1 / xwidth
    inv_yw = 1 / ywidth
    xhi = cells.shape[0]
    yhi = cells.shape[1] - 1
    for stored_box_idx in range(len(xcenters)):
        xc = xcenters[stored_box_idx]; yc = ycenters[stored_box_idx]
        xs = xscales[stored_box_idx];  ys = yscales[stored_box_idx]
        imin = lo_cell_idx(xc, xmult*xs, inv_xw)
        jmin = lo_cell_idx(yc, ymult*ys, inv_yw)
        itop = hi_cell_idx(xc, xmult*xs, inv_xw, xhi)
        jtop = hi_cell_idx(yc, ymult*ys, inv_yw, yhi)
        for i in range(imin, itop):
            for j in range(jmin, jtop):
                for pos in range(cells[i, j], cells[i, j + 1]):
                    nb_box_idx = nb_box_idxs[pos]
                    # on fly finding of stored boxes top-left cell idxs
                    nb_lowest_x = lo_cell_idx(idx_xcenters[nb_box_idx], xmult*idx_xscales[nb_box_idx], inv_xw)
                    nb_lowest_y = lo_cell_idx(idx_ycenters[nb_box_idx], ymult*idx_yscales[nb_box_idx], inv_yw)
                    if i != max(imin, nb_lowest_x) or j != max(jmin, nb_lowest_y):
                        continue
                    processor(stored_box_idx, nb_box_idx, *processor_args)


@dataclass
class GridIndex:
    cells: NDArray
    nb_box_idxs: NDArray   # array indices (0..N-1) per cell membership
    ids: NDArray             # ids[array_idx] -> original id (e.g. precursor_idx)
    xcenters: NDArray
    ycenters: NDArray
    xscales: NDArray
    yscales: NDArray
    xmult: float
    ymult: float
    xwidth: int
    ywidth: int

    @classmethod
    def from_xy(cls, xcenters, ycenters, xscales, yscales, xmult, ymult, ids):
        xwidth = scales_to_width(xscales)
        ywidth = scales_to_width(yscales)
        cells_cnt = lambda max_val, width: (int(max_val) + 1) // width + 1
        idx = np.zeros(
            (cells_cnt(xcenters.max(), xwidth), cells_cnt(ycenters.max(), ywidth) + 2),
            dtype=np.int64,
        )
        count_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, idx)
        total = inplace_start_pos(idx.ravel())
        flat = np.empty(total, dtype=np.int64)
        fill_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, idx, flat)
        return cls(cells=idx, nb_box_idxs=flat, ids=np.asarray(ids),
                   xcenters=xcenters, ycenters=ycenters, xscales=xscales, yscales=yscales,
                   xmult=xmult, ymult=ymult, xwidth=xwidth, ywidth=ywidth)

    def query(self, xcenters, ycenters, xscales, yscales, processor, *processor_args):
        _visit_intersections(
            xcenters, ycenters, xscales, yscales,
            self.xmult, self.ymult, self.xwidth, self.ywidth,
            self.cells, self.nb_box_idxs,
            self.xcenters, self.ycenters, self.xscales, self.yscales,
            processor, processor_args,
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

    frame_scan_grid_idx = GridIndex.from_xy(
        xcenters=data.frame,
        ycenters=data.scan,
        xscales=data.frame_scale,
        yscales=data.scan_scale,
        xmult=cfg.frame_mult,
        ymult=cfg.scan_mult,
        ids=data.precursor_idx,
    )


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
