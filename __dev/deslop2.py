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
    median_width = 2 * np.median(scales)
    return max(1, int(factor * median_width))



@numba.njit
def minmax(xx: NDArray):
    _min = xx[0]
    _max = xx[0]
    for x in xx:
        _min = min(_min, x)
        _max = max(_max, x)
    return _min, _max


def inplace_start_pos(idx):
    idx.flat[0] = 0
    idx.ravel()[1:] = np.cumsum(idx.ravel())[:-1]



@numba.njit(boundscheck=True)
def count_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, out,):
    inv_xw = 1 / xwidth
    inv_yw = 1 / ywidth
    xlo = 0
    ylo = 0
    xhi = out.shape[0]
    yhi = out.shape[1]
    for idx in range(len(xcenters)):
        xc = xcenters[idx]
        yc = ycenters[idx]
        xs = xscales[idx]
        ys = yscales[idx]
        imin = max(int((xc - xmult*xs) * inv_xw), xlo)
        jmin = max(int((yc - ymult*ys) * inv_yw), ylo)
        itop = min(int((xc + xmult*xs) * inv_xw) + 1, xhi)
        jtop = min(int((yc + ymult*ys) * inv_yw) + 1, yhi)
        for i in range(imin, itop):
            for j in range(jmin, jtop):
                out[i,j] += 1


@numba.njit(boundscheck=True)
def fill_box_content(xcenters, ycenters, xscales, yscales, xmult, ymult, xwidth, ywidth, idx, precursor_idxs, flat,):
    inv_xw = 1 / xwidth
    inv_yw = 1 / ywidth
    xhi = idx.shape[0]
    yhi = idx.shape[1] - 1
    cursors = np.zeros_like(idx)
    for n in range(len(xcenters)):
        xc = xcenters[n]; yc = ycenters[n]
        xs = xscales[n];  ys = yscales[n]
        imin = max(int((xc - xmult*xs) * inv_xw), 0)
        jmin = max(int((yc - ymult*ys) * inv_yw), 0)
        itop = min(int((xc + xmult*xs) * inv_xw) + 1, xhi)
        jtop = min(int((yc + ymult*ys) * inv_yw) + 1, yhi)
        for i in range(imin, itop):
            for j in range(jmin, jtop):
                flat[idx[i, j] + cursors[i, j]] = precursor_idxs[n]
                cursors[i, j] += 1

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

    frame = dictodot.DotDict()
    frame.min, frame.max = minmax(data.frame)
    frame.box_width = scales_to_width(data.frame_scale)
    
    scan = dictodot.DotDict()
    scan.min, scan.max = minmax(data.scan)
    scan.box_width = scales_to_width(data.scan_scale)

    cells_cnt = lambda max_box_val, box_width: (max_box_val+1) // box_width + 1

    # scan + 1 for it to be and idx
    idx = np.zeros(
        (
            cells_cnt(frame.max, frame.box_width),
            cells_cnt(scan.max, scan.box_width) + 2
        ),
        np.int64,
    )

    count_box_content(
        xcenters=data.frame,
        ycenters=data.scan,
        xscales=data.frame_scale,
        yscales=data.scan_scale,
        xmult=cfg.frame_mult,
        ymult=cfg.scan_mult,
        xwidth=frame.box_width,
        ywidth=scan.box_width,
        out=idx,
    )
    if PLOT:
        plt.matshow(idx, aspect="auto")
        plt.show()

    inplace_start_pos(idx)
    total = int(idx[-1, -1])
    precursor_idxs = np.empty(total, dtype=data.precursor_idx.dtype)
    

    fill_box_content(
        xcenters=data.frame,
        ycenters=data.scan,
        xscales=data.frame_scale,
        yscales=data.scan_scale,
        xmult=cfg.frame_mult,
        ymult=cfg.scan_mult,
        xwidth=frame.box_width,
        ywidth=scan.box_width,
        idx=idx,
        precursor_idxs=data.precursor_idx,
        flat=precursor_idxs,
    )

    
    idx, precursor_idxs


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
