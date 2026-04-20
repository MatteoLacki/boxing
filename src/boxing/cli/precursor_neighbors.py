"""Compute top-k precursor neighbors and save as CSR mmappet datasets."""
import argparse
import math
from pathlib import Path

import duckdb
import numpy as np

from boxing.spatial_index import dense_neighbors_to_csr, find_top_k_neighbors_2d_zz


_REQUIRED_COLUMNS = {
    "urt", "urt_scale",
    "frame", "frame_scale",
    "scan", "scan_scale",
    "tof", "mz",
    "intensity", "precursor_idx",
}

_BOXES_SQL = """
SELECT
    CAST(CEIL(urt - urt_scale) AS INTEGER) AS urt_lo,
    CAST(CEIL(frame - frame_scale) AS INTEGER) AS frame_lo,
    CAST(CEIL(scan - scan_scale) AS INTEGER) AS scan_lo,
    CAST(CEIL(tof_lo) AS INTEGER) AS tof_lo,

    CAST(
        CASE
            WHEN urt + urt_scale = FLOOR(urt + urt_scale)
                THEN urt + urt_scale + 1
            ELSE CEIL(urt + urt_scale)
        END AS INTEGER
    ) AS urt_hi,

    CAST(
        CASE
            WHEN frame + frame_scale = FLOOR(frame + frame_scale)
                THEN frame + frame_scale + 1
            ELSE CEIL(frame + frame_scale)
        END AS INTEGER
    ) AS frame_hi,

    CAST(
        CASE
            WHEN scan + scan_scale = FLOOR(scan + scan_scale)
                THEN scan + scan_scale + 1
            ELSE CEIL(scan + scan_scale)
        END AS INTEGER
    ) AS scan_hi,

    CAST(
        CASE
            WHEN tof_hi = FLOOR(tof_hi)
                THEN tof_hi + 1
            ELSE CEIL(tof_hi)
        END AS INTEGER
    ) AS tof_hi

FROM 'prec'
"""


def _add_tof_isolation_bounds(prec, dataset_path: Path) -> None:
    """Add tof_lo / tof_hi columns to prec in-place using DIA isolation window from TDF."""
    import pandas as pd
    from scipy.interpolate import CubicSpline
    from opentimspy import OpenTIMS
    from timstofu.timstofmisc import get_tof2mz_and_mz2tof

    data_handler = OpenTIMS(dataset_path)
    _, mz2tof = get_tof2mz_and_mz2tof(data_handler)

    windows = pd.DataFrame(
        data_handler.table2dict("DiaFrameMsMsWindows"), copy=False
    )
    isolation_widths = np.unique(windows.IsolationWidth)
    if len(isolation_widths) != 1:
        raise ValueError(
            f"Expected a single IsolationWidth, got {isolation_widths}"
        )
    mz_radius = isolation_widths[0] / 2.0

    mz_grid = np.arange(math.floor(prec.mz.min()), math.ceil(prec.mz.max()) + 1)
    tof_grid = mz2tof(mz_grid)
    tof2lo = CubicSpline(tof_grid, mz2tof(mz_grid - mz_radius))
    tof2hi = CubicSpline(tof_grid, mz2tof(mz_grid + mz_radius))

    prec["tof_lo"] = tof2lo(prec.tof)
    prec["tof_hi"] = tof2hi(prec.tof)


def compute_precursor_neighbors(
    precursors_path: Path,
    dataset_path: Path,
    out_path: Path,
    top_k: int = 64,
) -> None:
    """Compute top-k intense box-intersection neighbors for each precursor.

    Reads precursors_path (parquet), builds 6-D boxes (urt/frame/scan × tof),
    runs find_top_k_neighbors_2d_zz, and saves the CSR result to out_path via
    dense_neighbors_to_csr.

    Parameters
    ----------
    precursors_path : parquet file; must contain columns listed in
        _REQUIRED_COLUMNS.  tof_lo/tof_hi are derived from dataset_path.
    dataset_path    : Bruker .d folder; used to read DIA isolation window
        and tof↔mz calibration for computing tof_lo/tof_hi.
    out_path        : destination folder (must not exist); two mmappet datasets
        are written:
            neighbors.mmappet  — prec_idx (int32), intensity (int64)
            index.mmappet      — offset (int64)
    top_k           : maximum neighbors per precursor (default 64).
    """
    from numba_progress import ProgressBar
    from pandas_ops.io import read_df

    prec = read_df(precursors_path)

    missing = _REQUIRED_COLUMNS - set(prec.columns)
    if missing:
        raise ValueError(
            f"{precursors_path}: missing required columns: {sorted(missing)}"
        )

    _add_tof_isolation_bounds(prec, dataset_path)

    con = duckdb.connect()
    boxes = con.query(_BOXES_SQL).df()

    boxes_arr = boxes[
        ["frame_lo", "frame_hi", "scan_lo", "scan_hi", "tof_lo", "tof_hi"]
    ].to_numpy()
    prec_idxs = prec.precursor_idx.to_numpy(dtype=np.int32)

    with ProgressBar(
        total=len(boxes), desc=f"top-{top_k} neighbors (frame/scan + tof)"
    ) as progress:
        neighbor_ids, neighbor_ints, prec_to_row = find_top_k_neighbors_2d_zz(
            boxes_arr,
            prec.intensity.to_numpy(),
            top_k=top_k,
            progress=progress,
            precursor_idxs=prec_idxs,
        )

    dense_neighbors_to_csr(
        neighbor_ids, neighbor_ints,
        prec_to_row=prec_to_row,
        out_path=out_path,
    )

    print(f"Saved CSR neighbors to {out_path}")
    print(f"  precursors : {len(prec):,}")
    print(f"  top_k      : {top_k}")
    n_with_neighbors = int((neighbor_ids[:, 0] >= 0).sum())
    print(f"  with >=1 neighbor: {n_with_neighbors:,} / {len(prec):,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute top-k intense box-intersection neighbors for each precursor "
            "and save as CSR mmappet datasets."
        )
    )
    parser.add_argument(
        "precursors_path",
        type=Path,
        help="Precursor parquet (pre_sage_filtered_precursor_clusters or similar).",
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Bruker .d dataset folder (for DIA isolation window + tof calibration).",
    )
    parser.add_argument(
        "out_path",
        type=Path,
        help="Output folder (must not exist); CSR mmappet datasets written here.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        metavar="K",
        help="Max neighbors per precursor (default: 64).",
    )
    args = parser.parse_args()
    compute_precursor_neighbors(
        args.precursors_path,
        args.dataset_path,
        args.out_path,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
