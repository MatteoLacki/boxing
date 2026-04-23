"""Compute top-k precursor neighbors and save as CSR mmappet datasets."""
import argparse
import math
import tomllib
from pathlib import Path

import duckdb
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from boxing.spatial_index import dense_neighbors_to_csr, find_top_k_neighbors_2d_zz


class PrecursorNeighborsConfig(BaseModel):
    """Scale multipliers for outer and inner boxes when building neighbor shells.

    Outer box: frame ± frame_mult * frame_scale, scan ± scan_mult * scan_scale.
    Inner box: same center, frame_inner_mult / scan_inner_mult radii.
    Only precursors in the shell (inside outer, outside inner) are candidates.
    When frame_inner_mult == scan_inner_mult == 0 (default), no inner filter is
    applied and the result is identical to the outer-box-only behavior.

    >>> PrecursorNeighborsConfig(frame_mult=1.5, scan_mult=0.5, frame_inner_mult=0.3)
    PrecursorNeighborsConfig(frame_mult=1.5, scan_mult=0.5, frame_inner_mult=0.3, scan_inner_mult=0.0)
    >>> PrecursorNeighborsConfig(frame_mult=-1.0)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for PrecursorNeighborsConfig
    frame_mult
      Input should be greater than 0 ...
        For further information visit ...
    """

    model_config = ConfigDict(extra="ignore")

    frame_mult: float = Field(default=1.0, gt=0)
    scan_mult: float = Field(default=1.0, gt=0)
    frame_inner_mult: float = Field(default=0.0, ge=0)
    scan_inner_mult: float = Field(default=0.0, ge=0)
    mz_inner_radius_da: float | None = Field(default=None, ge=0)


def _load_config(config_path: Path | None) -> PrecursorNeighborsConfig:
    if config_path is None:
        return PrecursorNeighborsConfig()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    try:
        return PrecursorNeighborsConfig(**raw)
    except ValidationError as exc:
        lines = [f"Config error in {config_path}:"]
        for err in exc.errors():
            loc = (
                " → ".join(str(p) for p in err["loc"]) if err["loc"] else "(top level)"
            )
            lines.append(f"  {loc}: {err['msg']}")
        raise ValueError("\n".join(lines)) from exc


_REQUIRED_COLUMNS = {
    "frame",
    "frame_scale",
    "scan",
    "scan_scale",
    "tof",
    "mz",
    "intensity",
    "precursor_idx",
}

_BOXES_SQL = """
SELECT
    CAST(CEIL(frame - {frame_mult} * frame_scale) AS INTEGER) AS frame_lo,
    CAST(CEIL(scan - {scan_mult} * scan_scale) AS INTEGER) AS scan_lo,
    CAST(CEIL({tof_lo_col}) AS INTEGER) AS tof_lo,

    CAST(
        CASE
            WHEN frame + {frame_mult} * frame_scale = FLOOR(frame + {frame_mult} * frame_scale)
                THEN frame + {frame_mult} * frame_scale + 1
            ELSE CEIL(frame + {frame_mult} * frame_scale)
        END AS INTEGER
    ) AS frame_hi,

    CAST(
        CASE
            WHEN scan + {scan_mult} * scan_scale = FLOOR(scan + {scan_mult} * scan_scale)
                THEN scan + {scan_mult} * scan_scale + 1
            ELSE CEIL(scan + {scan_mult} * scan_scale)
        END AS INTEGER
    ) AS scan_hi,

    CAST(
        CASE
            WHEN {tof_hi_col} = FLOOR({tof_hi_col})
                THEN {tof_hi_col} + 1
            ELSE CEIL({tof_hi_col})
        END AS INTEGER
    ) AS tof_hi

FROM 'prec'
"""


def _add_inner_tof_bounds(prec, dataset_path: Path, mz_inner_radius_da: float) -> None:
    """Add tof_lo_inner / tof_hi_inner columns to prec in-place from mz ± mz_inner_radius_da."""
    from opentimspy import OpenTIMS
    from timstofu.timstofmisc import get_tof2mz_and_mz2tof

    _, mz2tof = get_tof2mz_and_mz2tof(OpenTIMS(dataset_path))
    prec["tof_lo_inner"] = mz2tof(prec.mz - mz_inner_radius_da).astype(float)
    prec["tof_hi_inner"] = mz2tof(prec.mz + mz_inner_radius_da).astype(float)


def _add_tof_isolation_bounds(prec, dataset_path: Path) -> None:
    """Add tof_lo / tof_hi columns to prec in-place using DIA isolation window from TDF."""
    import pandas as pd
    from scipy.interpolate import CubicSpline
    from opentimspy import OpenTIMS
    from timstofu.timstofmisc import get_tof2mz_and_mz2tof

    data_handler = OpenTIMS(dataset_path)
    _, mz2tof = get_tof2mz_and_mz2tof(data_handler)

    windows = pd.DataFrame(data_handler.table2dict("DiaFrameMsMsWindows"), copy=False)
    isolation_widths = np.unique(windows.IsolationWidth)
    if len(isolation_widths) != 1:
        raise ValueError(f"Expected a single IsolationWidth, got {isolation_widths}")
    mz_inner_radius_da = isolation_widths[0] / 2.0

    mz_grid = np.arange(math.floor(prec.mz.min()), math.ceil(prec.mz.max()) + 1)
    tof_grid = mz2tof(mz_grid)
    tof2lo = CubicSpline(tof_grid, mz2tof(mz_grid - mz_inner_radius_da))
    tof2hi = CubicSpline(tof_grid, mz2tof(mz_grid + mz_inner_radius_da))

    prec["tof_lo"] = tof2lo(prec.tof)
    prec["tof_hi"] = tof2hi(prec.tof)


def compute_precursor_neighbors(
    precursors_path: Path,
    dataset_path: Path,
    out_path: Path,
    top_k: int = 64,
    config_path: Path | None = None,
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
    config_path     : optional TOML with frame_mult/scan_mult; defaults
        to 1.0 for all multipliers when None.
    """
    from numba_progress import ProgressBar
    from pandas_ops.io import read_df

    cfg = _load_config(config_path)

    prec = read_df(precursors_path)

    missing = _REQUIRED_COLUMNS - set(prec.columns)
    if missing:
        raise ValueError(
            f"{precursors_path}: missing required columns: {sorted(missing)}"
        )

    _add_tof_isolation_bounds(prec, dataset_path)

    con = duckdb.connect()
    sql = _BOXES_SQL.format(
        frame_mult=cfg.frame_mult,
        scan_mult=cfg.scan_mult,
        tof_lo_col="tof_lo",
        tof_hi_col="tof_hi",
    )
    boxes = con.query(sql).df()
    boxes_arr = boxes[
        ["frame_lo", "frame_hi", "scan_lo", "scan_hi", "tof_lo", "tof_hi"]
    ].to_numpy()

    if cfg.frame_inner_mult > 0 or cfg.scan_inner_mult > 0:
        if cfg.mz_inner_radius_da is not None:
            _add_inner_tof_bounds(prec, dataset_path, cfg.mz_inner_radius_da)
            tof_lo_col, tof_hi_col = "tof_lo_inner", "tof_hi_inner"
        else:
            tof_lo_col, tof_hi_col = "tof_lo", "tof_hi"
        inner_sql = _BOXES_SQL.format(
            frame_mult=cfg.frame_inner_mult,
            scan_mult=cfg.scan_inner_mult,
            tof_lo_col=tof_lo_col,
            tof_hi_col=tof_hi_col,
        )
        inner_boxes = con.query(inner_sql).df()
        inner_boxes_arr = inner_boxes[
            ["frame_lo", "frame_hi", "scan_lo", "scan_hi", "tof_lo", "tof_hi"]
        ].to_numpy()
    else:
        inner_boxes_arr = None
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
            inner_boxes=inner_boxes_arr,
        )

    dense_neighbors_to_csr(
        neighbor_ids,
        neighbor_ints,
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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="TOML",
        help="Config TOML with urt_mult/frame_mult/scan_mult (default: all 1.0).",
    )
    args = parser.parse_args()
    compute_precursor_neighbors(
        args.precursors_path,
        args.dataset_path,
        args.out_path,
        top_k=args.top_k,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
