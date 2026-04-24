"""Compute top-k precursor neighbors and save as CSR mmappet datasets."""
import argparse
import tomllib
from pathlib import Path
from typing import Literal
from numba_progress import ProgressBar
from pandas_ops.io import read_df

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from boxing.spatial_index import dense_neighbors_to_csr, find_top_k_neighbors_2d_zz


class PrecursorNeighborsConfig(BaseModel):
    """Scale multipliers and geometry options for precursor-neighbor supports.

    Outer support: frame ± frame_mult * frame_scale,
    scan ± scan_mult * scan_scale, and mz ± DIA isolation m/z half-width.
    Inner support: same center, frame_inner_mult / scan_inner_mult radii, and
    mz_inner_radius_da or the outer m/z half-width.
    Only precursors in the shell (inside outer, outside inner) are candidates.
    When frame_inner_mult == scan_inner_mult == 0 (default), no inner filter is
    applied and the result is identical to the outer-box-only behavior.

    >>> PrecursorNeighborsConfig(frame_mult=1.5, scan_mult=0.5, frame_inner_mult=0.3)  # doctest: +ELLIPSIS
    PrecursorNeighborsConfig(frame_mult=1.5, scan_mult=0.5, frame_inner_mult=0.3, ... cylinder_radius=1.0)
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
    top_k: int = Field(default=64, gt=0)
    geometry: Literal["box", "cylinder"] = "box"
    cylinder_radius: float = Field(default=1.0, gt=0)


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
    "mz",
    "intensity",
    "precursor_idx",
}


def _read_isolation_mz_radius_da(data_handler) -> float:
    """Return the DIA isolation half-width in m/z units."""
    windows = data_handler.table2dict("DiaFrameMsMsWindows")
    isolation_widths = np.unique(np.asarray(windows["IsolationWidth"], dtype=float))
    if len(isolation_widths) != 1:
        raise ValueError(f"Expected a single IsolationWidth, got {isolation_widths}")
    return float(isolation_widths[0]) / 2.0


def _read_outer_mz_radius_da(dataset_path: Path) -> float:
    """Read the outer m/z radius from DIA isolation metadata."""
    from opentimspy import OpenTIMS

    data_handler = OpenTIMS(dataset_path)
    return _read_isolation_mz_radius_da(data_handler)


if False:
    precursors_path = Path(
        "temp/F9477/optimal2tier/pre_sage_filtered_precursor_clusters.parquet"
    )
    dataset_path = Path("data/F9477.d")
    config_path = Path("configs/precursor_neighbors/default.toml")
    out_path = Path("/home/matteo/tmp/F9477nbs.mmappet")
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 5)


def compute_precursor_neighbors(
    precursors_path: Path,
    dataset_path: Path,
    out_path: Path,
    config_path: Path | None = None,
) -> None:
    """Compute top-k intense box-intersection neighbors for each precursor.

    Reads precursors_path (parquet), builds 3-D supports over frame/scan/mz,
    runs find_top_k_neighbors_2d_zz, and saves the CSR result to out_path via
    dense_neighbors_to_csr.

    Parameters
    ----------
    precursors_path : parquet file; must contain columns listed in
        _REQUIRED_COLUMNS.
    dataset_path    : Bruker .d folder; used to read DIA isolation window
        as the outer m/z radius.
    out_path        : destination folder (must not exist); two mmappet datasets
        are written:
            neighbors.mmappet  — prec_idx (int32), intensity (int64)
            index.mmappet      — offset (int64)
    config_path     : optional TOML with frame_mult/scan_mult; defaults
        to 1.0 for all multipliers when None.
    """

    cfg = _load_config(config_path)

    prec = read_df(precursors_path)

    missing = _REQUIRED_COLUMNS - set(prec.columns)
    if missing:
        raise ValueError(
            f"{precursors_path}: missing required columns: {sorted(missing)}"
        )

    mz_radius_da = _read_outer_mz_radius_da(dataset_path)

    inner_xy_mults = None
    inner_mz_radius_da = None
    if cfg.frame_inner_mult > 0 or cfg.scan_inner_mult > 0:
        inner_xy_mults = (cfg.frame_inner_mult, cfg.scan_inner_mult)
        inner_mz_radius_da = cfg.mz_inner_radius_da

    centers = prec[["frame", "scan", "mz"]].to_numpy(dtype=float)
    scales = prec[["frame_scale", "scan_scale"]].to_numpy(dtype=float)
    prec_idxs = prec.precursor_idx.to_numpy(dtype=np.int32)

    with ProgressBar(
        total=len(prec), desc=f"top-{cfg.top_k} neighbors (frame/scan + mz)"
    ) as progress:
        neighbor_ids, neighbor_ints, prec_to_row = find_top_k_neighbors_2d_zz(
            centers,
            scales,
            mz_radius_da,
            prec.intensity.to_numpy(),
            top_k=cfg.top_k,
            xy_mults=(cfg.frame_mult, cfg.scan_mult),
            progress=progress,
            geometry=cfg.geometry,
            cylinder_radius=cfg.cylinder_radius,
            precursor_idxs=prec_idxs,
            inner_xy_mults=inner_xy_mults,
            inner_mz_radius_da=inner_mz_radius_da,
        )

    dense_neighbors_to_csr(
        neighbor_ids,
        neighbor_ints,
        prec_to_row=prec_to_row,
        out_path=out_path,
    )

    print(f"Saved CSR neighbors to {out_path}")
    print(f"  precursors : {len(prec):,}")
    print(f"  top_k      : {cfg.top_k}")
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
        help="Bruker .d dataset folder (for DIA isolation window metadata).",
    )
    parser.add_argument(
        "out_path",
        type=Path,
        help="Output folder (must not exist); CSR mmappet datasets written here.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="TOML",
        help="Config TOML with frame/scan multipliers and geometry options.",
    )
    args = parser.parse_args()
    compute_precursor_neighbors(
        args.precursors_path,
        args.dataset_path,
        args.out_path,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
