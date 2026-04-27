#!/usr/bin/env python3
"""Build sorted precursor boxes and a 3D GridIndex for C++ neighbor queries."""

import argparse
from pathlib import Path

import mmappet
import numpy as np
import pandas as pd
from pandas_ops.io import read_df

from boxing.precursor_neighbors_config import (
    read_outer_mz_radius_da,
    validate_config as _validate_precursor_neighbors_config,
)
from boxing.spatial_index import Boxes3D, GridIndex3D


_REQUIRED_COLUMNS = {
    "frame",
    "scan",
    "mz",
    "frame_scale",
    "scan_scale",
    "precursor_idx",
    "intensity",
}


def validate_config(config_path: Path) -> object:
    return _validate_precursor_neighbors_config(
        config_path,
        require_cpp_backend_compat=True,
    )


def build(
    precursors_path: Path,
    dataset_path: Path,
    output_dir: Path,
    config_path: Path,
) -> None:
    cfg = validate_config(config_path)
    prec = read_df(precursors_path)

    missing = _REQUIRED_COLUMNS - set(prec.columns)
    if missing:
        raise ValueError(
            f"{precursors_path}: missing required columns: {sorted(missing)}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    mz_radius_da = read_outer_mz_radius_da(dataset_path)

    boxes3d = Boxes3D(
        xcenters=prec["frame"].to_numpy(dtype=np.float64),
        ycenters=prec["scan"].to_numpy(dtype=np.float64),
        mz=prec["mz"].to_numpy(dtype=np.float64),
        xscales=prec["frame_scale"].to_numpy(dtype=np.float64),
        yscales=prec["scan_scale"].to_numpy(dtype=np.float64),
        mz_radius_da=mz_radius_da,
        xmult=cfg.frame_mult,
        ymult=cfg.scan_mult,
    )

    print(f"Building GridIndex3D for {len(boxes3d.mz)} precursors")
    idx = GridIndex3D.from_boxes(boxes3d)

    order = boxes3d.order
    n_precursors = len(order)
    inverse_order = np.empty(n_precursors, dtype=np.int64)
    inverse_order[order] = np.arange(n_precursors, dtype=np.int64)
    nb_idxs_remapped = inverse_order[idx.nb_box_idxs].astype(np.int32)

    cols: dict[str, np.ndarray] = {
        "frame": prec["frame"].to_numpy()[order].astype(np.uint32),
        "scan": prec["scan"].to_numpy()[order].astype(np.uint32),
        "mz": prec["mz"].to_numpy()[order].astype(np.float32),
        "frame_scale": prec["frame_scale"].to_numpy()[order].astype(np.float32),
        "scan_scale": prec["scan_scale"].to_numpy()[order].astype(np.float32),
        "precursor_idx": prec["precursor_idx"].to_numpy()[order].astype(np.uint32),
        "intensity": prec["intensity"].to_numpy()[order].astype(np.uint64),
    }
    boxes_scheme = pd.DataFrame({k: pd.Series(dtype=v.dtype) for k, v in cols.items()})
    boxes_ds = mmappet.open_new_dataset_dct(
        output_dir / "boxes.mmappet", boxes_scheme, n_precursors
    )
    for col, arr in cols.items():
        boxes_ds[col][:] = arr

    cells_flat = idx.cells.ravel().astype(np.int64)
    cells_scheme = pd.DataFrame({"offset": pd.Series(dtype=np.int64)})
    cells_ds = mmappet.open_new_dataset_dct(
        output_dir / "cells.mmappet", cells_scheme, len(cells_flat)
    )
    cells_ds["offset"][:] = cells_flat

    nb_scheme = pd.DataFrame({"idx": pd.Series(dtype=np.int32)})
    nb_ds = mmappet.open_new_dataset_dct(
        output_dir / "nb_idxs.mmappet", nb_scheme, len(nb_idxs_remapped)
    )
    nb_ds["idx"][:] = nb_idxs_remapped

    bx, by, bz_plus_1 = idx.cells.shape
    meta = {
        "bx": bx,
        "by": by,
        "bz_plus_1": bz_plus_1,
        "xwidth": idx.xwidth,
        "ywidth": idx.ywidth,
        "zwidth": idx.zwidth,
        "mz_offset": boxes3d.mz_offset,
    }
    with open(output_dir / "grid_meta.txt", "w") as f:
        for key, value in meta.items():
            f.write(f"{key} = {value}\n")

    print(
        f"Done. N={n_precursors}, cells={idx.cells.shape}, "
        f"nb_idxs={len(nb_idxs_remapped)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precursors_path", type=Path)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build(args.precursors_path, args.dataset_path, args.output_dir, args.config)


if __name__ == "__main__":
    main()
