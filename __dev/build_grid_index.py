#!/usr/bin/env python3
"""Build sorted boxes mmappet and GridIndex3D from precursor parquet.

Output directory layout:
    out_dir/boxes.mmappet/      sorted precursor boxes (u32 frame/scan, f32 mz, ...)
    out_dir/cells.mmappet/      flat int64 CSR offsets [BX*BY*(BZ+1)]
    out_dir/nb_idxs.mmappet/    int32 box indices per cell
    out_dir/grid_meta.txt       key = value metadata (bx, by, bz_plus_1, widths, mz_offset)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mmappet

def _import_deslop2():
    import importlib.util, tempfile, os
    src_path = Path(__file__).parent / "deslop2.py"
    src = src_path.read_text()
    # Strip IPython magic lines (%load_ext, %autoreload, etc.)
    lines = [l for l in src.splitlines() if not l.lstrip().startswith('%')]
    with tempfile.NamedTemporaryFile(mode='w', suffix='_deslop2.py', delete=False) as tmp:
        tmp.write('\n'.join(lines))
        tmp_path = tmp.name
    try:
        spec = importlib.util.spec_from_file_location('deslop2', tmp_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['deslop2'] = mod
        spec.loader.exec_module(mod)
    finally:
        os.unlink(tmp_path)
    return mod

_d2 = _import_deslop2()
Boxes3D     = _d2.Boxes3D
GridIndex3D = _d2.GridIndex3D


def build(parquet_path: Path, out_dir: Path, mz_radius_da: float,
          frame_mult: float = 2.5, scan_mult: float = 2.5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    prec = pd.read_parquet(parquet_path)
    has_intensity = "intensity" in prec.columns

    boxes3d = Boxes3D(
        xcenters=prec["frame"].to_numpy(dtype=np.float64),
        ycenters=prec["scan"].to_numpy(dtype=np.float64),
        mz=prec["mz"].to_numpy(dtype=np.float64),
        xscales=prec["frame_scale"].to_numpy(dtype=np.float64),
        yscales=prec["scan_scale"].to_numpy(dtype=np.float64),
        mz_radius_da=mz_radius_da,
        xmult=frame_mult,
        ymult=scan_mult,
    )

    print(f"Building GridIndex3D for {len(boxes3d.mz)} precursors…")
    idx = GridIndex3D.from_boxes(boxes3d)

    # order[i] = original index of the i-th sorted box
    order = boxes3d.order
    N = len(order)

    # inverse_order[original_idx] = sorted position
    inverse_order = np.empty(N, dtype=np.int64)
    inverse_order[order] = np.arange(N, dtype=np.int64)

    # Remap nb_box_idxs from original indices to sorted positions
    nb_idxs_remapped = inverse_order[idx.nb_box_idxs].astype(np.int32)

    # --- Sorted boxes mmappet ---
    boxes_out = out_dir / "boxes.mmappet"
    cols: dict[str, np.ndarray] = {
        "frame":         prec["frame"].to_numpy()[order].astype(np.uint32),
        "scan":          prec["scan"].to_numpy()[order].astype(np.uint32),
        "mz":            prec["mz"].to_numpy()[order].astype(np.float32),
        "frame_scale":   prec["frame_scale"].to_numpy()[order].astype(np.float32),
        "scan_scale":    prec["scan_scale"].to_numpy()[order].astype(np.float32),
        "precursor_idx": prec["precursor_idx"].to_numpy()[order].astype(np.uint32),
    }
    if has_intensity:
        cols["intensity"] = prec["intensity"].to_numpy()[order].astype(np.uint64)

    scheme = pd.DataFrame({k: pd.Series(dtype=v.dtype) for k, v in cols.items()})
    ds = mmappet.open_new_dataset_dct(boxes_out, scheme, N)
    for col, arr in cols.items():
        ds[col][:] = arr

    # --- cells.mmappet: flat int64[BX*BY*(BZ+1)] ---
    cells_flat = idx.cells.ravel().astype(np.int64)
    cells_scheme = pd.DataFrame({"offset": pd.Series(dtype=np.int64)})
    cells_ds = mmappet.open_new_dataset_dct(out_dir / "cells.mmappet", cells_scheme, len(cells_flat))
    cells_ds["offset"][:] = cells_flat

    # --- nb_idxs.mmappet: int32[M] ---
    nb_scheme = pd.DataFrame({"idx": pd.Series(dtype=np.int32)})
    nb_ds = mmappet.open_new_dataset_dct(out_dir / "nb_idxs.mmappet", nb_scheme, len(nb_idxs_remapped))
    nb_ds["idx"][:] = nb_idxs_remapped

    # --- grid_meta.txt ---
    bx, by, bz_plus_1 = idx.cells.shape
    meta = {
        "bx":         bx,
        "by":         by,
        "bz_plus_1":  bz_plus_1,
        "xwidth":     idx.xwidth,
        "ywidth":     idx.ywidth,
        "zwidth":     idx.zwidth,
        "mz_offset":  boxes3d.mz_offset,
    }
    with open(out_dir / "grid_meta.txt", "w") as f:
        for k, v in meta.items():
            f.write(f"{k} = {v}\n")

    print(
        f"Done. N={N}, cells={idx.cells.shape}, "
        f"nb_idxs={len(nb_idxs_remapped)}, has_intensity={has_intensity}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_path", type=Path, help="Precursor parquet file")
    parser.add_argument("output_dir",   type=Path, help="Output directory")
    parser.add_argument("--mz-radius-da", type=float, required=True,
                        help="DIA isolation half-width in Da")
    parser.add_argument("--frame-mult", type=float, default=2.5)
    parser.add_argument("--scan-mult",  type=float, default=2.5)
    args = parser.parse_args()
    build(args.parquet_path, args.output_dir,
          args.mz_radius_da, args.frame_mult, args.scan_mult)


if __name__ == "__main__":
    main()
