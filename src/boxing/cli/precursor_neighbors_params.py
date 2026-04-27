#!/usr/bin/env python3
"""Emit C++ precursor-neighbor parameters derived from config and DIA metadata."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from boxing.precursor_neighbors_config import read_outer_mz_radius_da, validate_config


def get_params(config_path: Path, dataset_path: Path) -> dict[str, float | int | str]:
    cfg = validate_config(config_path, require_cpp_backend_compat=True)
    mz_inner_radius_da = -1.0 if cfg.mz_inner_radius_da is None else cfg.mz_inner_radius_da
    return {
        "mz_radius_da": read_outer_mz_radius_da(dataset_path),
        "frame_mult": cfg.frame_mult,
        "scan_mult": cfg.scan_mult,
        "frame_inner_mult": cfg.frame_inner_mult,
        "scan_inner_mult": cfg.scan_inner_mult,
        "mz_inner_radius_da": mz_inner_radius_da,
        "top_k": cfg.top_k,
        "geometry": cfg.geometry,
    }


def format_cpp_args(params: dict[str, float | int | str]) -> str:
    args = {
        "--mz-radius-da": params["mz_radius_da"],
        "--frame-mult": params["frame_mult"],
        "--scan-mult": params["scan_mult"],
        "--frame-inner-mult": params["frame_inner_mult"],
        "--scan-inner-mult": params["scan_inner_mult"],
        "--mz-inner-radius-da": params["mz_inner_radius_da"],
        "--top-k": params["top_k"],
        "--geometry": params["geometry"],
    }
    return " ".join(
        f"{flag} {shlex.quote(str(value))}" for flag, value in args.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw parameters as JSON instead of C++ command-line flags.",
    )
    args = parser.parse_args()
    params = get_params(args.config_path, args.dataset_path)
    if args.json:
        print(json.dumps(params, sort_keys=True))
    else:
        print(format_cpp_args(params))


if __name__ == "__main__":
    main()
