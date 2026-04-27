"""Config and metadata helpers for precursor-neighbor computation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
from opentimspy import OpenTIMS
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PrecursorNeighborsConfig(BaseModel):
    """Scale multipliers and geometry options for precursor-neighbor supports."""

    model_config = ConfigDict(extra="ignore")

    frame_mult: float = Field(default=1.0, gt=0)
    scan_mult: float = Field(default=1.0, gt=0)
    frame_inner_mult: float = Field(default=0.0, ge=0)
    scan_inner_mult: float = Field(default=0.0, ge=0)
    mz_inner_radius_da: float | None = Field(default=None, ge=0)
    top_k: int = Field(default=64, gt=0)
    geometry: Literal["box", "cylinder"] = "box"
    cylinder_radius: float = Field(default=1.0, gt=0)


def validate_config(
    config_path: Path | None,
    *,
    require_cpp_backend_compat: bool = False,
) -> PrecursorNeighborsConfig:
    if config_path is None:
        cfg = PrecursorNeighborsConfig()
    else:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        raw = raw.get("precursor_neighbors", raw)
        try:
            cfg = PrecursorNeighborsConfig(**raw)
        except ValidationError as exc:
            lines = [f"Config error in {config_path} [precursor_neighbors]:"]
            for err in exc.errors():
                loc = (
                    " -> ".join(str(p) for p in err["loc"])
                    if err["loc"]
                    else "(top level)"
                )
                lines.append(f"  {loc}: {err['msg']}")
            raise ValueError("\n".join(lines)) from exc

    if require_cpp_backend_compat and cfg.cylinder_radius != 1.0:
        raise ValueError(
            f"Config error in {config_path} [precursor_neighbors]:\n"
            "  cylinder_radius: Python-backend only; C++ backend supports only "
            "cylinder_radius = 1.0"
        )
    return cfg


def read_isolation_mz_radius_da(data_handler) -> float:
    """Return the DIA isolation half-width in m/z units."""
    windows = data_handler.table2dict("DiaFrameMsMsWindows")
    isolation_widths = np.unique(np.asarray(windows["IsolationWidth"], dtype=float))
    if len(isolation_widths) != 1:
        raise ValueError(f"Expected a single IsolationWidth, got {isolation_widths}")
    return float(isolation_widths[0]) / 2.0


def read_outer_mz_radius_da(dataset_path: Path) -> float:
    """Read the outer m/z radius from DIA isolation metadata."""
    data_handler = OpenTIMS(dataset_path)
    return read_isolation_mz_radius_da(data_handler)
