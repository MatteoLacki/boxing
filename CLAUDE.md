# CLAUDE.md — boxing

## Package layout

```
src/boxing/
├── spatial_index.py        # 2D spatial index + top-k neighbor search (core)
├── connected_components.py # parallel connected-component labelling
├── utils.py                # counting sort, dtype helpers (Numba)
├── testing.py              # brute-force validators and reference implementations
├── __init__.py             # public API re-exports
└── cli/
    └── precursor_neighbors.py  # precursor-neighbors CLI (pipeline entry point)

tests/
├── test_spatial_index.py
└── test_connected_components.py

configs/precursor_neighbors/
└── default.toml            # neighbor geometry and radii/multiplier defaults
```

Entry point registered in `pyproject.toml`:
```
precursor-neighbors = boxing.cli.precursor_neighbors:main
```

---

## Core: `spatial_index.py`

### Centered geometry format
The public top-k API accepts:
- `centers`: `(N, 3)` columns `[xx, yy, mz]`
- `scales`: `(N, 2)` columns `[xx_scale, yy_scale]`
- `xy_mults`: first-two-dimension support multipliers
- `mz_radius_da`: scalar outer m/z half-width

Internally, the implementation derives endpoint boxes for bucket lookup. Overlap is strict open-interval on the internal bounds.

### Spatial index structure
```
row_starts:   int64[BX + 1]      absolute start of each xx-bucket in flat_members
cell_offsets: int64[BX, BY + 1]  per-row yy cumsum offsets (row-local)
flat_members: int32[M]           concatenated box indices per cell
```
Access cell (bx, by):
```python
start = row_starts[bx] + cell_offsets[bx, by]
end   = row_starts[bx] + cell_offsets[bx, by + 1]
```
Bucket assignment is half-open: last bucket = `(hi - 1) // width`, clamped to `[0, n_buckets - 1]`.

### Key functions

#### `find_top_k_neighbors_2d_zz(centers, scales, mz_radius_da, intensities, top_k, ..., xy_mults=(1,1), geometry="box", cylinder_radius=1.0, precursor_idxs=None, inner_xy_mults=None, inner_mz_radius_da=None)`
Top-level Python function (not JIT). Builds centered supports, sorts boxes, builds the internal index, runs visitor, and returns dense top-k neighbors.

- `neighbor_ids`: int32[N, top_k] — neighbor precursor ids when `precursor_idxs` is provided; -1 = empty slot
- `neighbor_ints`: int64[N, top_k] — corresponding intensities; 0 = empty slot
- `prec_to_row`: int32 mapping precursor_idx → row in result arrays when precursor ids are provided
- Zero-intensity neighbors are never recorded (sentinel).
- `geometry="box"` uses axis-aligned overlap in all dimensions.
- `geometry="cylinder"` requires z overlap and applies `cylinder_radius` to the normalized first-two-dimension center distance.

#### Shell filter
When `inner_xy_mults` is provided and either multiplier is positive, candidates are rejected if their outer support intersects the query precursor's inner support. `inner_mz_radius_da` defaults to the outer `mz_radius_da`.

#### `dense_neighbors_to_csr(neighbor_ids, neighbor_ints=None, prec_to_row=None, out_path=None)`
Converts dense (N, K) arrays to CSR. When `prec_to_row` provided, offsets are indexed by precursor index. When `out_path` provided, writes mmappet datasets (`neighbors.mmappet`, `index.mmappet`).

---

## CLI: `precursor_neighbors.py`

Reads precursor parquet, reads DIA isolation width from OpenTIMS, passes frame/scan/mz centers plus frame/scan scales and m/z radii to `find_top_k_neighbors_2d_zz`, then writes CSR mmappet output.

### `PrecursorNeighborsConfig` (Pydantic)

| field | default | constraint | meaning |
|---|---|---|---|
| `frame_mult` | 1.0 | gt=0 | outer box half-width in frame units × frame_scale |
| `scan_mult` | 1.0 | gt=0 | outer box half-width in scan units × scan_scale |
| `frame_inner_mult` | 0.0 | ge=0 | inner box half-width; 0 = no shell filter |
| `scan_inner_mult` | 0.0 | ge=0 | inner box half-width; 0 = no shell filter |
| `mz_inner_radius_da` | None | ge=0 | if set, inner m/z bounds use mz ± mz_inner_radius_da; otherwise use outer m/z radius |
| `top_k` | 64 | gt=0 | maximum neighbors per precursor |
| `geometry` | "box" | "box" or "cylinder" | first-two-dimension support geometry |
| `cylinder_radius` | 1.0 | gt=0 | normalized 2D cylinder radius when `geometry="cylinder"` |

Config loaded from `--config TOML` (optional). Falls back to model defaults when omitted.
Config validated by `check_configs` pipeline rule via `scripts/check_configs.py`.

### Output
Directory with two mmappet datasets:
- `neighbors.mmappet` — `prec_idx` (int32), `intensity` (int64)
- `index.mmappet` — `offset` (int64)

---

## `testing.py` — reference implementations

### `brute_force_intersections_zz(boxes_a, boxes_b) → NDArray[int64, (M, 2)]`
Numba parallel two-pass. Returns all (i, j) pairs where box i in A overlaps box j in B.

### `brute_force_top_k_neighbors_2d_zz(...) → list[int]`
Exhaustive O(N) search for centered top-k geometry.

### `validate_top_k_neighbors_2d_zz(...) → list[tuple]`
Checks a random sample of boxes against brute force. Returns `(box_idx, reason)` for mismatches. Supports shell filter. Validity criteria:
- No spurious ids
- Correct count: `min(top_k, genuine_neighbors)`
- No excluded neighbor strictly more intense than the weakest kept (ties at boundary allowed)

---

## `connected_components.py`

`get_connected_components_new(locs, scales, mult_tof, mult_urt, mult_scan, use_ellipsoid=False)`

Parallel two-pass CSR build + sequential DFS. Input `locs`: (N, 3) float64 `[tof, urt, scan]`, pre-sorted by `tof - mult_tof * scales`. Returns `(labels uint32[N], n_components)`, labels 1-indexed. Optionally applies Mahalanobis distance gating (`use_ellipsoid=True`).

---

## Tests (`tests/`)

Run with `pytest` from the boxing root.

- `test_spatial_index.py` — unit tests for index build, top-k correctness, CSR conversion, precursor_idxs remapping. Includes a `validate_top_k_neighbors_2d_zz` round-trip test. One test is skipped unless `temp/dev_intersection_boxes.parquet` exists.
- `test_connected_components.py` — connected component tests.

Key test fixture `zz_boxes`: 4 boxes with known m/z-filtered neighbor graph; all tests are hand-verifiable.
