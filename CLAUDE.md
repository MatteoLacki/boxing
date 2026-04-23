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
└── default.toml            # frame_mult, scan_mult, frame_inner_mult, scan_inner_mult
```

Entry point registered in `pyproject.toml`:
```
precursor-neighbors = boxing.cli.precursor_neighbors:main
```

---

## Core: `spatial_index.py`

### Box format
All functions use `(N, 6)` float64/int arrays with columns:
```
[xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi]
```
Overlap is strict open-interval: `a_lo < b_hi AND b_lo < a_hi` on every axis.

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

#### `build_spatial_index_2d(boxes_2d, bw_xx, bw_yy, n_xx, n_yy) → (row_starts, cell_offsets, flat_members)`
Two-pass CSR build (count → prefix-sum → fill). Takes 4-column 2D boxes.

#### `visit_box_intersections_2d_zz(boxes, ..., processor, processor_args, progress, ellipsoid_radius)`
`@numba.njit(parallel=True)`. Visits every intersecting (i, j) pair and calls `processor(i, j, *processor_args)`. Canonical-cell deduplication ensures each pair is visited once per direction. Optional `ellipsoid_radius` adds a 3D normalised centre-distance filter.

#### `find_top_k_neighbors_2d_zz(boxes, intensities, top_k, ..., precursor_idxs=None, inner_boxes=None) → (neighbor_ids, neighbor_ints, prec_to_row)`
Top-level Python function (not JIT). Sorts boxes, builds index, runs visitor.

- `neighbor_ids`: int32[N, top_k] — neighbor precursor (or box) ids; -1 = empty slot
- `neighbor_ints`: int64[N, top_k] — corresponding intensities; 0 = empty slot
- `prec_to_row`: int32 mapping precursor_idx → row in result arrays (-1 if absent)
- Zero-intensity neighbors are never recorded (sentinel).
- `inner_boxes`: when provided, activates shell filter (see below).

#### Shell filter (`inner_boxes`)
When `inner_boxes` is not None, uses `_top_k_neighbors_shell_processor` instead of `_top_k_neighbors_processor`. Candidate j is rejected if j's outer box intersects `inner_boxes[i]` on all three axes. When `inner_boxes` is None, original code path runs with zero overhead.

#### `dense_neighbors_to_csr(neighbor_ids, neighbor_ints=None, prec_to_row=None, out_path=None)`
Converts dense (N, K) arrays to CSR. When `prec_to_row` provided, offsets are indexed by precursor index. When `out_path` provided, writes mmappet datasets (`neighbors.mmappet`, `index.mmappet`).

#### `find_neighbors_2d_zz(boxes, ...) → (offsets, neighbors)`
Returns full symmetric neighbor graph as CSR (no top-k, no intensities).

---

## CLI: `precursor_neighbors.py`

Reads precursor parquet, builds 6D boxes (frame/scan + tof from DIA window), runs `find_top_k_neighbors_2d_zz`, writes CSR mmappet output.

### `PrecursorNeighborsConfig` (Pydantic)

| field | default | constraint | meaning |
|---|---|---|---|
| `frame_mult` | 1.0 | gt=0 | outer box half-width in frame units × frame_scale |
| `scan_mult` | 1.0 | gt=0 | outer box half-width in scan units × scan_scale |
| `frame_inner_mult` | 0.0 | ge=0 | inner box half-width; 0 = no shell filter |
| `scan_inner_mult` | 0.0 | ge=0 | inner box half-width; 0 = no shell filter |
| `mz_inner_radius_da` | None | ge=0 | if set, inner tof bounds = mz2tof(mz ± mz_inner_radius_da) instead of isolation window |

Config loaded from `--config TOML` (optional). Falls back to model defaults when omitted.
Config validated by `check_configs` pipeline rule via `scripts/check_configs.py`.

### Box SQL
`_BOXES_SQL` formats `{frame_mult}`, `{scan_mult}`, `{tof_lo_col}`, `{tof_hi_col}`. Outer boxes always use `tof_lo`/`tof_hi` (isolation window). Inner boxes use `tof_lo`/`tof_hi` by default; when `mz_inner_radius_da` is set, `_add_inner_tof_bounds` adds `tof_lo_inner`/`tof_hi_inner` columns and the inner SQL uses those instead.

### Output
Directory with two mmappet datasets:
- `neighbors.mmappet` — `prec_idx` (int32), `intensity` (int64)
- `index.mmappet` — `offset` (int64)

---

## `testing.py` — reference implementations

### `brute_force_intersections_zz(boxes_a, boxes_b) → NDArray[int64, (M, 2)]`
Numba parallel two-pass. Returns all (i, j) pairs where box i in A overlaps box j in B.

### `brute_force_top_k_neighbors_2d_zz(i, boxes, intensities, top_k, precursor_idxs=None, inner_boxes=None) → list[int]`
Exhaustive O(N) search for box i. Supports shell filter via `inner_boxes`.

### `validate_top_k_neighbors_2d_zz(boxes, intensities, neighbor_ids, neighbor_ints, top_k, *, inner_boxes=None, ...) → list[tuple]`
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

Key test fixture `zz_boxes`: 4 boxes with known tof-filtered neighbor graph; all tests are hand-verifiable.
