"""Tests for precursor-neighbor spatial indexing."""

import numpy as np
import pytest

from boxing.spatial_index import (
    dense_neighbors_to_csr,
    find_top_k_neighbors_2d_zz,
)
from boxing.testing import validate_top_k_neighbors_2d_zz


def _boxes_to_centered(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    centers = np.column_stack([
        (boxes[:, 0] + boxes[:, 1]) / 2.0,
        (boxes[:, 2] + boxes[:, 3]) / 2.0,
        (boxes[:, 4] + boxes[:, 5]) / 2.0,
    ])
    scales = np.column_stack([
        (boxes[:, 1] - boxes[:, 0]) / 2.0,
        (boxes[:, 3] - boxes[:, 2]) / 2.0,
    ])
    z_radii = (boxes[:, 5] - boxes[:, 4]) / 2.0
    if not np.allclose(z_radii, z_radii[0]):
        raise ValueError("test boxes must use a shared z radius")
    mz_radius_da = float(z_radii[0])
    return centers, scales, mz_radius_da


# ---------------------------------------------------------------------------
# find_top_k_neighbors_2d_zz
# ---------------------------------------------------------------------------
#
# 4 boxes — all pairwise overlapping in frame/scan; mz selectively filters:
#
#   idx | frame    | scan    | mz        | intensity
#    0  | [0, 10)  | [0, 10) | [ 0,  50) |  50
#    1  | [5, 15)  | [5, 15) | [ 0,  50) | 200
#    2  | [3, 12)  | [3, 12) | [60, 110) | 100
#    3  | [2,  8)  | [2,  8) | [40,  90) | 300
#
# mz overlaps:  0∩1 ✓  0∩2 ✗  0∩3 ✓  1∩2 ✗  1∩3 ✓  2∩3 ✓
#
# → neighbors with zz filter:
#   0: {1: 200, 3: 300}
#   1: {0:  50, 3: 300}
#   2: {3: 300}
#   3: {0:  50, 1: 200, 2: 100}


def _top_k_as_dict(neighbor_ids, neighbor_ints, i):
    """Valid (non-empty) slots of row i as {original_id: intensity}."""
    return {
        int(neighbor_ids[i, s]): int(neighbor_ints[i, s])
        for s in range(neighbor_ids.shape[1])
        if neighbor_ids[i, s] >= 0
    }


@pytest.fixture
def zz_boxes():
    xx_lo = np.array([0, 5, 3, 2], dtype=np.float64)
    xx_hi = np.array([10, 15, 12, 8], dtype=np.float64)
    yy_lo = np.array([0, 5, 3, 2], dtype=np.float64)
    yy_hi = np.array([10, 15, 12, 8], dtype=np.float64)
    zz_lo = np.array([0,  0, 60, 40], dtype=np.float64)
    zz_hi = np.array([50, 50, 110, 90], dtype=np.float64)
    intensities = np.array([50, 200, 100, 300], dtype=np.int64)
    boxes = np.column_stack([xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi])
    centers, scales, mz_radius_da = _boxes_to_centered(boxes)
    return centers, scales, mz_radius_da, intensities


def test_top_k_all_neighbors_fit(zz_boxes):
    """top_k >= max degree: all neighbors recorded, empty slots are -1/0."""
    centers, scales, mz_radius_da, intensities = zz_boxes
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3
    )
    assert ids.shape == (4, 3)
    assert ints.shape == (4, 3)
    assert _top_k_as_dict(ids, ints, 0) == {1: 200, 3: 300}
    assert _top_k_as_dict(ids, ints, 1) == {0: 50, 3: 300}
    assert _top_k_as_dict(ids, ints, 2) == {3: 300}
    assert _top_k_as_dict(ids, ints, 3) == {0: 50, 1: 200, 2: 100}


def test_top_k_eviction(zz_boxes):
    """top_k=2: box 3 has 3 neighbors; only the 2 most intense survive."""
    centers, scales, mz_radius_da, intensities = zz_boxes
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=2
    )
    assert _top_k_as_dict(ids, ints, 0) == {1: 200, 3: 300}
    assert _top_k_as_dict(ids, ints, 1) == {0: 50, 3: 300}
    assert _top_k_as_dict(ids, ints, 2) == {3: 300}
    # box 3 neighbors: {0:50, 1:200, 2:100}; top-2 by intensity = {1:200, 2:100}
    assert _top_k_as_dict(ids, ints, 3) == {1: 200, 2: 100}


def test_top_k_no_neighbors():
    """Isolated box (no overlapping neighbors): all slots empty."""
    boxes = np.array([
        [0,  10,   0,  10,  0, 50],
        [100, 110, 100, 110, 0, 50],
    ], dtype=np.int64)
    centers, scales, mz_radius_da = _boxes_to_centered(boxes)
    intensities = np.array([100, 200], dtype=np.int64)
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3
    )
    assert _top_k_as_dict(ids, ints, 0) == {}
    assert _top_k_as_dict(ids, ints, 1) == {}


def test_top_k_mz_filter_removes_pairs():
    """Two boxes overlapping in xx/yy but disjoint in zz: no neighbors recorded."""
    boxes = np.array([
        [0, 10,   0, 10,   0,  50],
        [5, 15,   5, 15, 125, 175],
    ], dtype=np.int64)
    centers, scales, mz_radius_da = _boxes_to_centered(boxes)
    intensities = np.array([100, 200], dtype=np.int64)
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=2
    )
    assert _top_k_as_dict(ids, ints, 0) == {}
    assert _top_k_as_dict(ids, ints, 1) == {}


def test_top_k_cylinder_ignores_zz_distance():
    """Cylinder mode filters xx/yy center distance only; zz remains overlap-only."""
    boxes = np.array([
        [0, 10, 0, 10, 0, 100],
        [2, 12, 2, 12, 99, 199],
    ], dtype=np.int64)
    centers, scales, mz_radius_da = _boxes_to_centered(boxes)
    intensities = np.array([100, 200], dtype=np.int64)

    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers,
        scales,
        mz_radius_da,
        intensities,
        top_k=1,
        geometry="cylinder",
        cylinder_radius=1.0,
    )

    assert _top_k_as_dict(ids, ints, 0) == {1: 200}
    assert _top_k_as_dict(ids, ints, 1) == {0: 100}


def test_top_k_validated_against_brute_force(zz_boxes):
    """find_top_k_neighbors_2d_zz result matches brute-force on all 4 boxes."""
    centers, scales, mz_radius_da, intensities = zz_boxes
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3
    )
    mismatches = validate_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, ids, ints, top_k=3,
        indices=np.arange(4),
    )
    assert mismatches == [], mismatches


def test_top_k_shell_filter_compiles_and_excludes_inner(zz_boxes):
    """Shell mode rejects candidates intersecting the query inner support."""
    centers, scales, mz_radius_da, intensities = zz_boxes
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers,
        scales,
        mz_radius_da,
        intensities,
        top_k=3,
        inner_xy_mults=(2.5, 2.5),
        inner_mz_radius_da=4.0,
    )
    mismatches = validate_top_k_neighbors_2d_zz(
        centers,
        scales,
        mz_radius_da,
        intensities,
        ids,
        ints,
        top_k=3,
        indices=np.arange(4),
        inner_xy_mults=(2.5, 2.5),
        inner_mz_radius_da=4.0,
    )
    assert mismatches == [], mismatches


# ---------------------------------------------------------------------------
# dense_neighbors_to_csr
# ---------------------------------------------------------------------------


def test_dense_to_csr_basic(zz_boxes):
    """CSR offsets indexed by box position (no precursor_idxs)."""
    centers, scales, mz_radius_da, intensities = zz_boxes
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3
    )
    offsets, flat_ids, flat_ints = dense_neighbors_to_csr(ids, ints)

    assert len(offsets) == 5
    assert int(offsets[-1]) == len(flat_ids) == len(flat_ints)

    def row(i):
        return set(int(x) for x in flat_ids[offsets[i]:offsets[i+1]])

    assert row(0) == {1, 3}
    assert row(1) == {0, 3}
    assert row(2) == {3}
    assert row(3) == {0, 1, 2}


def test_dense_to_csr_precursor_idxs(zz_boxes):
    """With sparse precursor_idxs, offsets[prec_idx] gives the correct neighbors."""
    centers, scales, mz_radius_da, intensities = zz_boxes
    # sparse precursor indices: boxes 0..3 → prec ids 10,20,30,40
    pidxs = np.array([10, 20, 30, 40], dtype=np.int32)
    ids, ints, prec_to_row = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3, precursor_idxs=pidxs
    )
    offsets, flat_ids, _ = dense_neighbors_to_csr(ids, ints, prec_to_row=prec_to_row)

    # offsets has size len(prec_to_row)+1 = max(pidxs)+2 = 42
    assert len(offsets) == 42

    def row(prec_idx):
        return set(int(x) for x in flat_ids[offsets[prec_idx]:offsets[prec_idx+1]])

    # box 0 (prec 10) neighbors: {1,3} → prec ids {20,40}
    assert row(10) == {20, 40}
    # box 1 (prec 20) neighbors: {0,3} → prec ids {10,40}
    assert row(20) == {10, 40}
    # box 2 (prec 30) neighbors: {3} → prec ids {40}
    assert row(30) == {40}
    # box 3 (prec 40) neighbors: {0,1,2} → prec ids {10,20,30}
    assert row(40) == {10, 20, 30}
    # gaps are empty
    assert row(15) == set()


def test_dense_to_csr_no_neighbors():
    """Isolated boxes: offsets all zero, flat_ids empty."""
    boxes = np.array([
        [0,  10,   0,  10,  0, 50],
        [100, 110, 100, 110, 0, 50],
    ], dtype=np.int64)
    centers, scales, mz_radius_da = _boxes_to_centered(boxes)
    intensities = np.array([100, 200], dtype=np.int64)
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3
    )
    offsets, flat_ids = dense_neighbors_to_csr(ids)
    assert list(offsets) == [0, 0, 0]
    assert len(flat_ids) == 0


def test_dense_to_csr_no_neighbors_mmappet(tmp_path):
    """Zero-neighbor CSR output writes readable empty mmappet neighbors."""
    ids = np.full((2, 3), -1, dtype=np.int32)
    ints = np.zeros((2, 3), dtype=np.int64)

    offsets, flat_ids, flat_ints = dense_neighbors_to_csr(
        ids, ints, out_path=tmp_path / "csr"
    )

    assert list(offsets) == [0, 0, 0]
    assert len(flat_ids) == 0
    assert len(flat_ints) == 0

    import mmappet

    neighbors = mmappet.open_dataset_dct(tmp_path / "csr" / "neighbors.mmappet")
    assert list(neighbors) == ["prec_idx", "intensity"]
    assert neighbors["prec_idx"].dtype == np.dtype(np.int32)
    assert neighbors["intensity"].dtype == np.dtype(np.int64)
    assert len(neighbors["prec_idx"]) == 0
    assert len(neighbors["intensity"]) == 0

    index = mmappet.open_dataset_dct(tmp_path / "csr" / "index.mmappet")
    np.testing.assert_array_equal(index["offset"], np.array([0, 0, 0], dtype=np.int64))


def test_top_k_validate_with_precursor_idxs(zz_boxes):
    """validate_top_k_neighbors_2d_zz accepts precursor_idxs and reports no mismatches."""
    centers, scales, mz_radius_da, intensities = zz_boxes
    pidxs = np.array([10, 20, 30, 40], dtype=np.int32)
    ids, ints, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3, precursor_idxs=pidxs
    )
    mismatches = validate_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, ids, ints, top_k=3,
        indices=np.arange(4),
        precursor_idxs=pidxs,
    )
    assert mismatches == [], mismatches


def test_top_k_precursor_idxs(zz_boxes):
    """precursor_idxs remaps recorded neighbor ids; no-arg call returns box ids."""
    centers, scales, mz_radius_da, intensities = zz_boxes
    pidxs = np.array([10, 20, 30, 40], dtype=np.int32)

    ids_prec, _, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3, precursor_idxs=pidxs
    )
    ids_box,  _, _ = find_top_k_neighbors_2d_zz(
        centers, scales, mz_radius_da, intensities, top_k=3
    )

    valid_prec = set(int(x) for x in ids_prec[ids_prec >= 0])
    valid_box  = set(int(x) for x in ids_box[ids_box >= 0])

    assert valid_prec <= {10, 20, 30, 40}
    assert valid_box  <= {0, 1, 2, 3}

    # spot-check: box 0 neighbors are {1,3} → precursor ids {20,40}
    assert set(int(x) for x in ids_prec[0][ids_prec[0] >= 0]) == {20, 40}
    assert set(int(x) for x in ids_box[0][ids_box[0] >= 0])   == {1, 3}
