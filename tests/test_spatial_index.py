"""Tests for spatial_index_box_intersection."""
from pathlib import Path

import numpy as np
import pytest

from boxing.spatial_index import (
    box_widths_2d,
    build_spatial_index_2d,
    count_cell_memberships,
    find_top_k_neighbors_2d_zz,
    get_cell_members,
    get_multiplied_median_bucket_widths,
)
from boxing.testing import validate_top_k_neighbors_2d_zz


# ---------------------------------------------------------------------------
# Tiny hand-checkable example
# ---------------------------------------------------------------------------
#
# 3 boxes, bucket_width = 10, grid BF=2 x BS=3:
#
#   Box 0: frame [0,  5), scan [ 0,  5)
#     frame buckets: 0//10=0 .. (5-1)//10=0  → {0}
#     scan  buckets: 0//10=0 .. (5-1)//10=0  → {0}
#     cells: (0,0)
#
#   Box 1: frame [0, 15), scan [ 5, 20)
#     frame buckets: 0//10=0 .. (15-1)//10=1 → {0,1}
#     scan  buckets: 5//10=0 .. (20-1)//10=1 → {0,1}
#     cells: (0,0),(0,1),(1,0),(1,1)
#
#   Box 2: frame [10, 20), scan [20, 25)
#     frame buckets: 10//10=1 .. (20-1)//10=1 → {1}
#     scan  buckets: 20//10=2 .. (25-1)//10=2 → {2}
#     cells: (1,2)
#
# Expected cell contents:
#   (0,0): {0,1}   (0,1): {1}   (0,2): {}
#   (1,0): {1}     (1,1): {1}   (1,2): {2}


@pytest.fixture
def tiny_boxes():
    return np.array([
        [0,  5,  0,  5],
        [0, 15,  5, 20],
        [10, 20, 20, 25],
    ], dtype=np.uint32)


def test_count_cell_memberships_tiny(tiny_boxes):
    counts = count_cell_memberships(tiny_boxes, 10, 10, 2, 3)
    expected = np.array([[2, 1, 0], [1, 1, 1]], dtype=np.int64)
    np.testing.assert_array_equal(counts, expected)


def test_build_spatial_index_tiny(tiny_boxes):
    rs, co, fm = build_spatial_index_2d(tiny_boxes, 10, 10, 2, 3)

    def cell(bf, bs):
        return set(get_cell_members(rs, co, fm, bf, bs).tolist())

    assert cell(0, 0) == {0, 1}
    assert cell(0, 1) == {1}
    assert cell(0, 2) == set()
    assert cell(1, 0) == {1}
    assert cell(1, 1) == {1}
    assert cell(1, 2) == {2}


def test_build_index_total_member_count(tiny_boxes):
    """Total flat_members length = sum of all per-cell memberships."""
    rs, co, fm = build_spatial_index_2d(tiny_boxes, 10, 10, 2, 3)
    # Box 0: 1 cell, Box 1: 4 cells, Box 2: 1 cell → 6 total
    assert len(fm) == 6


def test_build_index_invalid_bucket_width(tiny_boxes):
    with pytest.raises(ValueError):
        build_spatial_index_2d(tiny_boxes, 0, 10, 2, 3)
    with pytest.raises(ValueError):
        build_spatial_index_2d(tiny_boxes, 10, -1, 2, 3)


def test_box_spanning_full_axis_clamped():
    """Box wider than the grid is clamped to the last bucket, not out-of-bounds."""
    boxes = np.array([[0, 1000, 0, 1000]], dtype=np.uint32)
    # grid only has 2 buckets of width 10 → max bucket index = 1
    rs, co, fm = build_spatial_index_2d(boxes, 10, 10, 2, 2)
    # Box must appear in all 4 cells (2×2)
    for bf in range(2):
        for bs in range(2):
            assert 0 in set(get_cell_members(rs, co, fm, bf, bs).tolist())


def test_bucket_boundary_touching():
    """Box whose hi lands exactly on a bucket boundary: (hi-1)//width is correct."""
    # Box frame [0, 10) with width=10 → last_bucket = (10-1)//10 = 0 → only bucket 0
    boxes = np.array([[0, 10, 0, 10]], dtype=np.uint32)
    rs, co, fm = build_spatial_index_2d(boxes, 10, 10, 2, 2)
    assert set(get_cell_members(rs, co, fm, 0, 0).tolist()) == {0}
    assert set(get_cell_members(rs, co, fm, 1, 0).tolist()) == set()


# ---------------------------------------------------------------------------
# Smoke test on real data
# ---------------------------------------------------------------------------

_REAL_DATA = (
    Path(__file__).parent.parent.parent.parent
    / "temp"
    / "dev_intersection_boxes.parquet"
)


# ---------------------------------------------------------------------------
# find_top_k_neighbors_2d_zz
# ---------------------------------------------------------------------------
#
# 4 boxes — all pairwise overlapping in frame/scan; tof selectively filters:
#
#   idx | frame    | scan    | tof       | intensity
#    0  | [0, 10)  | [0, 10) | [ 0,  50) |  50
#    1  | [5, 15)  | [5, 15) | [ 0,  50) | 200
#    2  | [3, 12)  | [3, 12) | [60, 100) | 100
#    3  | [2,  8)  | [2,  8) | [40,  80) | 300
#
# tof overlaps:  0∩1 ✓  0∩2 ✗  0∩3 ✓  1∩2 ✗  1∩3 ✓  2∩3 ✓
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
    xx_lo = np.array([0, 5, 3, 2], dtype=np.int32)
    xx_hi = np.array([10, 15, 12, 8], dtype=np.int32)
    yy_lo = np.array([0, 5, 3, 2], dtype=np.int32)
    yy_hi = np.array([10, 15, 12, 8], dtype=np.int32)
    zz_lo = np.array([0,  0, 60, 40], dtype=np.int64)
    zz_hi = np.array([50, 50, 100, 80], dtype=np.int64)
    intensities = np.array([50, 200, 100, 300], dtype=np.int64)
    boxes = np.column_stack([xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi])
    return boxes, intensities


def test_top_k_all_neighbors_fit(zz_boxes):
    """top_k >= max degree: all neighbors recorded, empty slots are -1/0."""
    boxes, intensities = zz_boxes
    ids, ints = find_top_k_neighbors_2d_zz(boxes, intensities, top_k=3)
    assert ids.shape == (4, 3)
    assert ints.shape == (4, 3)
    assert _top_k_as_dict(ids, ints, 0) == {1: 200, 3: 300}
    assert _top_k_as_dict(ids, ints, 1) == {0: 50, 3: 300}
    assert _top_k_as_dict(ids, ints, 2) == {3: 300}
    assert _top_k_as_dict(ids, ints, 3) == {0: 50, 1: 200, 2: 100}


def test_top_k_eviction(zz_boxes):
    """top_k=2: box 3 has 3 neighbors; only the 2 most intense survive."""
    boxes, intensities = zz_boxes
    ids, ints = find_top_k_neighbors_2d_zz(boxes, intensities, top_k=2)
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
    intensities = np.array([100, 200], dtype=np.int64)
    ids, ints = find_top_k_neighbors_2d_zz(boxes, intensities, top_k=3)
    assert _top_k_as_dict(ids, ints, 0) == {}
    assert _top_k_as_dict(ids, ints, 1) == {}


def test_top_k_tof_filter_removes_pairs():
    """Two boxes overlapping in xx/yy but disjoint in zz: no neighbors recorded."""
    boxes = np.array([
        [0, 10,   0, 10,   0,  50],
        [5, 15,   5, 15, 100, 200],
    ], dtype=np.int64)
    intensities = np.array([100, 200], dtype=np.int64)
    ids, ints = find_top_k_neighbors_2d_zz(boxes, intensities, top_k=2)
    assert _top_k_as_dict(ids, ints, 0) == {}
    assert _top_k_as_dict(ids, ints, 1) == {}


@pytest.mark.skipif(not _REAL_DATA.exists(), reason="dev data not available")
def test_self_intersection_count_on_real_data():
    """Build index on 4M boxes and verify a positive self-intersection count."""
    import pandas as pd

    df = pd.read_parquet(_REAL_DATA)
    # Negative frame_lo values (-9 present in this dataset) are handled
    # correctly (clipped to 0 in build_spatial_index_2d).
    boxes_2d = df[["frame_lo", "frame_hi", "scan_lo", "scan_hi"]].to_numpy()

    fw, sw = box_widths_2d(boxes_2d)
    bw_f, bw_s = get_multiplied_median_bucket_widths(fw, sw)

    n_fb = int(boxes_2d[:, 1].max()) // bw_f + 1
    n_sb = int(boxes_2d[:, 3].max()) // bw_s + 1

    rs, co, fm = build_spatial_index_2d(boxes_2d, bw_f, bw_s, n_fb, n_sb)

    # Count broad-phase self-intersection pairs via C(n,2) per cell
    total_pairs = np.int64(0)
    for bf in range(n_fb):
        for bs in range(n_sb):
            n = int(co[bf, bs + 1] - co[bf, bs])
            total_pairs += n * (n - 1) // 2

    assert len(fm) > 0
    assert total_pairs > 0
    assert rs[-1] == len(fm)  # row_starts[-1] == total members


def test_top_k_validated_against_brute_force(zz_boxes):
    """find_top_k_neighbors_2d_zz result matches brute-force on all 4 boxes."""
    boxes, intensities = zz_boxes
    ids, ints = find_top_k_neighbors_2d_zz(boxes, intensities, top_k=3)
    mismatches = validate_top_k_neighbors_2d_zz(
        boxes, intensities, ids, ints, top_k=3,
        indices=np.arange(4),
    )
    assert mismatches == [], mismatches
