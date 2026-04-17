"""Tests for boxing.connected_components."""
import numpy as np
import pytest

from boxing.connected_components import get_connected_components_new


def _make_locs_scales(centers, half_widths):
    """Build (locs, scales) sorted by tof_start = locs[:,0] - scales[:,0]."""
    locs = np.array(centers, dtype=np.float64)
    scales = np.array(half_widths, dtype=np.float64)
    order = np.argsort(locs[:, 0] - scales[:, 0])
    return locs[order], scales[order]


def test_two_overlapping_boxes_one_component():
    locs, scales = _make_locs_scales(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
    )
    labels, cnt = get_connected_components_new(locs, scales, 1.0, 1.0, 1.0)
    assert cnt == 1
    assert labels[0] == labels[1]


def test_two_isolated_boxes_two_components():
    locs, scales = _make_locs_scales(
        [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]],
        [[1.0, 1.0, 1.0], [1.0,   1.0,   1.0]],
    )
    labels, cnt = get_connected_components_new(locs, scales, 1.0, 1.0, 1.0)
    assert cnt == 2
    assert labels[0] != labels[1]


def test_chain_three_boxes_one_component():
    # box0 overlaps box1, box1 overlaps box2, but box0 does not overlap box2
    locs, scales = _make_locs_scales(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    )
    labels, cnt = get_connected_components_new(locs, scales, 1.0, 1.0, 1.0)
    assert cnt == 1
    assert labels[0] == labels[1] == labels[2]


def test_single_box():
    locs, scales = _make_locs_scales([[5.0, 5.0, 5.0]], [[1.0, 1.0, 1.0]])
    labels, cnt = get_connected_components_new(locs, scales, 1.0, 1.0, 1.0)
    assert cnt == 1
    assert labels[0] == 1


def test_ellipsoid_mode_rejects_corner_neighbor():
    # Two boxes touch at corners — pass box check but fail Mahalanobis ≤ 1
    locs, scales = _make_locs_scales(
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    )
    # Box mode: they overlap (box touching is still overlap)
    _, cnt_box = get_connected_components_new(locs, scales, 1.0, 1.0, 1.0, use_ellipsoid=False)
    # Ellipsoid mode: Mahalanobis distance = sqrt(3) > 1 → separate components
    _, cnt_ell = get_connected_components_new(locs, scales, 1.0, 1.0, 1.0, use_ellipsoid=True)
    assert cnt_box == 1
    assert cnt_ell == 2


def test_labels_are_one_indexed():
    locs, scales = _make_locs_scales(
        [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
        [[1.0, 1.0, 1.0], [1.0,  1.0,  1.0]],
    )
    labels, cnt = get_connected_components_new(locs, scales, 1.0, 1.0, 1.0)
    assert set(labels.tolist()) == {1, 2}
