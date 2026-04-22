"""Brute-force reference implementations and validators for boxing functions."""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(parallel=True)
def _count_intersections_zz(boxes_a, boxes_b):
    n_a = boxes_a.shape[0]
    n_b = boxes_b.shape[0]
    counts = np.zeros(n_a, dtype=np.int64)
    for i in prange(n_a):
        c = np.int64(0)
        for j in range(n_b):
            if (boxes_a[i, 0] < boxes_b[j, 1] and boxes_b[j, 0] < boxes_a[i, 1] and
                    boxes_a[i, 2] < boxes_b[j, 3] and boxes_b[j, 2] < boxes_a[i, 3] and
                    boxes_a[i, 4] < boxes_b[j, 5] and boxes_b[j, 4] < boxes_a[i, 5]):
                c += 1
        counts[i] = c
    return counts


@njit(parallel=True)
def _fill_intersections_zz(boxes_a, boxes_b, offsets, out):
    n_a = boxes_a.shape[0]
    n_b = boxes_b.shape[0]
    for i in prange(n_a):
        k = offsets[i]
        for j in range(n_b):
            if (boxes_a[i, 0] < boxes_b[j, 1] and boxes_b[j, 0] < boxes_a[i, 1] and
                    boxes_a[i, 2] < boxes_b[j, 3] and boxes_b[j, 2] < boxes_a[i, 3] and
                    boxes_a[i, 4] < boxes_b[j, 5] and boxes_b[j, 4] < boxes_a[i, 5]):
                out[k, 0] = i
                out[k, 1] = j
                k += 1


def brute_force_intersections_zz(
    boxes_a: NDArray,
    boxes_b: NDArray,
) -> NDArray:
    """Return all (i, j) pairs where box i in boxes_a overlaps box j in boxes_b.

    boxes_a, boxes_b : (N, 6) arrays with columns [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi].
    Overlap is strict (open intervals): a_lo < b_hi and b_lo < a_hi on every axis.

    Returns shape (M, 2) int64 array of (i, j) pairs, parallelised over rows of boxes_a.
    """
    boxes_a = np.ascontiguousarray(boxes_a, dtype=np.int64)
    boxes_b = np.ascontiguousarray(boxes_b, dtype=np.int64)
    counts = _count_intersections_zz(boxes_a, boxes_b)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    out = np.empty((int(offsets[-1]), 2), dtype=np.int64)
    _fill_intersections_zz(boxes_a, boxes_b, offsets[:-1], out)
    return out


def brute_force_top_k_neighbors_2d_zz(
    i: int,
    boxes: NDArray,
    intensities: NDArray,
    top_k: int,
    precursor_idxs: NDArray | None = None,
) -> list[int]:
    """Return exact top-k neighbour indices for box i by exhaustive search.

    boxes : (N, 6) array with columns [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi].
    A neighbour j is any box (j != i) whose intervals overlap on all three axes.
    Among all neighbours, return the indices of the top_k with highest intensity.
    When there are fewer than top_k neighbours, return all of them.
    Tie-breaking at the boundary is arbitrary (argpartition order).
    precursor_idxs : when provided, returned ids are precursor indices rather
        than box indices.
    """
    boxes = np.asarray(boxes)
    intensities = np.asarray(intensities)
    xx_lo, xx_hi = boxes[:, 0], boxes[:, 1]
    yy_lo, yy_hi = boxes[:, 2], boxes[:, 3]
    zz_lo, zz_hi = boxes[:, 4], boxes[:, 5]

    mask = (
        (xx_lo < xx_hi[i]) & (xx_lo[i] < xx_hi) &
        (yy_lo < yy_hi[i]) & (yy_lo[i] < yy_hi) &
        (zz_lo < zz_hi[i]) & (zz_lo[i] < zz_hi)
    )
    mask[i] = False
    positions = np.where(mask)[0]
    if len(positions) <= top_k:
        result = positions
    else:
        intens = intensities[positions]
        top_local = np.argpartition(intens, -top_k)[-top_k:]
        result = positions[top_local]
    if precursor_idxs is not None:
        return np.asarray(precursor_idxs)[result].tolist()
    return result.tolist()


def validate_top_k_neighbors_2d_zz(
    boxes: NDArray,
    intensities: NDArray,
    neighbor_ids: NDArray,
    neighbor_ints: NDArray,
    top_k: int,
    *,
    indices: NDArray | None = None,
    K: int = 100,
    seed: int = 42,
    precursor_idxs: NDArray | None = None,
) -> list[tuple]:
    """Validate neighbor_ids / neighbor_ints against brute-force results.

    boxes : (N, 6) array with columns [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi].

    Parameters
    ----------
    indices:
        Exact box indices to check.  When None, K random indices are sampled
        using the given seed.
    K, seed:
        Used only when indices is None.
    precursor_idxs : when provided, neighbor_ids are expected to contain
        precursor indices (precursor_idxs[j]) rather than box indices (j).

    Returns
    -------
    list of (box_idx, reason_str) for every mismatch.  Empty → all valid.

    Validity criteria
    -----------------
    - No spurious ids: every returned id is a genuine neighbour.
    - Correct count: min(top_k, total_genuine_neighbours).
    - No excluded neighbour strictly more intense than the weakest kept.
      (Ties at the boundary are allowed to break either way.)
    """
    boxes = np.asarray(boxes)
    intensities = np.asarray(intensities)
    xx_lo, xx_hi = boxes[:, 0], boxes[:, 1]
    yy_lo, yy_hi = boxes[:, 2], boxes[:, 3]
    zz_lo, zz_hi = boxes[:, 4], boxes[:, 5]

    if precursor_idxs is not None:
        precursor_idxs = np.asarray(precursor_idxs)

    N = len(boxes)
    if indices is None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(N, size=K, replace=False)

    mismatches: list[tuple] = []

    for i in indices:
        i = int(i)
        mask = (
            (xx_lo < xx_hi[i]) & (xx_lo[i] < xx_hi) &
            (yy_lo < yy_hi[i]) & (yy_lo[i] < yy_hi) &
            (zz_lo < zz_hi[i]) & (zz_lo[i] < zz_hi)
        )
        mask[i] = False
        all_positions = np.where(mask)[0]  # box indices of genuine neighbours

        if precursor_idxs is not None:
            all_ids = set(precursor_idxs[all_positions].tolist())
        else:
            all_ids = set(all_positions.tolist())

        valid_slots = neighbor_ids[i] >= 0
        actual_ids = set(neighbor_ids[i][valid_slots].tolist())
        actual_intens = neighbor_ints[i][valid_slots]

        spurious = actual_ids - all_ids
        if spurious:
            mismatches.append((i, f"spurious ids: {spurious}"))
            continue

        expected_count = min(top_k, len(all_positions))
        if len(actual_ids) != expected_count:
            mismatches.append((i, f"count {len(actual_ids)} != {expected_count}"))
            continue

        if len(actual_intens) > 0:
            min_kept = int(actual_intens.min())
            # Excluded neighbours: genuine positions whose id is not in actual_ids.
            if precursor_idxs is not None:
                excluded_positions = all_positions[
                    ~np.isin(precursor_idxs[all_positions], list(actual_ids))
                ]
            else:
                excluded_positions = np.array(
                    list(all_ids - actual_ids), dtype=np.int64
                )
            if len(excluded_positions) > 0:
                excluded_intens = intensities[excluded_positions]
                if (excluded_intens > min_kept).any():
                    mismatches.append(
                        (i, f"better neighbour excluded (min_kept={min_kept})")
                    )

    return mismatches
