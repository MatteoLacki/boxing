"""Brute-force reference implementations and validators for boxing functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def brute_force_top_k_neighbors_2d_zz(
    i: int,
    boxes: NDArray,
    intensities: NDArray,
    top_k: int,
) -> list[int]:
    """Return exact top-k neighbour indices for box i by exhaustive search.

    boxes : (N, 6) array with columns [xx_lo, xx_hi, yy_lo, yy_hi, zz_lo, zz_hi].
    A neighbour j is any box (j != i) whose intervals overlap on all three axes.
    Among all neighbours, return the indices of the top_k with highest intensity.
    When there are fewer than top_k neighbours, return all of them.
    Tie-breaking at the boundary is arbitrary (argpartition order).
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
        return positions.tolist()
    intens = intensities[positions]
    top_local = np.argpartition(intens, -top_k)[-top_k:]
    return positions[top_local].tolist()


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
        all_positions = np.where(mask)[0]

        valid_slots = neighbor_ids[i] >= 0
        actual_ids = set(neighbor_ids[i][valid_slots].tolist())
        actual_intens = neighbor_ints[i][valid_slots]

        spurious = actual_ids - set(all_positions.tolist())
        if spurious:
            mismatches.append((i, f"spurious ids: {spurious}"))
            continue

        expected_count = min(top_k, len(all_positions))
        if len(actual_ids) != expected_count:
            mismatches.append((i, f"count {len(actual_ids)} != {expected_count}"))
            continue

        if len(actual_intens) > 0:
            min_kept = int(actual_intens.min())
            excluded_ids = np.array(
                list(set(all_positions.tolist()) - actual_ids), dtype=np.int64
            )
            if len(excluded_ids) > 0:
                excluded_intens = intensities[excluded_ids]
                if (excluded_intens > min_kept).any():
                    mismatches.append(
                        (i, f"better neighbour excluded (min_kept={min_kept})")
                    )

    return mismatches
