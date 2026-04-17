"""Parallel connected-component labelling for 3D box/ellipsoid neighborhoods."""
from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit
def first_candidate(xmax_arr: NDArray, xmin_query: float) -> int:
    """Binary search: return first index where xmax_arr[idx] >= xmin_query."""
    lo = 0
    hi = xmax_arr.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if xmax_arr[mid] < xmin_query:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit
def find_nbs_new(
    locs: NDArray,
    scales: NDArray,
    mult_tof: float,
    mult_urt: float,
    mult_scan: float,
    i: int,
    xmin_arr: NDArray,
    xmax_arr: NDArray,
    use_ellipsoid: bool,
    out: NDArray | None = None,
) -> int:
    """Unified neighbor finder: box intersection with optional Mahalanobis gate.

    First checks per-axis box overlap in all three dimensions (tof via the
    binary-search loop bounds, urt and scan explicitly).  When
    ``use_ellipsoid=True`` a pair that passes all three box checks is accepted
    only if the Mahalanobis distance between the two centres under the averaged
    diagonal covariance is ≤ 1:

        Σ_d ( Δloc_d / ((scales[i,d] + scales[j,d]) * mult_d) )² ≤ 1

    Parameters
    ----------
    locs         : (N, 3) float64 [tof, urt, scan], sorted by xmin_arr.
    scales       : (N, 3) float64 [tof_scale, urt_scale, scan_scale].
    mult_tof/urt/scan : per-dimension multipliers.
    i            : query index.
    xmin_arr     : precomputed locs[:,0] - mult_tof * scales[:,0]  (sorted).
    xmax_arr     : precomputed locs[:,0] + mult_tof * scales[:,0].
    use_ellipsoid: False → axis-aligned box intersection only.
                   True  → box pre-filter + Mahalanobis ≤ 1.
    out          : optional int64 buffer to write neighbour indices into.
    """
    cnt = 0
    N = locs.shape[0]

    tof_i   = locs[i, 0];   urt_i   = locs[i, 1];   scan_i   = locs[i, 2]
    stof_i  = scales[i, 0]; surt_i  = scales[i, 1];  sscan_i  = scales[i, 2]

    j = first_candidate(xmax_arr, xmin_arr[i])
    while j < N and xmin_arr[j] <= xmax_arr[i]:
        surt_sum  = surt_i  + scales[j, 1]
        sscan_sum = sscan_i + scales[j, 2]
        if abs(urt_i - locs[j, 1]) <= mult_urt * surt_sum:
            if abs(scan_i - locs[j, 2]) <= mult_scan * sscan_sum:
                neighbor = True
                if use_ellipsoid:
                    stof_sum = stof_i + scales[j, 0]
                    dtof  = (tof_i  - locs[j, 0]) / (mult_tof  * stof_sum)
                    durt  = (urt_i  - locs[j, 1]) / (mult_urt  * surt_sum)
                    dscan = (scan_i - locs[j, 2]) / (mult_scan * sscan_sum)
                    neighbor = dtof*dtof + durt*durt + dscan*dscan <= 1.0
                if neighbor:
                    if out is not None:
                        out[cnt] = j
                    cnt += 1
        j += 1
    return cnt


@numba.njit(parallel=True)
def get_connected_components_new(
    locs: NDArray,
    scales: NDArray,
    mult_tof: float,
    mult_urt: float,
    mult_scan: float,
    use_ellipsoid: bool = False,
):
    """Parallel connected-components labelling via two-pass CSR + sequential DFS.

    Parameters
    ----------
    locs          : (N, 3) float64 [tof, urt, scan],
                    sorted by  locs[:,0] - mult_tof * scales[:,0].
    scales        : (N, 3) float64 [tof_scale, urt_scale, scan_scale].
    mult_tof/urt/scan : per-dimension multipliers.
    use_ellipsoid : False (default) → axis-aligned box intersection.
                    True            → box pre-filter + Mahalanobis ≤ 1.

    Returns
    -------
    labels   : np.ndarray[uint32], shape (N,)  — 1-indexed component labels.
    cc_count : int                             — total number of components.
    """
    N = locs.shape[0]
    xmin_arr = locs[:, 0] - mult_tof * scales[:, 0]
    xmax_arr = locs[:, 0] + mult_tof * scales[:, 0]

    for i in range(1, N):
        assert xmin_arr[i] >= xmin_arr[i - 1], \
            "locs must be sorted by locs[:,0] - mult_tof * scales[:,0]"

    # Pass 1 (parallel): count neighbors per node
    counts = np.zeros(N, dtype=np.int64)
    for i in numba.prange(N):
        counts[i] = find_nbs_new(
            locs, scales, mult_tof, mult_urt, mult_scan, i, xmin_arr, xmax_arr, use_ellipsoid
        )

    # Prefix sum (sequential): build CSR offsets
    offsets = np.empty(N + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(N):
        offsets[i + 1] = offsets[i] + counts[i]

    # Pass 2 (parallel): fill flat adjacency array
    adj = np.empty(offsets[N], dtype=np.int64)
    for i in numba.prange(N):
        find_nbs_new(
            locs, scales, mult_tof, mult_urt, mult_scan, i, xmin_arr, xmax_arr, use_ellipsoid,
            adj[offsets[i]:offsets[i + 1]],
        )

    # Sequential DFS on CSR graph
    labels = np.zeros(N, dtype=np.uint32)
    cc_idx = np.uint32(0)
    stack = [np.int64(0)]
    stack.pop()

    for node in range(N):
        if labels[node] != 0:
            continue

        cc_idx += np.uint32(1)
        labels[node] = cc_idx
        stack.append(np.int64(node))

        while len(stack) > 0:
            v = stack.pop()
            for k in range(offsets[v], offsets[v + 1]):
                u = adj[k]
                if u != v and labels[u] == 0:
                    labels[u] = cc_idx
                    stack.append(np.int64(u))

    return labels, int(cc_idx)
