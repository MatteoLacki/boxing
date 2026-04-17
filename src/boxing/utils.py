"""Minimal counting and sorting utilities used by the spatial index."""
from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit
def _get_min_int_data_type(x, signed=True):
    if signed:
        if x <= 2**7 - 1:
            return 8
        elif x <= 2**15 - 1:
            return 16
        elif x <= 2**31 - 1:
            return 32
        else:
            return 64
    else:
        if x <= 2**8 - 1:
            return 8
        elif x <= 2**16 - 1:
            return 16
        elif x <= 2**32 - 1:
            return 32
        else:
            return 64


def get_min_int_data_type(x, signed: bool = True):
    """Return the minimal numpy dtype capable of representing index x."""
    bits = _get_min_int_data_type(x, signed)
    return np.dtype(f"{'int' if signed else 'uint'}{bits}")


@numba.njit
def count1D(
    xx: NDArray,
    counts: NDArray | None = None,
    mask: NDArray | None = None,
) -> NDArray:
    """Count occurrences of non-negative integer values in a 1D array.

    >>> import numpy as np
    >>> count1D(np.array([0, 1, 1, 2, 0, 2, 2]))
    array([2, 2, 3], dtype=uint32)
    """
    if counts is None:
        counts = np.zeros(shape=int(xx.max()) + 1, dtype=np.uint32)

    have_mask = True
    if mask is None:
        have_mask = False
        mask = np.array([True], np.bool_)

    for i, x in enumerate(xx):
        counts[x] += mask[i * have_mask]

    return counts


@numba.njit
def _argcountsort(xx: NDArray, counts: NDArray, order: NDArray) -> None:
    cumsums = np.cumsum(counts)
    for i in range(len(xx) - 1, -1, -1):
        x = xx[i]
        cumsums[x] -= np.uint32(1)
        j = cumsums[x]
        order[j] = i


def argcountsort(
    xx: NDArray,
    counts: NDArray | None = None,
    order: NDArray | None = None,
    return_counts: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """Stable argsort via counting sort. Returns indices such that xx[order] is sorted.

    >>> import numpy as np
    >>> order = argcountsort(np.array([2, 0, 1, 2, 1]))
    >>> order
    array([1, 2, 4, 0, 3])
    """
    assert len(xx.shape) == 1, "Input `xx` must be 1D."
    counts = count1D(xx) if counts is None else counts
    if order is None:
        order = np.empty(xx.shape, get_min_int_data_type(len(xx), signed=False))
    assert xx.shape == order.shape
    _argcountsort(xx, counts, order)
    if return_counts:
        return order, counts
    return order
