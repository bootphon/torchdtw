"""Independent NumPy DTW reference used as a ground-truth oracle.

This is deliberately a separate, straightforward implementation (a full O(N*M) cost matrix
plus an explicit traceback) so the tests anchor the compiled kernels to a known-correct
result, instead of only comparing the CPU and CUDA backends against each other -- a bug
shared by both backends would slip through a pure parity test. Everything is computed in
float64. The ``min`` tie-break (diag <= left <= up) mirrors csrc/dtw.cpp so the traceback
path matches the kernels cell-for-cell.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def reference_cost(distances: npt.NDArray) -> npt.NDArray[np.float64]:
    """Accumulated DTW cost matrix, computed in float64."""
    d = np.asarray(distances, dtype=np.float64)
    n, m = d.shape
    cost = np.empty((n, m), dtype=np.float64)
    cost[0, 0] = d[0, 0]
    for i in range(1, n):
        cost[i, 0] = d[i, 0] + cost[i - 1, 0]
    for j in range(1, m):
        cost[0, j] = d[0, j] + cost[0, j - 1]
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = d[i, j] + min(cost[i - 1, j], cost[i - 1, j - 1], cost[i, j - 1])
    return cost


def reference_path(cost: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    """Optimal traceback path through ``cost`` with the diag <= left <= up tie-break."""
    n, m = cost.shape
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 and j > 0:
        c_up, c_left, c_diag = cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
        if c_diag <= c_left and c_diag <= c_up:
            i, j = i - 1, j - 1
        elif c_left <= c_up:
            j -= 1
        else:
            i -= 1
        path.append((i, j))
    while i > 0:
        i -= 1
        path.append((i, j))
    while j > 0:
        j -= 1
        path.append((i, j))
    path.reverse()
    return path


def reference_dtw(distances: npt.NDArray) -> float:
    """DTW cost: accumulated cost at the corner divided by the optimal path length."""
    cost = reference_cost(distances)
    path = reference_path(cost)
    return float(cost[-1, -1] / len(path))


def reference_dtw_int(distances: npt.NDArray) -> int:
    """Integer floor-division DTW, matching the integer-dtype kernels.

    The kernels accumulate the cost in the storage dtype and then divide in int64 (floor). When
    the cost does not overflow the storage dtype, the float64 ``reference_cost`` holds the exact
    integer cost, so flooring it and integer-dividing by the path length reproduces the kernels.
    Only valid for inputs small enough that the cost stays within the dtype -- the integer tests
    keep values and dimensions small precisely so this holds.
    """
    cost = reference_cost(distances)
    path = reference_path(cost)
    return int(cost[-1, -1]) // len(path)


def reference_dtw_batch(
    distances: npt.NDArray, sx: npt.NDArray, sy: npt.NDArray, *, symmetric: bool
) -> npt.NDArray[np.float64]:
    """Batched DTW over the (sx[i], sy[j]) sub-block of each ``distances[i, j]`` pair.

    Degenerate pairs (zero-length sequences) and, in the symmetric case, the diagonal and
    lower triangle are left at zero -- matching the kernels' zero-initialised output.
    """
    d = np.asarray(distances, dtype=np.float64)
    n1, n2 = d.shape[0], d.shape[1]
    out = np.zeros((n1, n2), dtype=np.float64)

    def pair(i: int, j: int) -> float:
        n, m = int(sx[i]), int(sy[j])
        if n <= 0 or m <= 0:
            return 0.0
        return reference_dtw(d[i, j, :n, :m])

    if symmetric:
        for j in range(n1):
            for i in range(j):
                value = pair(i, j)
                out[i, j] = value
                out[j, i] = value
    else:
        for i in range(n1):
            for j in range(n2):
                out[i, j] = pair(i, j)
    return out
