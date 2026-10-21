"""Independent NumPy DTW reference used as a ground-truth oracle.

This is deliberately a separate, straightforward implementation (a full O(N*M) cost matrix
plus an explicit traceback) so the tests anchor the compiled kernels to a known-correct
result, instead of only comparing the CPU and CUDA backends against each other -- a bug
shared by both backends would slip through a pure parity test. Everything is computed in
float64. The ``min`` tie-break (diag <= left <= up) mirrors csrc/dtw.cpp so the traceback
path matches the kernels cell-for-cell.

The soft-DTW references are written with torch instead, so that autograd through them gives the
reference gradient of the backward kernels.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import torch


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


def reference_in_band(i: int, j: int, n: int, m: int, bandwidth: int) -> bool:
    """Sakoe-Chiba band of csrc/soft_dtw.h, in exact rational arithmetic.

    The band surrounds the diagonal joining (0, 0) to (n - 1, m - 1). Its width is measured along the
    longer axis and floored at half the diagonal's slope.
    """
    if bandwidth < 0 or min(n, m) == 1:
        return True
    short_index, long_index, short, long = (i, j, n, m) if n <= m else (j, i, m, n)
    slope = Fraction(long - 1, short - 1)
    return abs(long_index - short_index * slope) <= max(Fraction(bandwidth), slope / 2)


def reference_soft_dtw(distances: torch.Tensor, gamma: float, bandwidth: int = -1) -> torch.Tensor:
    """Soft-DTW of a 2D ``distances`` tensor by its recursion, cell by cell.

    Cells on the first row or column have a single parent, and cells outside the band are +inf.
    Differentiable with respect to ``distances``.
    """
    n, m = distances.shape
    inf = distances.new_tensor(float("inf"))
    cost: list[list[torch.Tensor]] = [[inf] * m for _ in range(n)]
    for i, j in itertools.product(range(n), range(m)):
        if not reference_in_band(i, j, n, m, bandwidth):
            continue
        parents = [cost[a][b] for a, b in ((i - 1, j - 1), (i, j - 1), (i - 1, j)) if a >= 0 and b >= 0]
        if not parents:
            cost[i][j] = distances[i, j]
        elif len(parents) == 1:
            cost[i][j] = distances[i, j] + parents[0]
        else:
            cost[i][j] = distances[i, j] - gamma * torch.logsumexp(-torch.stack(parents) / gamma, 0)
    return cost[n - 1][m - 1]


def _alignment_paths(n: int, m: int, i: int = 0, j: int = 0) -> list[list[tuple[int, int]]]:
    """Every monotonic alignment path from (i, j) to (n - 1, m - 1)."""
    if (i, j) == (n - 1, m - 1):
        return [[(i, j)]]
    steps = [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
    return [[(i, j), *tail] for a, b in steps if a < n and b < m for tail in _alignment_paths(n, m, a, b)]


def brute_force_soft_dtw(distances: torch.Tensor, gamma: float) -> float:
    """Soft-DTW by its definition: the soft minimum of the costs of every alignment path."""
    n, m = distances.shape
    costs = torch.stack([sum(distances[i, j] for i, j in path) for path in _alignment_paths(n, m)])
    return float(-gamma * torch.logsumexp(-costs / gamma, 0))
