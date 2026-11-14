"""Independent soft-DTW references, to check the kernels against.

Written with torch, so that autograd through them gives the reference gradient of the backward kernels. Everything
is meant to run in float64.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import torch


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
