"""Additional implementations of DTW."""

import numba
import numpy as np
import numpy.typing as npt
import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from .dtw_cython import _dtw_cython, _dtw_cython_batch


def dtw_torch(distances: torch.Tensor) -> torch.Tensor:
    """Naive DTW implementation."""
    N, M = distances.shape
    cost = torch.zeros_like(distances)
    cost[:, 0] = torch.cumsum(distances[:, 0], 0)
    cost[0, :] = torch.cumsum(distances[0, :], 0)
    for i in range(1, N):
        for j in range(1, M):
            cost[i, j] = distances[i, j] + min(cost[i - 1, j], cost[i - 1, j - 1], cost[i, j - 1])
    path_len, i, j = 1, N - 1, M - 1
    while i > 0 and j > 0:
        c_up, c_left, c_diag = cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
        if c_diag <= c_left and c_diag <= c_up:
            i -= 1
            j -= 1
        elif c_left <= c_up:
            j -= 1
        else:
            i -= 1
        path_len += 1
    if i == 0:
        path_len += j
    if j == 0:
        path_len += i
    return cost[N - 1, M - 1] / path_len


def dtw_cython(distances: torch.Tensor) -> torch.Tensor:
    """Cython DTW."""
    return torch.tensor(_dtw_cython(distances.cpu().numpy()), device=distances.device)


def dtw_cython_batch(
    distances: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    *,
    symmetric: bool,
) -> torch.Tensor:
    """Batched Cython DTW."""
    return torch.from_numpy(
        _dtw_cython_batch(
            distances.cpu().numpy(),
            sx.cpu().numpy(),
            sy.cpu().numpy(),
            symmetric,
        ),
    ).to(distances.device)


@numba.jit(nopython=True)
def _backtrace(trace: npt.NDArray[np.float32]) -> float:
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    path_len = 0
    trace[0, :] = 2
    trace[:, 0] = 1
    while i > 0 and j > 0:
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError(trace[i, j])
        path_len += 1
    if i == 0:
        path_len += j
    if j == 0:
        path_len += i
    return path_len


@numba.jit(nopython=True, parallel=True)
def _dtw_numba(x: npt.NDArray[np.float32]) -> float:
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.int32)
    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t
    return cost[-1, -1] / _backtrace(trace)


def dtw_numba(distances: torch.Tensor) -> torch.Tensor:
    """Numba implementation from Whisper: https://github.com/openai/whisper/blob/main/whisper/timing.py."""
    return torch.tensor(_dtw_numba(distances.cpu().numpy()), device=distances.device)


@triton.jit
def _dtw_triton_kernel(
    cost: torch.Tensor,
    trace: torch.Tensor,
    x: torch.Tensor,
    x_stride: int,
    cost_stride: int,
    trace_stride: int,
    N: int,
    M: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    for k in range(1, N + M + 1):  # k = i + j
        tl.debug_barrier()

        p0 = cost + (k - 1) * cost_stride
        p1 = cost + k * cost_stride
        p2 = cost + k * cost_stride + 1

        c0 = tl.load(p0 + offsets, mask=mask)
        c1 = tl.load(p1 + offsets, mask=mask)
        c2 = tl.load(p2 + offsets, mask=mask)

        x_row = tl.load(x + (k - 1) * x_stride + offsets, mask=mask, other=0)
        cost_row = x_row + tl.minimum(tl.minimum(c0, c1), c2)

        cost_ptr = cost + (k + 1) * cost_stride + 1
        tl.store(cost_ptr + offsets, cost_row, mask=mask)

        trace_ptr = trace + (k + 1) * trace_stride + 1
        tl.store(trace_ptr + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
        tl.store(trace_ptr + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
        tl.store(trace_ptr + offsets, 0, mask=mask & (c0 <= c1) & (c0 <= c2))


def dtw_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton implementation from Whisper: https://github.com/openai/whisper/blob/main/whisper/triton_ops.py."""
    BLOCK_SIZE = 1024
    M, N = x.shape
    assert M < BLOCK_SIZE, f"M should be smaller than {BLOCK_SIZE=}"  # noqa: S101
    x_skew = F.pad(x, (0, M + 1), value=torch.inf).flatten()[: M * (N + M)].reshape(M, N + M)
    x_skew = x_skew.T.contiguous()
    cost = torch.ones(N + M + 2, M + 2) * torch.inf
    cost[0, 0] = 0
    cost = cost.to(x.device)
    trace = torch.zeros_like(cost, dtype=torch.int32)
    _dtw_triton_kernel[(1,)](
        cost,
        trace,
        x_skew,
        x_skew.stride(0),
        cost.stride(0),
        trace.stride(0),
        N,
        M,
        BLOCK_SIZE=BLOCK_SIZE,  # ty: ignore[invalid-argument-type]
    )
    trace = trace.T.flatten()[: (M + 1) * (M + N + 3)].reshape(M + 1, M + N + 3)[:, : N + 1]
    flat_index = M * (M + N + 3) + N
    row = flat_index % (N + M + 2)
    col = flat_index // (N + M + 2)
    return cost[row, col] / _backtrace(trace.cpu().numpy())
