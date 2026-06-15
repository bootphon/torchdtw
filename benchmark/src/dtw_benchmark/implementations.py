"""Additional implementations of DTW."""

import numba
import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    pass

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


@numba.jit(nopython=True)
def _dtw_core(x: npt.NDArray[np.float32]) -> float:
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.int32)
    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]  # diag
            c1 = cost[i - 1, j]  # up
            c2 = cost[i, j - 1]  # left
            # Tie-break diag <= left <= up, matching csrc/dtw.cpp.
            if c0 <= c1 and c0 <= c2:
                c, t = c0, 0
            elif c2 <= c1:
                c, t = c2, 2
            else:
                c, t = c1, 1
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t
    return cost[-1, -1] / _backtrace(trace)


def dtw_numba(distances: torch.Tensor) -> torch.Tensor:
    """Numba implementation from Whisper: https://github.com/openai/whisper/blob/main/whisper/timing.py."""
    return torch.tensor(_dtw_core(distances.cpu().numpy()), device=distances.device)


@numba.jit(nopython=True, parallel=True)
def _dtw_numba_batch(
    distances: npt.NDArray[np.float32],
    sx: npt.NDArray[np.int64],
    sy: npt.NDArray[np.int64],
    symmetric: bool,  # noqa: FBT001
) -> npt.NDArray[np.float32]:
    nx = distances.shape[0]
    ny = distances.shape[1]
    out = np.zeros((nx, ny), dtype=np.float32)
    # Flatten the (i, j) pairs to compute into task arrays so the work can run over a single prange.
    num_tasks = 0
    for i in range(nx):
        num_tasks += ny - (i + 1 if symmetric else 0)
    task_i = np.empty(num_tasks, dtype=np.int64)
    task_j = np.empty(num_tasks, dtype=np.int64)
    k = 0
    for i in range(nx):
        for j in range(i + 1 if symmetric else 0, ny):
            task_i[k] = i
            task_j[k] = j
            k += 1
    for k in numba.prange(num_tasks):
        i = task_i[k]
        j = task_j[k]
        n = sx[i]
        m = sy[j]
        if n > 0 and m > 0:  # degenerate (zero-length) pairs are left at zero
            value = _dtw_core(distances[i, j, :n, :m])
            out[i, j] = value
            if symmetric:
                out[j, i] = value
    return out


def dtw_numba_batch(
    distances: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    *,
    symmetric: bool,
) -> torch.Tensor:
    """Batched Numba DTW, parallel over the (sx[i], sy[j]) pairs."""
    out = _dtw_numba_batch(
        distances.cpu().numpy(),
        sx.cpu().numpy().astype(np.int64),
        sy.cpu().numpy().astype(np.int64),
        symmetric,
    )
    return torch.from_numpy(out).to(distances.device)


@numba.jit(nopython=True, parallel=True)
def _batch_backtrace(
    trace: npt.NDArray[np.int32],
    corner: npt.NDArray[np.float32],
    task_i: npt.NDArray[np.int64],
    task_j: npt.NDArray[np.int64],
    task_n: npt.NDArray[np.int64],
    task_m: npt.NDArray[np.int64],
    n1: int,
    n2: int,
    symmetric: bool,  # noqa: FBT001
) -> npt.NDArray[np.float32]:
    """Normalize per-pair corner costs by the backtrace path length, in parallel over pairs.

    ``trace`` holds the 1-based (i, j) predecessor of every pair, padded to (P, s1+1, s2+1).
    """
    out = np.zeros((n1, n2), dtype=np.float32)
    for p in numba.prange(task_i.shape[0]):
        n = task_n[p]
        m = task_m[p]
        path_len = _backtrace(np.ascontiguousarray(trace[p, : n + 1, : m + 1]))
        value = corner[p] / path_len
        out[task_i[p], task_j[p]] = value
        if symmetric:
            out[task_j[p], task_i[p]] = value
    return out


try:

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

            # Stored in order up, left, diag so the last write wins: tie-break diag <= left <= up,
            # matching csrc/dtw.cpp and the other implementations.
            trace_ptr = trace + (k + 1) * trace_stride + 1
            tl.store(trace_ptr + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
            tl.store(trace_ptr + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
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

    @triton.jit
    def _dtw_triton_batch_kernel(
        distances: torch.Tensor,
        cost: torch.Tensor,
        trace: torch.Tensor,
        corner: torch.Tensor,
        task_off: torch.Tensor,
        task_n: torch.Tensor,
        task_m: torch.Tensor,
        dist_row_stride: int,
        cost_pair_stride: int,
        cost_stride: int,
        trace_pair_stride: int,
        trace_row_stride: int,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        # One program per pair; lanes run along the rows (i) of a single anti-diagonal at a time.
        p = tl.program_id(0)
        M = tl.load(task_n + p)  # rows = sx[i]
        N = tl.load(task_m + p)  # cols = sy[j]
        dist_p = distances + tl.load(task_off + p)
        cost_p = cost + p * cost_pair_stride
        trace_p = trace + p * trace_pair_stride

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        for k in range(1, N + M + 1):  # k = i + j + 1
            tl.debug_barrier()
            c0 = tl.load(cost_p + (k - 1) * cost_stride + offsets, mask=mask)  # diag
            c1 = tl.load(cost_p + k * cost_stride + offsets, mask=mask)  # up
            c2 = tl.load(cost_p + k * cost_stride + 1 + offsets, mask=mask)  # left

            cell_i = offsets
            cell_j = (k - 1) - offsets
            valid = mask & (cell_j >= 0) & (cell_j < N)
            x_row = tl.load(dist_p + cell_i * dist_row_stride + cell_j, mask=valid, other=float("inf"))
            cost_row = x_row + tl.minimum(tl.minimum(c0, c1), c2)
            tl.store(cost_p + (k + 1) * cost_stride + 1 + offsets, cost_row, mask=mask)

            # Natural-layout trace, tie-break diag <= left <= up. Each lane owns a distinct cell.
            # Stored into a (1-based) padded grid so _backtrace's virtual (0, *)/(*, 0) boundary lines up.
            t = tl.where((c0 <= c1) & (c0 <= c2), 0, tl.where(c2 <= c1, 2, 1))
            tl.store(trace_p + (cell_i + 1) * trace_row_stride + (cell_j + 1), t, mask=valid)
            tl.store(corner + p + offsets * 0, cost_row, mask=valid & (cell_i == M - 1) & (cell_j == N - 1))

    def dtw_triton_batch(
        distances: torch.Tensor,
        sx: torch.Tensor,
        sy: torch.Tensor,
        *,
        symmetric: bool,
    ) -> torch.Tensor:
        """Batched Triton DTW: a single kernel launch with one program per (sx[i], sy[j]) pair.

        Each program runs the skewed anti-diagonal forward DP of ``dtw_triton`` into its own slice of
        the batched cost/trace buffers; the path-length normalization is then done in one parallel
        Numba pass over the natural-layout traces.
        """
        n1, n2, s1, s2 = distances.shape
        device = distances.device

        # Enumerate the (non-degenerate) pairs to compute.
        if symmetric:
            i_idx, j_idx = torch.triu_indices(n1, n2, offset=1)
        else:
            grid_i, grid_j = torch.meshgrid(torch.arange(n1), torch.arange(n2), indexing="ij")
            i_idx, j_idx = grid_i.reshape(-1), grid_j.reshape(-1)
        task_n = sx.cpu()[i_idx]
        task_m = sy.cpu()[j_idx]
        keep = (task_n > 0) & (task_m > 0)
        i_idx, j_idx, task_n, task_m = i_idx[keep], j_idx[keep], task_n[keep], task_m[keep]
        if i_idx.numel() == 0:
            return torch.zeros((n1, n2), device=device, dtype=distances.dtype)

        num_pairs = i_idx.numel()
        max_n, max_m = int(task_n.max()), int(task_m.max())
        block_size = triton.next_power_of_2(max_n)
        rows, cols = max_n + max_m + 2, max_n + 2

        task_off = ((i_idx * n2 + j_idx) * s1 * s2).to(device=device, dtype=torch.int64)
        task_n_d = task_n.to(device=device, dtype=torch.int32)
        task_m_d = task_m.to(device=device, dtype=torch.int32)
        cost = torch.full((num_pairs, rows, cols), torch.inf, device=device)
        cost[:, 0, 0] = 0
        # Padded (s1+1, s2+1) so each pair's trace matches _backtrace's 1-based layout.
        trace = torch.zeros((num_pairs, s1 + 1, s2 + 1), device=device, dtype=torch.int32)
        corner = torch.full((num_pairs,), torch.inf, device=device)

        _dtw_triton_batch_kernel[(num_pairs,)](
            distances.contiguous(),
            cost,
            trace,
            corner,
            task_off,
            task_n_d,
            task_m_d,
            distances.stride(2),
            cost.stride(0),
            cost.stride(1),
            trace.stride(0),
            trace.stride(1),
            BLOCK_SIZE=block_size,  # ty: ignore[invalid-argument-type]
        )

        out = _batch_backtrace(
            trace.cpu().numpy(),
            corner.cpu().numpy(),
            i_idx.numpy().astype(np.int64),
            j_idx.numpy().astype(np.int64),
            task_n.numpy().astype(np.int64),
            task_m.numpy().astype(np.int64),
            n1,
            n2,
            symmetric,
        )
        return torch.from_numpy(out).to(device)

except NameError:

    def dtw_triton(_: torch.Tensor) -> torch.Tensor:
        """Act as a placeholder when triton is not found."""
        raise NameError

    def dtw_triton_batch(*_args: object, **_kwargs: object) -> torch.Tensor:
        """Act as a placeholder when triton is not found."""
        raise NameError
