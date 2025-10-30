import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def dtw_kernel(
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


def dtw_cuda(x, BLOCK_SIZE: int = 1024):
    M, N = x.shape
    assert M < BLOCK_SIZE, f"M should be smaller than {BLOCK_SIZE=}"
    x_skew = F.pad(x, (0, M + 1), value=torch.inf).flatten()[: M * (N + M)].reshape(M, N + M)
    x_skew = x_skew.T.contiguous()
    cost = torch.ones(N + M + 2, M + 2) * torch.inf
    cost[0, 0] = 0
    cost = cost.to(x.device)
    trace = torch.zeros_like(cost, dtype=torch.int32)
    dtw_kernel[(1,)](
        cost,
        trace,
        x_skew,
        x_skew.stride(0),
        cost.stride(0),
        trace.stride(0),
        N,
        M,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    trace = trace.T.flatten()[: (M + 1) * (M + N + 3)].reshape(M + 1, M + N + 3)[:, : N + 1]
    return backtrace(trace.cpu().numpy())
