"""CUDA placement of the wavefront diagonals around the shared-memory limit, for dtw_batch.

Each CUDA block holds 3 cost diagonals (in the accumulator type) and 3 path-length diagonals (uint16) of length
distances.size(3). They are kept in 48 KiB of shared memory when they fit, and in global memory otherwise (see
plan_diagonals in csrc/cuda/common.cuh). The parity and oracle tests cap dimensions at 1280, so these tests cover the
boundary: both placements must match the CPU exactly.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torchdtw import dtw_batch

from .conftest import HIGH_MINUS_LOW, LOW, assert_equal, make_tensor

SHARED_BYTES = 49152
# Size of the accumulator of each dtype: float for half/bfloat16, the dtype itself otherwise.
ACC_SIZE = {
    torch.float64: 8,
    torch.int64: 8,
    torch.float32: 4,
    torch.float16: 4,
    torch.bfloat16: 4,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
}
# Cost accumulates in float for these, so CPU and CUDA run an identical recurrence and match exactly.
FLOAT_DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]


def diagonal_bytes(m: int, acc_size: int) -> int:
    """Bytes of diagonals per block for distances.size(3) == m, as computed by dtw_batch_cuda_impl."""
    cost_bytes = (3 * m * acc_size + 1) & ~1
    return cost_bytes + 3 * m * 2


def capacity(limit: int, acc_size: int) -> int:
    """Largest distances.size(3) whose diagonals fit in ``limit`` bytes."""
    m = limit // (3 * (acc_size + 2))
    while diagonal_bytes(m + 1, acc_size) <= limit:
        m += 1
    return m


def check_matches_cpu(d: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> None:
    """dtw_batch on CUDA matches the CPU exactly."""
    assert_equal(
        dtw_batch(d.cuda(), sx.cuda(), sy.cuda(), symmetric=symmetric).cpu(),
        dtw_batch(d, sx, sy, symmetric=symmetric),
    )


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@settings(max_examples=20)
@given(n=st.integers(1, 16), low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_at_max_shared_memory_matches_cpu(
    dtype: torch.dtype, n: int, low: float, high_minus_low: float
) -> None:
    """Wavefronts of varying depth across the widest size(3) in shared memory match CPU exactly."""
    m = capacity(SHARED_BYTES, ACC_SIZE[dtype])
    d = make_tensor((1, 1, n, m), dtype=dtype, low=low, high=high_minus_low + low)
    check_matches_cpu(d, torch.tensor([n]), torch.tensor([m]), symmetric=False)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", list(ACC_SIZE))
def test_dtw_batch_shared_memory_boundary(dtype: torch.dtype) -> None:
    """Both sides of the shared-memory limit match CPU: the widest size(3) in shared memory, then in global memory."""
    high = 1 if dtype in FLOAT_DTYPES else 2  # Integer costs stay small enough not to overflow any dtype.
    cap = capacity(SHARED_BYTES, ACC_SIZE[dtype])
    for m in (cap, cap + 1, 2 * cap):
        d = make_tensor((2, 2, 3, m), dtype=dtype, low=0, high=high)
        sy = torch.tensor([m, m // 2])
        check_matches_cpu(d, torch.tensor([3, 2]), sy, symmetric=False)
        check_matches_cpu(d, torch.tensor([3, 2]), torch.tensor([m, m]), symmetric=True)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("symmetric", [False, True])
def test_dtw_batch_global_memory_chunks(symmetric: bool) -> None:  # noqa: FBT001
    """Enough pairs in global memory to need several launches within the scratch budget match CPU.

    With float32 and size(3) == 2731, just beyond shared memory, a block needs 49168 bytes of scratch, so 256 MiB
    fit 5459 blocks.
    """
    m = capacity(SHARED_BYTES, 4) + 1
    g = torch.Generator().manual_seed(0)
    if symmetric:
        # 5995 pairs. Symmetric pairs have sx[y] columns, at most size(2) == 2: small, but each one distinct.
        d = torch.rand((110, 110, 2, m), generator=g)
        sx = sy = torch.randint(1, 3, (110,), generator=g)
    else:
        d = torch.rand((6000, 1, 2, m), generator=g)
        sx, sy = torch.randint(1, 3, (6000,), generator=g), torch.randint(1, m + 1, (1,), generator=g)
    check_matches_cpu(d, sx, sy, symmetric=symmetric)
