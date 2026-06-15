"""CUDA shared-memory capacity edge: the largest valid distances.size(3) per dtype.

The CUDA kernel holds 3 cost diagonals (in the accumulator type) and 3 path-length diagonals
(uint16) of length distances.size(3) in 48 KiB of shared memory (see csrc/cuda/dtw.cu). The
parity/oracle tests cap dimensions at 1280, so the upper half of the valid size(3) range -- and
the exact capacity boundary -- is otherwise untested, even though a sizing bug would hide there.

The per-dtype capacity is MAX_SHARED_BYTES / (3 * (sizeof(acc_t) + sizeof(uint16_t))), with acc_t
being float for half/bfloat16 and the storage type otherwise. These caps are verified arithmetic,
matching the kernel's smem formula.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torchdtw import dtw_batch

from .conftest import HIGH_MINUS_LOW, LOW, assert_equal, make_tensor

# dtype -> largest valid distances.size(3).
CAPACITY = {
    torch.float64: 1638,
    torch.int64: 1638,
    torch.float32: 2730,
    torch.float16: 2730,
    torch.bfloat16: 2730,
    torch.int32: 2730,
    torch.int16: 4096,
    torch.int8: 5461,
    torch.uint8: 5461,
}

# Cost accumulates in float for these, so CPU and CUDA run an identical recurrence and match exactly.
FLOAT_DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@settings(max_examples=20)
@given(n=st.integers(1, 16), low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_at_max_shared_memory_matches_cpu(
    dtype: torch.dtype, n: int, low: float, high_minus_low: float
) -> None:
    """Wavefronts of varying depth across the widest valid size(3) match CPU exactly (fills the 1281+ gap)."""
    m = CAPACITY[dtype]
    d = make_tensor((1, 1, n, m), dtype=dtype, low=low, high=high_minus_low + low)
    sx = torch.tensor([n], dtype=torch.long)
    sy = torch.tensor([m], dtype=torch.long)
    assert_equal(
        dtw_batch(d, sx, sy, symmetric=False),
        dtw_batch(d.cuda(), sx.cuda(), sy.cuda(), symmetric=False).cpu(),
    )


@pytest.mark.requires_gpu
@pytest.mark.parametrize(("dtype", "cap"), list(CAPACITY.items()))
def test_dtw_batch_shared_memory_boundary(dtype: torch.dtype, cap: int) -> None:
    """size(3) == cap fits the shared-memory budget; cap + 1 is rejected."""
    sx = torch.tensor([1], dtype=torch.long, device="cuda")

    d_fits = make_tensor((1, 1, 1, cap), dtype=dtype, low=0, high=1, device="cuda")
    dtw_batch(d_fits, sx, torch.tensor([cap], dtype=torch.long, device="cuda"), symmetric=False)

    d_over = make_tensor((1, 1, 1, cap + 1), dtype=dtype, low=0, high=1, device="cuda")
    with pytest.raises(RuntimeError, match="shared memory"):
        dtw_batch(d_over, sx, torch.tensor([cap + 1], dtype=torch.long, device="cuda"), symmetric=False)
