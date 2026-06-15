"""Integer-dtype correctness: floor-division semantics and the narrow path-length guard.

Integer dtypes divide the accumulated cost by the path length in int64 and floor the result.
The rest of the suite only checks CPU/CUDA parity for integers, which is blind to a shared
logic bug, so these tests anchor the integer result to an independent floor-division reference.

They also pin the documented div-by-zero guard: the path length must not be cast to the (narrow)
storage dtype before dividing -- a length of 256 becomes 0 as int8/uint8, and ``cost / 0`` traps
on integer division by zero (a hard SIGFPE on CPU). See the ``is_integral_v`` branches in
csrc/dtw.cpp and csrc/cuda/dtw.cu.
"""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torchdtw import dtw

from .conftest import assert_equal, make_tensor
from .reference import reference_dtw_int

INTEGRAL_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
# Values 0..3 over <= 8x8 keep the cost <= 15 * 3 = 45, within int8/uint8, so the float64
# reference cost is the exact integer cost and floor division reproduces the kernels.
SMALL = st.integers(1, 8)


@pytest.mark.parametrize("dtype", INTEGRAL_DTYPES)
@given(x=SMALL, y=SMALL)
def test_dtw_integer_floor_division_cpu(dtype: torch.dtype, x: int, y: int) -> None:
    """CPU integer dtw matches the independent floor-division reference."""
    d = make_tensor((x, y), dtype=dtype, low=0, high=4)
    expected = torch.tensor(reference_dtw_int(d.numpy()), dtype=dtype)
    assert_equal(dtw(d), expected)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", INTEGRAL_DTYPES)
@given(x=SMALL, y=SMALL)
def test_dtw_integer_floor_division_cuda(dtype: torch.dtype, x: int, y: int) -> None:
    """CUDA integer dtw matches the independent floor-division reference."""
    d = make_tensor((x, y), dtype=dtype, low=0, high=4)
    expected = torch.tensor(reference_dtw_int(d.numpy()), dtype=dtype)
    assert_equal(dtw(d.cuda()).cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@given(k=st.integers(1, 4), value=st.integers(1, 6))
def test_dtw_narrow_path_length_div_guard_cpu(dtype: torch.dtype, k: int, value: int) -> None:
    """A path length that is a multiple of 256 must divide in int64, not as the storage dtype.

    256*k casts to 0 as int8/uint8 (a divide-by-zero SIGFPE); the column forces a straight-down
    path of length 256*k.
    """
    d = torch.zeros((256 * k, 1), dtype=dtype)
    d[0, 0] = value  # cost == value (no overflow); value // (256*k) == 0
    assert_equal(dtw(d), torch.tensor(0, dtype=dtype))


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@given(k=st.integers(1, 4), value=st.integers(1, 6))
def test_dtw_narrow_path_length_div_guard_cuda(dtype: torch.dtype, k: int, value: int) -> None:
    """Same guard on CUDA: casting a length that is a multiple of 256 to int8/uint8 divides by zero."""
    d = torch.zeros((256 * k, 1), dtype=dtype, device="cuda")
    d[0, 0] = value
    assert_equal(dtw(d).cpu(), torch.tensor(0, dtype=dtype))
