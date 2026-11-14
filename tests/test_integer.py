"""Integer-dtype correctness: division semantics, exact int64 results and the narrow path-length guard.

Integer dtypes divide the accumulated cost by the path length in int64, which rounds toward zero: a negative
cost of -7 over a path of 2 gives -3, not the floor -4. The rest of the suite only checks CPU/CUDA parity for
integers, which is blind to a shared logic bug, so these tests anchor the integer result to an independent
reference.

They also pin the documented div-by-zero guard: the path length must not be cast to the (narrow)
storage dtype before dividing -- a length of 256 becomes 0 as int8/uint8, and ``cost / 0`` traps
on integer division by zero (a hard SIGFPE on CPU). See the ``is_integral_v`` branches in
csrc/dtw.cpp and csrc/cuda/dtw.cu.
"""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torchdtw import dtw, dtw_batch, dtw_path

from .conftest import assert_equal, make_tensor
from .reference import reference_dtw_int

INTEGRAL_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
SIGNED_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
# Values 0..3 (or -4..3) over <= 8x8 keep |cost| <= 15 * 4 = 60, within int8/uint8, so the float64
# reference cost is the exact integer cost and its truncated quotient reproduces the kernels.
SMALL = st.integers(1, 8)


@pytest.mark.parametrize("dtype", INTEGRAL_DTYPES)
@given(x=SMALL, y=SMALL)
def test_dtw_integer_division_cpu(dtype: torch.dtype, x: int, y: int) -> None:
    """CPU integer dtw matches the independent integer-division reference."""
    d = make_tensor((x, y), dtype=dtype, low=0, high=4)
    expected = torch.tensor(reference_dtw_int(d.numpy()), dtype=dtype)
    assert_equal(dtw(d), expected)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", INTEGRAL_DTYPES)
@given(x=SMALL, y=SMALL)
def test_dtw_integer_division_cuda(dtype: torch.dtype, x: int, y: int) -> None:
    """CUDA integer dtw matches the independent integer-division reference."""
    d = make_tensor((x, y), dtype=dtype, low=0, high=4)
    expected = torch.tensor(reference_dtw_int(d.numpy()), dtype=dtype)
    assert_equal(dtw(d.cuda()).cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", SIGNED_DTYPES)
def test_dtw_negative_integer_rounds_toward_zero(device: str, dtype: torch.dtype) -> None:
    """A negative cost is rounded toward zero, not floored: -7 over a path of 2 gives -3, not -4."""
    d = torch.tensor([[-3, -4]], dtype=dtype, device=device)
    assert_equal(dtw(d).cpu(), torch.tensor(-3, dtype=dtype))
    assert_equal(dtw_batch(d[None, None]).cpu(), torch.tensor([[-3]], dtype=dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", SIGNED_DTYPES)
@given(x=SMALL, y=SMALL)
def test_dtw_negative_integer_division(device: str, dtype: torch.dtype, x: int, y: int) -> None:
    """Integer dtw with negative distances matches the reference, which rounds toward zero."""
    d = make_tensor((x, y), dtype=dtype, low=-4, high=4)
    expected = torch.tensor(reference_dtw_int(d.numpy()), dtype=dtype)
    assert_equal(dtw(d.to(device)).cpu(), expected)
    assert_equal(dtw_batch(d[None, None].to(device)).cpu(), expected[None, None])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("normalize", [True, False])
def test_dtw_int64_exact_above_2_53(device: str, normalize: bool) -> None:
    """int64 costs above 2^53, where float64 has no exact integers, come out exact from dtw and dtw_batch."""
    value = 2**53 + 1
    d = torch.tensor([[value]], dtype=torch.int64, device=device)
    expected = torch.tensor(value, dtype=torch.int64)
    assert_equal(dtw(d, normalize=normalize).cpu(), expected)
    assert_equal(dtw_batch(d[None, None], normalize=normalize).cpu(), expected[None, None])


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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
@pytest.mark.parametrize("dtype", INTEGRAL_DTYPES)
def test_dtw_integer_accumulates_without_overflow(device: str, dtype: torch.dtype) -> None:
    """The cost accumulates in a wider dtype: a constant max-value matrix has the max value as normalized cost."""
    value = torch.iinfo(dtype).max if dtype != torch.int64 else 2**40
    d = torch.full((2, 3, 16, 16), value, dtype=dtype, device=device)
    assert_equal(dtw(d[0, 0]).cpu(), torch.tensor(value, dtype=dtype))
    assert_equal(dtw_batch(d).cpu(), torch.full((2, 3), value, dtype=dtype))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16])
def test_dtw_narrow_integer_longest_cuda_path(device: str, dtype: torch.dtype) -> None:
    """8- and 16-bit costs accumulate in int32 without overflow up to the longest CUDA path (65535 cells)."""
    value = torch.iinfo(dtype).max
    d = torch.full((1, 65535), value, dtype=dtype, device=device)
    assert_equal(dtw(d).cpu(), torch.tensor(value, dtype=dtype))


@pytest.mark.parametrize("n", [65534, 65535, 70000])
def test_dtw_int16_long_cpu_path(n: int) -> None:
    """CPU paths are not capped: past 65535 cells, int16 costs switch from int32 to int64 accumulation.

    A path of n + 1 cells of -32768 overflows int32 from n = 65535 on, on both sides of the switch.
    """
    value = torch.iinfo(torch.int16).min
    d = torch.full((n, 2), value, dtype=torch.int16)
    assert_equal(dtw(d), torch.tensor(value, dtype=torch.int16))
    assert_equal(dtw_batch(d[None, None]), torch.full((1, 1), value, dtype=torch.int16))
    assert dtw_path(d).shape == (n + 1, 2)
