"""Tests for floating-point and integer dtype dispatch on CPU and CUDA backends."""

import pytest
import torch
from hypothesis import given

from torchdtw import dtw, dtw_batch

from .conftest import BATCH, DIM, assert_equal, make_tensor

FLOATING_DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]
INTEGRAL_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
DISTANCES_DTYPES = FLOATING_DTYPES + INTEGRAL_DTYPES
SX_DTYPES = [*INTEGRAL_DTYPES, torch.uint16, torch.uint32, torch.uint64]


@pytest.mark.parametrize("dtype", DISTANCES_DTYPES)
@given(x=DIM, y=DIM)
def test_dtw_dispatch_cpu(dtype: torch.dtype, x: int, y: int) -> None:
    """Verify that dtw runs on CPU for every supported distances dtype."""
    d = make_tensor((x, y), dtype=dtype, low=0, high=4)
    out = dtw(d)
    assert out.dtype == dtype
    assert out.shape == ()


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", DISTANCES_DTYPES)
@given(x=DIM, y=DIM)
def test_dtw_dispatch_cuda(dtype: torch.dtype, x: int, y: int) -> None:
    """Compare CPU and CUDA dtw outputs for every supported distances dtype."""
    d = make_tensor((x, y), dtype=dtype, low=0, high=4)
    assert_equal(dtw(d), dtw(d.cuda()).cpu())


@pytest.mark.parametrize("dtype", DISTANCES_DTYPES)
@given(n=BATCH, m=BATCH, x=DIM, y=DIM)
def test_dtw_batch_distances_dispatch_cpu(dtype: torch.dtype, n: int, m: int, x: int, y: int) -> None:
    """Verify that dtw_batch runs on CPU for every supported distances dtype."""
    d = make_tensor((n, m, x, y), dtype=dtype, low=0, high=4)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y + 1)
    out = dtw_batch(d, sx, sy, symmetric=False)
    assert out.dtype == dtype
    assert out.shape == (n, m)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", DISTANCES_DTYPES)
@given(n=BATCH, m=BATCH, x=DIM, y=DIM)
def test_dtw_batch_distances_dispatch_cuda(dtype: torch.dtype, n: int, m: int, x: int, y: int) -> None:
    """Compare CPU and CUDA dtw_batch outputs for every supported distances dtype."""
    d = make_tensor((n, m, x, y), dtype=dtype, low=0, high=4)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y + 1)
    assert_equal(
        dtw_batch(d, sx, sy, symmetric=False),
        dtw_batch(d.cuda(), sx.cuda(), sy.cuda(), symmetric=False).cpu(),
    )


@pytest.mark.parametrize("sx_dtype", SX_DTYPES)
@given(n=BATCH, m=BATCH, x=DIM, y=DIM)
def test_dtw_batch_sx_dispatch_cpu(sx_dtype: torch.dtype, n: int, m: int, x: int, y: int) -> None:
    """Verify that dtw_batch runs on CPU for every supported sx/sy integer dtype."""
    d = make_tensor((n, m, x, y), dtype=torch.float32, low=0.0, high=1.0)
    sx = make_tensor((n,), dtype=sx_dtype, low=1, high=x + 1)
    sy = make_tensor((m,), dtype=sx_dtype, low=1, high=y + 1)
    out = dtw_batch(d, sx, sy, symmetric=False)
    assert out.dtype == torch.float32
    assert out.shape == (n, m)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("sx_dtype", SX_DTYPES)
@given(n=BATCH, m=BATCH, x=DIM, y=DIM)
def test_dtw_batch_sx_dispatch_cuda(sx_dtype: torch.dtype, n: int, m: int, x: int, y: int) -> None:
    """Compare CPU and CUDA dtw_batch outputs for every supported sx/sy integer dtype."""
    d = make_tensor((n, m, x, y), dtype=torch.float32, low=0.0, high=1.0)
    sx = make_tensor((n,), dtype=sx_dtype, low=1, high=x + 1)
    sy = make_tensor((m,), dtype=sx_dtype, low=1, high=y + 1)
    assert_equal(
        dtw_batch(d, sx, sy, symmetric=False),
        dtw_batch(d.cuda(), sx.cuda(), sy.cuda(), symmetric=False).cpu(),
    )
