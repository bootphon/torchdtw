"""Compare CPU and CUDA dtw implementations."""

import pytest
import torch
from hypothesis import given

from torchdtw import dtw, dtw_batch

from .conftest import BATCH, DIM, HIGH_MINUS_LOW, LOW, assert_equal, make_tensor


@pytest.mark.requires_gpu
@given(x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw(x: int, y: int, low: float, high_minus_low: float) -> None:
    """Compare the output of dtw between CPU and GPU implementations."""
    d = make_tensor((x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    assert_equal(dtw(d), dtw(d.cuda()).cpu())


@pytest.mark.requires_gpu
@given(n=BATCH, x=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_symmetric(n: int, x: int, low: float, high_minus_low: float) -> None:
    """Compare the output of dtw_batch between CPU and GPU implementations, symmetric case."""
    d = make_tensor((n, n, x, x), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1)
    i, j = torch.triu_indices(n, n)
    d[i, j] = d[j, i]
    assert_equal(dtw_batch(d, sx, sx, symmetric=True), dtw_batch(d.cuda(), sx.cuda(), sx.cuda(), symmetric=True).cpu())


@pytest.mark.requires_gpu
@given(n=BATCH, m=BATCH, x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_not_symmetric(n: int, m: int, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Compare the output of dtw_batch between CPU and GPU implementations, non symmetric case."""
    d = make_tensor((n, m, x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y + 1)
    assert_equal(
        dtw_batch(d, sx, sy, symmetric=False),
        dtw_batch(d.cuda(), sx.cuda(), sy.cuda(), symmetric=False).cpu(),
    )
