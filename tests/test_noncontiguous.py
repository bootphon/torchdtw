"""Strided / non-contiguous inputs.

The accessors read strides(), so the kernels must handle inputs whose memory layout is not
contiguous. torch.testing.make_tensor(noncontiguous=True) builds such tensors directly --
including on CUDA, where .cuda() would otherwise re-pack a strided tensor to contiguous. The
rest of the suite only feeds contiguous data, so an accessor that silently assumed contiguity
would pass everything else. noncontiguous=True only produces a strided tensor when a dimension
has more than one element, hence the >= 2 lower bounds below.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torchdtw import dtw, dtw_batch, dtw_path

from .conftest import HIGH_MINUS_LOW, LOW, assert_equal, make_tensor
from .reference import reference_cost, reference_dtw, reference_dtw_batch, reference_path

NC_DIM = st.integers(2, 64)
NC_BATCH = st.integers(2, 3)
NC_BATCH_DIM = st.integers(2, 24)


@given(x=NC_DIM, y=NC_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_noncontiguous_cpu(x: int, y: int, low: float, high_minus_low: float) -> None:
    """CPU dtw on a non-contiguous tensor matches its contiguous copy and the reference."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low, noncontiguous=True)
    assert not d.is_contiguous()
    expected = torch.tensor(reference_dtw(d.numpy()), dtype=torch.float64)
    assert_equal(dtw(d), dtw(d.contiguous()))
    assert_equal(dtw(d), expected)


@pytest.mark.requires_gpu
@given(x=NC_DIM, y=NC_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_noncontiguous_cuda(x: int, y: int, low: float, high_minus_low: float) -> None:
    """CUDA dtw on a non-contiguous device tensor matches its contiguous copy and the reference."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device="cuda", noncontiguous=True)
    assert not d.is_contiguous()
    expected = torch.tensor(reference_dtw(d.cpu().numpy()), dtype=torch.float64)
    assert_equal(dtw(d).cpu(), dtw(d.contiguous()).cpu())
    assert_equal(dtw(d).cpu(), expected)


@given(x=NC_DIM, y=NC_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_path_noncontiguous(x: int, y: int, low: float, high_minus_low: float) -> None:
    """dtw_path on a non-contiguous tensor matches the reference traceback."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low, noncontiguous=True)
    assert not d.is_contiguous()
    expected = torch.tensor(reference_path(reference_cost(d.numpy())), dtype=torch.long)
    assert_equal(dtw_path(d), expected)


@settings(max_examples=50)
@given(n=NC_BATCH, m=NC_BATCH, x=NC_BATCH_DIM, y=NC_BATCH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_noncontiguous_cpu(n: int, m: int, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Asymmetric dtw_batch with non-contiguous distances and non-contiguous sx/sy (CPU)."""
    d = make_tensor((n, m, x, y), dtype=torch.float64, low=low, high=high_minus_low + low, noncontiguous=True)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1, noncontiguous=True)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y + 1, noncontiguous=True)
    assert not d.is_contiguous()
    assert not sx.is_contiguous()
    expected = torch.from_numpy(reference_dtw_batch(d.numpy(), sx.numpy(), sy.numpy(), symmetric=False))
    assert_equal(dtw_batch(d, sx, sy, symmetric=False), expected)


@settings(max_examples=50)
@given(n=NC_BATCH, x=NC_BATCH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_symmetric_noncontiguous_cpu(n: int, x: int, low: float, high_minus_low: float) -> None:
    """Symmetric dtw_batch (triangular index path) with non-contiguous distances and sx (CPU)."""
    d = make_tensor((n, n, x, x), dtype=torch.float64, low=low, high=high_minus_low + low, noncontiguous=True)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1, noncontiguous=True)
    assert not d.is_contiguous()
    expected = torch.from_numpy(reference_dtw_batch(d.numpy(), sx.numpy(), sx.numpy(), symmetric=True))
    assert_equal(dtw_batch(d, sx, sx, symmetric=True), expected)


@pytest.mark.requires_gpu
@settings(max_examples=50)
@given(n=NC_BATCH, m=NC_BATCH, x=NC_BATCH_DIM, y=NC_BATCH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_noncontiguous_cuda(n: int, m: int, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Asymmetric dtw_batch with non-contiguous CUDA distances and sx/sy matches the reference."""
    d = make_tensor(
        (n, m, x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device="cuda", noncontiguous=True
    )
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1, device="cuda", noncontiguous=True)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y + 1, device="cuda", noncontiguous=True)
    assert not d.is_contiguous()
    assert not sx.is_contiguous()
    expected = torch.from_numpy(
        reference_dtw_batch(d.cpu().numpy(), sx.cpu().numpy(), sy.cpu().numpy(), symmetric=False)
    )
    assert_equal(dtw_batch(d, sx, sy, symmetric=False).cpu(), expected)
