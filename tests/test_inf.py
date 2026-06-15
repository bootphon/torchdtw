"""Infinite (masking).

A distance of +inf marks a forbidden alignment: the cost recurrence only ever adds and takes
minima (never subtracts), and the result is well-defined -- a path that must cross an inf cell
yields inf, and a finite "channel" through the matrix is followed around the masked cells.
This is the standard way to mask DTW, so it is pinned here:
inf results are deterministic, identical on CPU and CUDA, and match the float64 reference.
"""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torchdtw import dtw, dtw_path

from .conftest import assert_equal, make_tensor
from .reference import reference_cost, reference_dtw, reference_path

INF = float("inf")
INF_DIM = st.integers(1, 48)
INF_FLOAT_DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]


def test_dtw_inf_forced_corner_is_inf() -> None:
    """A path must include both corners, so an inf corner forces an inf result."""
    d = torch.tensor([[INF, 1.0], [1.0, 1.0]], dtype=torch.float64)
    assert torch.isinf(dtw(d))


def test_dtw_inf_finite_channel_is_followed() -> None:
    """A finite diagonal channel surrounded by inf gives the finite cost of that channel."""
    d = torch.tensor([[1.0, INF, INF], [INF, 1.0, INF], [INF, INF, 1.0]], dtype=torch.float64)
    assert_equal(dtw(d), torch.tensor(1.0, dtype=torch.float64))
    assert_equal(dtw_path(d), torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long))


def test_dtw_inf_off_path_is_ignored() -> None:
    """An inf off the optimal path does not affect the result."""
    d = torch.tensor([[1.0, 1.0, INF], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    assert_equal(dtw(d), torch.tensor(1.0, dtype=torch.float64))


@given(x=INF_DIM, y=INF_DIM, p=st.floats(0.0, 0.5), seed=st.integers(0, 2**31 - 1))
def test_dtw_inf_mask_matches_reference(x: int, y: int, p: float, seed: int) -> None:
    """A randomly inf-masked matrix matches the float64 reference (cost only adds/mins, never NaN)."""
    d = make_tensor((x, y), dtype=torch.float64, low=0.1, high=10.0)
    g = torch.Generator().manual_seed(seed)
    d = d.masked_fill(torch.rand((x, y), generator=g) < p, INF)
    expected = torch.tensor(reference_dtw(d.numpy()), dtype=torch.float64)
    assert_equal(dtw(d), expected)


def test_dtw_path_inf_mask_matches_reference() -> None:
    """dtw_path routes around inf exactly as the reference traceback does."""
    d = torch.tensor([[1.0, 1.0, INF], [INF, 1.0, INF], [INF, 1.0, 1.0]], dtype=torch.float64)
    expected = torch.tensor(reference_path(reference_cost(d.numpy())), dtype=torch.long)
    assert_equal(dtw_path(d), expected)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", INF_FLOAT_DTYPES)
@given(x=INF_DIM, y=INF_DIM, p=st.floats(0.0, 0.5), seed=st.integers(0, 2**31 - 1))
def test_dtw_inf_mask_cpu_cuda_parity(dtype: torch.dtype, x: int, y: int, p: float, seed: int) -> None:
    """Inf masking is identical on CPU and CUDA across float dtypes."""
    d = make_tensor((x, y), dtype=dtype, low=0.1, high=10.0)
    g = torch.Generator().manual_seed(seed)
    d = d.masked_fill(torch.rand((x, y), generator=g) < p, INF)
    assert_equal(dtw(d), dtw(d.cuda()).cpu())
