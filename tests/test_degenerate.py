"""Zero-length sequences exercise the N<=0 / M<=0 degenerate branch in both kernels.

The rest of the suite samples sequence lengths from low=1, so the explicit zero-write branch
(csrc/dtw.cpp and csrc/cuda/dtw.cu) is never hit. dtw_batch must return 0 for any pair whose
sx[i] or sy[j] is zero, and CPU and CUDA must agree.
"""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torchdtw import dtw_batch

from .conftest import assert_equal, make_tensor

DEG_DIM = st.integers(1, 32)
DEG_BATCH = st.integers(1, 4)


@given(n=DEG_BATCH, m=DEG_BATCH, x=DEG_DIM, y=DEG_DIM)
def test_dtw_batch_zero_lengths_cpu(n: int, m: int, x: int, y: int) -> None:
    """Pairs with a zero-length sequence are zero; others are unaffected (CPU)."""
    d = make_tensor((n, m, x, y), dtype=torch.float32, low=0.1, high=1.0)
    sx = make_tensor((n,), dtype=torch.long, low=0, high=x + 1)
    sy = make_tensor((m,), dtype=torch.long, low=0, high=y + 1)
    out = dtw_batch(d, sx, sy, symmetric=False)
    degenerate = (sx.view(n, 1) == 0) | (sy.view(1, m) == 0)
    assert torch.equal(out[degenerate], torch.zeros_like(out[degenerate]))


@given(n=DEG_BATCH, m=DEG_BATCH, x=DEG_DIM, y=DEG_DIM)
def test_dtw_batch_all_zero_lengths_cpu(n: int, m: int, x: int, y: int) -> None:
    """A batch where every length is zero returns an all-zero matrix."""
    d = make_tensor((n, m, x, y), dtype=torch.float32, low=0.1, high=1.0)
    sx = torch.zeros((n,), dtype=torch.long)
    sy = torch.zeros((m,), dtype=torch.long)
    out = dtw_batch(d, sx, sy, symmetric=False)
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.requires_gpu
@given(n=DEG_BATCH, m=DEG_BATCH, x=DEG_DIM, y=DEG_DIM)
def test_dtw_batch_zero_lengths_cuda_parity(n: int, m: int, x: int, y: int) -> None:
    """CPU and CUDA agree on batches containing zero-length sequences (asymmetric)."""
    d = make_tensor((n, m, x, y), dtype=torch.float32, low=0.1, high=1.0)
    sx = make_tensor((n,), dtype=torch.long, low=0, high=x + 1)
    sy = make_tensor((m,), dtype=torch.long, low=0, high=y + 1)
    assert_equal(
        dtw_batch(d, sx, sy, symmetric=False),
        dtw_batch(d.cuda(), sx.cuda(), sy.cuda(), symmetric=False).cpu(),
    )


@pytest.mark.requires_gpu
@given(n=DEG_BATCH, x=DEG_DIM)
def test_dtw_batch_zero_lengths_cuda_parity_symmetric(n: int, x: int) -> None:
    """CPU and CUDA agree on symmetric batches containing zero-length sequences."""
    d = make_tensor((n, n, x, x), dtype=torch.float32, low=0.1, high=1.0)
    sx = make_tensor((n,), dtype=torch.long, low=0, high=x + 1)
    assert_equal(
        dtw_batch(d, sx, sx, symmetric=True),
        dtw_batch(d.cuda(), sx.cuda(), sx.cuda(), symmetric=True).cpu(),
    )
