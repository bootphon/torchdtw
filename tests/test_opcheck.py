"""Check for compatibility with torch.compile."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.library import opcheck

import torchdtw

from .conftest import BATCH, DIM, HIGH_MINUS_LOW, LOW, make_tensor


@given(x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_dtw(x: int, y: int, low: float, high_minus_low: float) -> None:
    """Verify that dtw can be torch compiled."""
    sample = make_tensor((x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    opcheck(torch.ops.torchdtw.dtw.default, (sample,))


@pytest.mark.requires_gpu
@given(x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_dtw_cuda(x: int, y: int, low: float, high_minus_low: float) -> None:
    """Verify that dtw can be torch compiled on CUDA."""
    sample = make_tensor((x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    opcheck(torch.ops.torchdtw.dtw.default, (sample.cuda(),))


@given(n=BATCH, x=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_dtw_batch_symmetric(n: int, x: int, low: float, high_minus_low: float) -> None:
    """Verify that dtw_batch can be torch compiled, with symmetric input."""
    sample = make_tensor((n, n, x, x), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x)
    i, j = torch.triu_indices(n, n)
    sample[i, j] = sample[j, i]
    opcheck(torch.ops.torchdtw.dtw_batch.default, (sample, sx, sx), {"symmetric": True})


@pytest.mark.requires_gpu
@given(n=BATCH, x=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_dtw_batch_symmetric_cuda(n: int, x: int, low: float, high_minus_low: float) -> None:
    """Verify that dtw_batch can be torch compiled on CUDA, with symmetric input."""
    sample = make_tensor((n, n, x, x), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x)
    i, j = torch.triu_indices(n, n)
    sample[i, j] = sample[j, i]
    opcheck(torch.ops.torchdtw.dtw_batch.default, (sample.cuda(), sx.cuda(), sx.cuda()), {"symmetric": True})


@given(n=BATCH, m=BATCH, x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_dtw_batch_not_symmetric(n: int, m: int, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Verify that dtw_batch can be torch compiled, with symmetric input."""
    sample = make_tensor((n, m, x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y)
    opcheck(torch.ops.torchdtw.dtw_batch.default, (sample, sx, sy), {"symmetric": False})


@pytest.mark.requires_gpu
@given(n=BATCH, m=BATCH, x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_dtw_batch_not_symmetric_cuda(
    n: int, m: int, x: int, y: int, low: float, high_minus_low: float
) -> None:
    """Verify that dtw_batch can be torch compiled on CUDA, with symmetric input."""
    sample = make_tensor((n, m, x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y)
    opcheck(torch.ops.torchdtw.dtw_batch.default, (sample.cuda(), sx.cuda(), sy.cuda()), {"symmetric": False})


SOFT_DIM = st.integers(1, 64)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
@given(n=BATCH, x=SOFT_DIM, y=SOFT_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_soft_dtw(device: str, n: int, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Verify that soft_dtw, and its backward, can be torch compiled.

    The raw op is not differentiable (soft_dtw is differentiated by an autograd.Function), so its inputs do not
    require grad here, and the backward op is checked on its own.
    """
    sample = make_tensor((n, x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device=device)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x, device=device)
    for keep_r in (False, True):
        opcheck(torch.ops.torchdtw.soft_dtw.default, (sample, sx, None, 1.0, 2, False, keep_r))
    cost, R = torch.ops.torchdtw.soft_dtw.default(sample, sx, None, 1.0, 2, symmetric=False, keep_R=True)
    opcheck(torch.ops.torchdtw.soft_dtw_backward.default, (torch.rand_like(cost), sample, R, sx, None, 1.0, False))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
@given(n=BATCH, m=BATCH, x=SOFT_DIM, y=SOFT_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_soft_dtw_batch(
    device: str, n: int, m: int, x: int, y: int, low: float, high_minus_low: float
) -> None:
    """Verify that soft_dtw on all pairs, symmetric or not, and its backward, can be torch compiled."""
    for symmetric in (False, True):
        m = n if symmetric else m
        sample = make_tensor((n, m, x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device=device)
        sy = make_tensor((m,), dtype=torch.long, low=1, high=y, device=device).int()
        args = (sample, None, sy, 0.5, -1, symmetric)
        opcheck(torch.ops.torchdtw.soft_dtw.default, (*args, True))
        cost, R = torch.ops.torchdtw.soft_dtw.default(*args, keep_R=True)
        opcheck(
            torch.ops.torchdtw.soft_dtw_backward.default,
            (torch.rand_like(cost), sample, R, None, sy, 0.5, symmetric),
        )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
@given(n=BATCH, m=BATCH, x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_opcheck_dtw_batch_without_lengths(
    device: str, n: int, m: int, x: int, y: int, low: float, high_minus_low: float
) -> None:
    """Verify that dtw_batch without sx, or sy, can be torch compiled."""
    sample = make_tensor((n, m, x, y), dtype=torch.float32, low=low, high=high_minus_low + low, device=device)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y, device=device)
    opcheck(torch.ops.torchdtw.dtw_batch.optional_lengths, (sample, None, None), {"symmetric": False})
    opcheck(torch.ops.torchdtw.dtw_batch.optional_lengths, (sample, None, sy), {"symmetric": False})


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
def test_compile_soft_dtw_autograd(device: str) -> None:
    """soft_dtw compiles in a single graph, with the same costs and gradients as in eager mode."""

    def loss(d: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
        return torchdtw.soft_dtw(d, sx, gamma=0.5, bandwidth=2).sum()

    torch.compiler.reset()
    sx = torch.tensor([5, 3, 1], device=device)
    d = torch.rand(3, 5, 6, dtype=torch.float64, device=device, requires_grad=True)
    d_eager = d.detach().clone().requires_grad_()
    torch.compile(loss, fullgraph=True)(d, sx).backward()
    loss(d_eager, sx).backward()
    torch.testing.assert_close(d.grad, d_eager.grad, rtol=0, atol=0)
