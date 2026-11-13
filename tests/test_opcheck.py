"""Check for compatibility with torch.compile."""

import pytest
import torch
from hypothesis import given
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.library import opcheck

import torchdtw  # ruff: ignore[unused-import] # Need to import it to register dtw operation

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


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((torch.empty(2, 2, 3, 3), torch.empty(3, dtype=torch.long), None, False), "batch dimensions"),
        ((torch.empty(2, 2, 3, 3), torch.empty(2, 1, dtype=torch.long), None, False), "1D"),
        ((torch.empty(2, 2, 3, 3), torch.empty(2), None, False), "integer dtype"),
        ((torch.empty(2, 2, 3, 3), torch.empty(2, dtype=torch.bool), None, False), "integer dtype"),
        (
            (torch.empty(2, 2, 3, 3), torch.empty(2, dtype=torch.int32), torch.empty(2, dtype=torch.long), False),
            "dtypes do not match",
        ),
        ((torch.empty(2, 3, 3, 3), None, None, True), "symmetric"),
        ((torch.empty(2, 2, 3, 3, dtype=torch.bool), None, None, False), "not implemented"),
        ((torch.empty(2, 2, 3), None, None, False), "4D"),
    ],
)
def test_fake_dtw_batch_rejects_like_cpu(args: tuple, match: str) -> None:
    """The fake dtw_batch kernel rejects the same invalid inputs as the CPU kernel."""
    op = torch.ops.torchdtw.dtw_batch.optional_lengths
    with pytest.raises(RuntimeError, match=match):
        op(*args)
    with FakeTensorMode() as mode, pytest.raises(RuntimeError, match=match):
        op(*(mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args))
