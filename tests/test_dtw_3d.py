"""Tests of dtw on 3D distances: B independent pairs, with optional per-pair lengths sx, sy."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.library import opcheck

from torchdtw import dtw, dtw_batch
from torchdtw._errors import SinglePairLengthsError

from .conftest import assert_equal, make_tensor

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.uint8, torch.int64]
LENGTHS_DTYPES = [torch.int32, torch.int64, torch.uint8, torch.int16, torch.uint64]


def _paired(b: int, n: int, m: int, dtype: torch.dtype, device: str, *, lengths_dtype: torch.dtype = torch.int32):
    g = torch.Generator().manual_seed(b * 1009 + n * 31 + m)
    if dtype.is_floating_point:
        d = torch.rand((b, n, m), generator=g, dtype=torch.float32).to(dtype)
    else:
        d = torch.randint(0, 50, (b, n, m), generator=g).to(dtype)
    sx = torch.randint(1, n + 1, (b,), generator=g).to(lengths_dtype)
    sy = torch.randint(1, m + 1, (b,), generator=g).to(lengths_dtype)
    return d.to(device), sx.to(device), sy.to(device)


def _per_pair(d: torch.Tensor, sx: torch.Tensor | None, sy: torch.Tensor | None, *, normalize: bool) -> torch.Tensor:
    out = []
    for b in range(d.shape[0]):
        n = int(sx[b]) if sx is not None else d.shape[1]
        m = int(sy[b]) if sy is not None else d.shape[2]
        out.append(dtw(d[b, :n, :m], normalize=normalize))
    return torch.stack(out) if out else d.new_empty((0,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("with_lengths", ["both", "sx", "sy", "none"])
def test_dtw_3d_matches_per_pair(device: str, dtype: torch.dtype, normalize: bool, with_lengths: str) -> None:
    """3D dtw is the per-pair 2D dtw on the (sx[b], sy[b]) sub-blocks, bit for bit."""
    d, sx, sy = _paired(7, 9, 13, dtype, device)
    sx = sx if with_lengths in {"both", "sx"} else None
    sy = sy if with_lengths in {"both", "sy"} else None
    cost = dtw(d, sx, sy, normalize=normalize)
    assert cost.shape == (7,)
    assert cost.dtype == dtype
    assert cost.device == d.device
    assert_equal(cost, _per_pair(d, sx, sy, normalize=normalize))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lengths_dtype", LENGTHS_DTYPES)
def test_dtw_3d_matches_dtw_batch(device: str, lengths_dtype: torch.dtype) -> None:
    """3D dtw == dtw_batch on (B, 1, N, M) with a uniform sy, and == the diagonal of dtw_batch on all pairs."""
    d, sx, sy = _paired(6, 11, 11, torch.float32, device, lengths_dtype=lengths_dtype)
    for normalize in (True, False):
        cost = dtw(d, sx, sy, normalize=normalize)
        uniform = sy[:1]
        assert_equal(
            dtw(d, sx, uniform.expand(6), normalize=normalize),
            dtw_batch(d[:, None], sx, uniform, normalize=normalize)[:, 0],
        )
        all_pairs = d[:, None].expand(6, 6, 11, 11)
        assert_equal(cost, dtw_batch(all_pairs, sx, sy, normalize=normalize).diagonal())


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lengths_dtype", LENGTHS_DTYPES)
@given(b=st.integers(1, 9), n=st.integers(1, 40), m=st.integers(1, 40))
def test_dtw_3d_cuda_parity(dtype: torch.dtype, lengths_dtype: torch.dtype, b: int, n: int, m: int) -> None:
    """CPU and CUDA 3D dtw are bit-identical."""
    d, sx, sy = _paired(b, n, m, dtype, "cpu", lengths_dtype=lengths_dtype)
    for normalize in (True, False):
        assert_equal(
            dtw(d.cuda(), sx.cuda(), sy.cuda(), normalize=normalize).cpu(), dtw(d, sx, sy, normalize=normalize)
        )
        assert_equal(dtw(d.cuda(), normalize=normalize).cpu(), dtw(d, normalize=normalize))


@pytest.mark.parametrize("device", DEVICES)
def test_dtw_3d_noncontiguous(device: str) -> None:
    """Strided distances and lengths are read through their strides."""
    d = make_tensor((5, 8, 10), dtype=torch.float32, low=0, high=1, device=device, noncontiguous=True)
    lengths = torch.tensor([8, 1, 3, 8, 5, 2, 7, 6, 4, 8], dtype=torch.int32, device=device)
    sx, sy = lengths[::2], lengths[1::2]
    assert_equal(dtw(d, sx, sy), _per_pair(d, sx, sy, normalize=True))
    t = d.transpose(1, 2)
    assert_equal(dtw(t, sy, sx), _per_pair(t, sy, sx, normalize=True))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(0, 3, 4), (3, 0, 4), (3, 4, 0), (0, 0, 0)])
def test_dtw_3d_empty(device: str, shape: tuple[int, int, int]) -> None:
    """B == 0, N == 0 and M == 0 are accepted, as in dtw_batch: empty pairs cost 0."""
    d = torch.empty(shape, device=device)
    for normalize in (True, False):
        cost = dtw(d, normalize=normalize)
        assert cost.shape == (shape[0],)
        assert_equal(cost, torch.zeros(shape[0], device=device))
        lengths = torch.zeros(shape[0], dtype=torch.int32, device=device)
        assert_equal(dtw(d, lengths, lengths, normalize=normalize), cost)


@pytest.mark.parametrize("device", DEVICES)
def test_dtw_3d_zero_and_negative_lengths(device: str) -> None:
    """A length <= 0 costs 0, as in dtw_batch."""
    d, sx, sy = _paired(4, 5, 6, torch.float32, device)
    sx = torch.tensor([0, 3, -2, 5], dtype=torch.int32, device=device)
    cost = dtw(d, sx, sy)
    assert_equal(cost[[0, 2]], torch.zeros(2, device=device))
    assert_equal(cost[[1, 3]], _per_pair(d[[1, 3]], sx[[1, 3]], sy[[1, 3]], normalize=True))


def test_dtw_3d_length_too_large_cpu() -> None:
    """Out-of-range lengths fail hard on the CPU, as in dtw_batch."""
    d, sx, sy = _paired(3, 4, 5, torch.float32, "cpu")
    with pytest.raises(RuntimeError, match=r"sx values must not exceed"):
        dtw(d, torch.tensor([1, 5, 1], dtype=torch.int32), sy)
    with pytest.raises(RuntimeError, match=r"sy values must not exceed"):
        dtw(d, sx, torch.tensor([1, 1, 6], dtype=torch.int32))


@pytest.mark.parametrize("device", DEVICES)
def test_dtw_3d_validation(device: str) -> None:
    """2D dtw rejects sx and sy; 3D dtw checks the lengths as dtw_batch."""
    d, sx, sy = _paired(3, 4, 5, torch.float32, device)
    with pytest.raises(SinglePairLengthsError):
        dtw(d[0], sx, sy)
    with pytest.raises(SinglePairLengthsError):
        dtw(d[0], sy=sy)
    with pytest.raises(RuntimeError, match="sizes must match"):
        dtw(d, sx[:2], sy)
    with pytest.raises(RuntimeError, match="sizes must match"):
        dtw(d, sx, sy[:2])
    with pytest.raises(RuntimeError, match="dtypes do not match"):
        dtw(d, sx, sy.long())
    with pytest.raises(RuntimeError, match="integer dtype"):
        dtw(d, sx.float(), sy.float())
    with pytest.raises(RuntimeError, match="1D"):
        dtw(d, sx[:, None], sy)


def test_dtw_batch_still_rejects_3d() -> None:
    """The 3D overloads of dtw do not make the dtw_batch ops accept 3D distances."""
    d, sx, sy = _paired(3, 4, 5, torch.float32, "cpu")
    with pytest.raises(RuntimeError, match="4D"):
        torch.ops.torchdtw.dtw_batch.default(d, sx, sy, False, True)
    with pytest.raises(RuntimeError, match="3D"):
        torch.ops.torchdtw.dtw.lengths(d[:, None], sx, sy, True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("normalize", [True, False])
def test_opcheck_dtw_3d(device: str, normalize: bool) -> None:
    """Both 3D overloads of dtw can be torch compiled."""
    d, sx, sy = _paired(4, 6, 7, torch.float32, device)
    opcheck(torch.ops.torchdtw.dtw.lengths, (d, sx, sy, normalize))
    opcheck(torch.ops.torchdtw.dtw.optional_lengths, (d, sx, None, normalize))
    opcheck(torch.ops.torchdtw.dtw.optional_lengths, (d, None, None, normalize))
    opcheck(torch.ops.torchdtw.dtw.optional_lengths, (torch.empty(0, 3, 4, device=device), None, None, normalize))


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((torch.empty(2, 1, 3, 4), None, None), "3D"),
        ((torch.empty(2, 3, 4, dtype=torch.bool), None, None), "not implemented"),
        ((torch.empty(2, 3, 4), torch.ones(3, dtype=torch.int32), None), "sizes must match"),
        ((torch.empty(2, 3, 4), torch.ones(2, dtype=torch.int32), torch.ones(2, dtype=torch.long)), "dtypes"),
    ],
)
def test_fake_dtw_3d_rejects_like_cpu(args: tuple, match: str) -> None:
    """The fake 3D dtw kernel rejects the same invalid inputs as the CPU kernel."""
    op = torch.ops.torchdtw.dtw.optional_lengths
    with pytest.raises(RuntimeError, match=match):
        op(*args, True)
    with FakeTensorMode() as mode, pytest.raises(RuntimeError, match=match):
        op(*(mode.from_tensor(a) if a is not None else None for a in args), True)


def test_dtw_3d_compile() -> None:
    """3D dtw runs under torch.compile with fullgraph."""
    d, sx, sy = _paired(4, 6, 7, torch.float32, "cpu")
    compiled = torch.compile(dtw, fullgraph=True)
    assert_equal(compiled(d, sx, sy), dtw(d, sx, sy))
