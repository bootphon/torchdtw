"""Input-validation error paths.

The kernels guard their preconditions with STD_TORCH_CHECK, which surfaces as RuntimeError in
Python. None of these were exercised before. Each test triggers exactly one check and matches a
substring of its message.
"""

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

import pytest
import torch

from torchdtw import dtw, dtw_batch, dtw_path

from .conftest import make_tensor


def _batch(n: int = 2, m: int = 2, x: int = 3, y: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = make_tensor((n, m, x, y), dtype=torch.float32, low=0.0, high=1.0)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y + 1)
    return d, sx, sy


@pytest.mark.parametrize("op", [dtw, dtw_path])
def test_dtw_requires_2d(op) -> None:
    """Dtw / dtw_path reject non-2D inputs."""
    with pytest.raises(RuntimeError, match="2D tensor"):
        op(make_tensor((2, 3, 4), dtype=torch.float32, low=0.0, high=1.0))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
@pytest.mark.parametrize("op", [dtw, dtw_path])
def test_dtw_rejects_empty(op, device: str) -> None:
    """Dtw / dtw_path reject an empty 2D tensor: an empty sequence has no alignment path to normalize by."""
    with pytest.raises(RuntimeError, match="Empty input tensor"):
        op(torch.empty((0, 3), dtype=torch.float32, device=device))


EMPTY_SHAPES = {"n1 == 0": (0, 0, 3, 4), "s1 == 0": (2, 2, 0, 4), "s2 == 0": (2, 2, 3, 0)}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("shape", EMPTY_SHAPES.values(), ids=EMPTY_SHAPES.keys())
def test_dtw_batch_empty_inputs(shape: tuple[int, int, int, int], device: str, symmetric: bool) -> None:
    """dtw_batch accepts an empty batch, or empty sequences: every pair is empty, of cost 0."""
    n1, n2, s1, s2 = shape
    d = torch.rand(shape, device=device)
    expected = torch.zeros((n1, n2), device=device)
    sx = torch.full((n1,), min(s1, 2), dtype=torch.int32, device=device)
    sy = torch.full((n2,), min(s2, 2), dtype=torch.int32, device=device)
    for lengths in ((sx, sy), (sx.long(), -sy.long()), (None, None)):
        torch.testing.assert_close(dtw_batch(d, *lengths, symmetric=symmetric), expected, rtol=0, atol=0)


def test_dtw_batch_empty_sequences_length_exceeds_block() -> None:
    """On CPU, dtw_batch rejects positive lengths of empty sequences, which exceed their padded size of 0."""
    d = torch.rand(2, 2, 3, 0)
    with pytest.raises(RuntimeError, match="sy values must not exceed"):
        dtw_batch(d, sy=torch.tensor([1, 0]))


def test_dtw_batch_requires_4d() -> None:
    """dtw_batch rejects distances that are not 4D."""
    _, sx, sy = _batch()
    d = make_tensor((2, 3), dtype=torch.float32, low=0.0, high=1.0)
    with pytest.raises(RuntimeError, match="4D tensor"):
        dtw_batch(d, sx, sy, symmetric=False)


def test_dtw_batch_requires_1d_lengths() -> None:
    """dtw_batch rejects sx/sy that are not 1D."""
    d, _, sy = _batch()
    sx = make_tensor((2, 1), dtype=torch.long, low=1, high=3)
    with pytest.raises(RuntimeError, match="1D tensors"):
        dtw_batch(d, sx, sy, symmetric=False)


def test_dtw_batch_length_size_mismatch() -> None:
    """dtw_batch rejects sx/sy whose sizes do not match the first two dims of distances."""
    d, _, sy = _batch(n=2)
    sx = make_tensor((3,), dtype=torch.long, low=1, high=3)
    with pytest.raises(RuntimeError, match="sizes must match"):
        dtw_batch(d, sx, sy, symmetric=False)


def test_dtw_batch_symmetric_requires_square() -> None:
    """Symmetric dtw_batch requires distances.size(0) == distances.size(1)."""
    d, sx, sy = _batch(n=2, m=3)  # sx/sy sized to pass the size-match check first
    with pytest.raises(RuntimeError, match="symmetric dtw_batch requires"):
        dtw_batch(d, sx, sy, symmetric=True)


def test_dtw_batch_length_dtype_mismatch() -> None:
    """dtw_batch rejects sx and sy with different integer dtypes."""
    d, _, _ = _batch()
    sx = make_tensor((2,), dtype=torch.int32, low=1, high=3)
    sy = make_tensor((2,), dtype=torch.int64, low=1, high=3)
    with pytest.raises(RuntimeError, match="dtype"):
        dtw_batch(d, sx, sy, symmetric=False)


def test_dtw_batch_length_exceeds_block() -> None:
    """dtw_batch rejects sx values larger than distances.size(2)."""
    d, _, sy = _batch(x=3)
    sx = torch.tensor([5, 1], dtype=torch.long)  # 5 > size(2) == 3
    with pytest.raises(RuntimeError, match="sx values must not exceed"):
        dtw_batch(d, sx, sy, symmetric=False)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_dtw_batch_default_lengths(device: str, symmetric: bool) -> None:
    """Absent sx or sy default to the padded sizes."""
    d = make_tensor((3, 3, 4, 5), dtype=torch.float32, low=0.0, high=1.0, device=device)
    sx = torch.full((3,), 4, device=device)
    sy = torch.full((3,), 5, device=device)
    expected = dtw_batch(d, sx, sy, symmetric=symmetric)
    torch.testing.assert_close(dtw_batch(d, symmetric=symmetric), expected, rtol=0, atol=0)
    torch.testing.assert_close(dtw_batch(d, sx, symmetric=symmetric), expected, rtol=0, atol=0)
    torch.testing.assert_close(dtw_batch(d, sy=sy, symmetric=symmetric), expected, rtol=0, atol=0)


def test_dtw_batch_lengths_must_be_integers() -> None:
    """dtw_batch rejects floating-point sx and sy."""
    d, sx, sy = _batch()
    with pytest.raises(RuntimeError, match="integer dtype"):
        dtw_batch(d, sx.float(), sy.float(), symmetric=False)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.int16])
@pytest.mark.parametrize("symmetric", [False, True])
def test_dtw_batch_cuda_negative_lengths_are_empty(symmetric: bool, dtype: torch.dtype) -> None:
    """On CUDA, negative lengths are empty, as on CPU, for the kernels specialized for int32/int64 lengths or not."""
    d = make_tensor((3, 3, 4, 4), dtype=torch.float32, low=0.0, high=1.0)
    lengths, empty = torch.tensor([4, -2, 3], dtype=dtype), torch.tensor([4, 0, 3], dtype=dtype)
    expected = dtw_batch(d, empty, empty, symmetric=symmetric)
    out = dtw_batch(d.cuda(), lengths.cuda(), lengths.cuda(), symmetric=symmetric)
    torch.testing.assert_close(out.cpu(), expected)


# Each case runs a CUDA op whose lengths exceed the padded sizes. The resulting device-side assert leaves the CUDA
# context unusable, hence a subprocess per case.
OUT_OF_RANGE_CASES = {
    "dtw_batch int32": "torchdtw.dtw_batch(d, torch.tensor([5, 1], dtype=torch.int32, device='cuda'), s32)",
    "dtw_batch int64": "torchdtw.dtw_batch(d, s64, torch.tensor([1, 5], device='cuda'), symmetric=True)",
    "dtw_batch int16": "torchdtw.dtw_batch(d, torch.tensor([5, 1], dtype=torch.int16, device='cuda'))",
    "dtw_batch s2 == 0": "torchdtw.dtw_batch(d[..., :0], s32, torch.tensor([1, 0], dtype=torch.int32, device='cuda'))",
}


@pytest.mark.requires_gpu
@pytest.mark.parametrize("call", OUT_OF_RANGE_CASES.values(), ids=OUT_OF_RANGE_CASES.keys())
def test_cuda_lengths_out_of_range_assert(call: str) -> None:
    """On CUDA, lengths beyond the padded sizes trigger a device-side assert, where the CPU backend raises."""
    script = f"""
import torch, torchdtw
d = torch.rand(2, 2, 3, 4, device="cuda")
s32 = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
s64 = torch.tensor([3, 3], device="cuda")
{call}
torch.cuda.synchronize()
"""
    # The script is built from the constants above.
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode != 0
    assert "must not exceed" in result.stdout + result.stderr


@pytest.mark.requires_gpu
def test_dtw_batch_lengths_must_be_cuda() -> None:
    """On CUDA, dtw_batch requires sx/sy to also be on CUDA."""
    d, sx, sy = _batch()
    with pytest.raises(RuntimeError, match="same device"):
        dtw_batch(d.cuda(), sx, sy, symmetric=False)


@pytest.mark.requires_gpu
def test_dtw_batch_path_length_overflow() -> None:
    """CUDA dtw_batch rejects sequence-length sums exceeding the uint16 path-length capacity."""
    d = torch.zeros((1, 1, 1, 70000), dtype=torch.float32, device="cuda")
    sx = torch.tensor([1], dtype=torch.long, device="cuda")
    sy = torch.tensor([70000], dtype=torch.long, device="cuda")
    with pytest.raises(RuntimeError, match="path-length capacity"):
        dtw_batch(d, sx, sy, symmetric=False)


@pytest.mark.requires_gpu
def test_dtw_batch_grid_dimension_too_large() -> None:
    """CUDA dtw_batch rejects distances.size(1) beyond the CUDA grid.y limit (65535) with a clear message."""
    d = torch.zeros((1, 70000, 1, 1), dtype=torch.float32, device="cuda")
    sx = torch.ones(1, dtype=torch.long, device="cuda")
    sy = torch.ones(70000, dtype=torch.long, device="cuda")
    with pytest.raises(RuntimeError, match="CUDA grid limit"):
        dtw_batch(d, sx, sy, symmetric=False)
