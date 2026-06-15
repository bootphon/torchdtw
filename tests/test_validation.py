"""Input-validation error paths.

The kernels guard their preconditions with STD_TORCH_CHECK, which surfaces as RuntimeError in
Python. None of these were exercised before. Each test triggers exactly one check and matches a
substring of its message.
"""

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
def test_dtw_requires_2d(op) -> None:  # noqa: ANN001
    """Dtw / dtw_path reject non-2D inputs."""
    with pytest.raises(RuntimeError, match="2D tensor"):
        op(make_tensor((2, 3, 4), dtype=torch.float32, low=0.0, high=1.0))


@pytest.mark.parametrize("op", [dtw, dtw_path])
def test_dtw_rejects_empty(op) -> None:  # noqa: ANN001
    """Dtw / dtw_path reject an empty 2D tensor."""
    with pytest.raises(RuntimeError, match="Empty input tensor"):
        op(torch.empty((0, 3), dtype=torch.float32))


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


@pytest.mark.requires_gpu
def test_dtw_batch_lengths_must_be_cuda() -> None:
    """On CUDA, dtw_batch requires sx/sy to also be on CUDA."""
    d, sx, sy = _batch()
    with pytest.raises(RuntimeError, match="must be on CUDA"):
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


@pytest.mark.requires_gpu
def test_dtw_batch_shared_memory_too_large() -> None:
    """CUDA dtw_batch rejects distances.size(3) too large to fit the dtype's shared-memory budget."""
    d = torch.zeros((1, 1, 1, 1700), dtype=torch.float64, device="cuda")  # over the ~1638 float64 cap
    sx = torch.tensor([1], dtype=torch.long, device="cuda")
    sy = torch.tensor([1700], dtype=torch.long, device="cuda")
    with pytest.raises(RuntimeError, match="shared memory"):
        dtw_batch(d, sx, sy, symmetric=False)
