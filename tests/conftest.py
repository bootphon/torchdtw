"""pytest configuration."""

import pytest
import torch
from hypothesis import settings
from hypothesis import strategies as st

settings.register_profile("default", deadline=None)
settings.load_profile("default")

DIM, BATCH = st.integers(1, 1280), st.integers(1, 3)
LOW, HIGH_MINUS_LOW = st.floats(-100, 100), st.floats(0.1, 100)


def make_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    low: float,
    high: float,
    device: str = "cpu",
    noncontiguous: bool = False,
) -> torch.Tensor:
    """Build a tensor for testing.

    ``noncontiguous=True`` yields a strided (non-contiguous) tensor; build it on the target
    ``device`` directly, since moving a non-contiguous tensor with ``.cuda()`` re-packs it.
    """
    if low == high and dtype == torch.long:
        return torch.ones(shape, dtype=torch.long, device=device)
    return torch.testing.make_tensor(
        shape, dtype=dtype, device=device, low=low, high=high, noncontiguous=noncontiguous
    )


def assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert tensors equal."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic CPU batch ``(distances, lengths)`` for the stream / multi-GPU tests."""
    g = torch.Generator().manual_seed(0)
    d = torch.rand((8, 8, 32, 32), generator=g, dtype=torch.float32)
    sx = torch.randint(1, 33, (8,), generator=g, dtype=torch.long)
    return d, sx


def pytest_configure(config: pytest.Config) -> None:
    """Register GPU markers."""
    config.addinivalue_line("markers", "requires_gpu: skip test if no GPU is available")
    config.addinivalue_line("markers", "requires_multi_gpu: skip test if fewer than 2 GPUs are available")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """Skip tests requiring (multiple) GPUs when the hardware is not available."""
    cuda_available = torch.cuda.is_available()
    multi_gpu = cuda_available and torch.cuda.device_count() >= 2
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_multi_gpu = pytest.mark.skip(reason="fewer than 2 GPUs available")
    for item in items:
        if "requires_gpu" in item.keywords and not cuda_available:
            item.add_marker(skip_gpu)
        if "requires_multi_gpu" in item.keywords and not multi_gpu:
            item.add_marker(skip_multi_gpu)
