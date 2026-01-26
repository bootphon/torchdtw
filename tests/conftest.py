"""pytest configuration."""

import pytest
import torch
from hypothesis import settings
from hypothesis import strategies as st

settings.register_profile("default", deadline=None)
settings.load_profile("default")

DIM, BATCH = st.integers(1, 1280), st.integers(1, 3)
LOW, HIGH_MINUS_LOW = st.floats(-100, 100), st.floats(0.1, 100)


def make_tensor(shape: tuple[int, ...], *, dtype: torch.dtype, low: float, high: float) -> torch.Tensor:
    """Build a tensor for testing."""
    if low == high and dtype == torch.long:
        return torch.ones(shape, dtype=torch.long, device="cpu")
    return torch.testing.make_tensor(shape, dtype=dtype, device="cpu", low=low, high=high)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert tensors equal."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def pytest_configure(config: pytest.Config) -> None:
    """Add 'requires_gpu' marker."""
    config.addinivalue_line("markers", "requires_gpu: skip test if no GPU is available")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """Skip tests marked with 'requires_gpu' if CUDA not available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "requires_gpu" in item.keywords:
            item.add_marker(skip_gpu)
