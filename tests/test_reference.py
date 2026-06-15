"""Ground-truth correctness: anchor the kernels to an independent NumPy reference.

Every other correctness test compares the two backends against each other (CPU vs CUDA) or
a reduced-precision dtype against float32. Those are blind to a bug shared by both backends.
These tests pin the computed value against a separate float64 implementation (tests/reference.py)
and against a handful of hand-computed examples, so the actual DTW *value* is anchored to a
known-correct result.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torchdtw import dtw, dtw_batch

from .conftest import HIGH_MINUS_LOW, LOW, assert_equal, make_tensor
from .reference import reference_dtw, reference_dtw_batch

# Pure-Python reference loops over every cell, so keep the oracle dimensions small.
ORACLE_DIM = st.integers(1, 64)
ORACLE_BATCH_DIM = st.integers(1, 24)
ORACLE_BATCH = st.integers(1, 3)

# Hand-computed (distances, expected DTW cost) pairs. See the docstring arithmetic in the repo
# history; each is small enough to verify the cost matrix and path length by hand.
HAND_EXAMPLES = [
    ([[5.0]], 5.0),
    ([[1.0, 2.0], [3.0, 4.0]], 2.5),  # cost corner 5, path (0,0)->(1,1) length 2
    ([[1.0, 2.0, 3.0]], 2.0),  # single row: cost 6, path length 3
    ([[1.0], [2.0], [3.0]], 2.0),  # single column: cost 6, path length 3
    ([[1.0, 3.0, 3.0], [3.0, 1.0, 3.0], [3.0, 3.0, 1.0]], 1.0),  # diagonal path, cost 3, length 3
]


@pytest.mark.parametrize(("matrix", "expected"), HAND_EXAMPLES)
def test_dtw_hand_examples(matrix: list[list[float]], expected: float) -> None:
    """Dtw matches values computed by hand on tiny matrices."""
    d = torch.tensor(matrix, dtype=torch.float64)
    assert_equal(dtw(d), torch.tensor(expected, dtype=torch.float64))


@given(x=ORACLE_DIM, y=ORACLE_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_matches_reference(x: int, y: int, low: float, high_minus_low: float) -> None:
    """Dtw matches the independent float64 reference."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low)
    expected = torch.tensor(reference_dtw(d.numpy()), dtype=torch.float64)
    assert_equal(dtw(d), expected)


@pytest.mark.requires_gpu
@given(x=ORACLE_DIM, y=ORACLE_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_cuda_matches_reference(x: int, y: int, low: float, high_minus_low: float) -> None:
    """The CUDA dtw kernel matches the independent float64 reference."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low)
    expected = torch.tensor(reference_dtw(d.numpy()), dtype=torch.float64)
    assert_equal(dtw(d.cuda()).cpu(), expected)


@settings(max_examples=50)
@given(n=ORACLE_BATCH, m=ORACLE_BATCH, x=ORACLE_BATCH_DIM, y=ORACLE_BATCH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_matches_reference(n: int, m: int, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Asymmetric dtw_batch matches the independent float64 reference, including zero lengths."""
    d = make_tensor((n, m, x, y), dtype=torch.float64, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=0, high=x + 1)
    sy = make_tensor((m,), dtype=torch.long, low=0, high=y + 1)
    expected = torch.from_numpy(reference_dtw_batch(d.numpy(), sx.numpy(), sy.numpy(), symmetric=False))
    assert_equal(dtw_batch(d, sx, sy, symmetric=False), expected)


@settings(max_examples=50)
@given(n=ORACLE_BATCH, x=ORACLE_BATCH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_symmetric_matches_reference(n: int, x: int, low: float, high_minus_low: float) -> None:
    """Symmetric dtw_batch matches the reference on CPU (this path is otherwise GPU-only tested)."""
    d = make_tensor((n, n, x, x), dtype=torch.float64, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=0, high=x + 1)
    expected = torch.from_numpy(reference_dtw_batch(d.numpy(), sx.numpy(), sx.numpy(), symmetric=True))
    assert_equal(dtw_batch(d, sx, sx, symmetric=True), expected)
