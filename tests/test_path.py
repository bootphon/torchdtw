"""Tests for dtw_path, which is otherwise entirely uncovered.

dtw_path has a CPU-only kernel (csrc/dtw.cpp::dtw_path_cpu) and a device round-trip in the
Python wrapper. These tests cover its output contract (shape/dtype), that the returned indices
form a valid monotonic warping path, that it matches the independent reference traceback, that
it is consistent with the dtw value, and the CUDA round-trip.
"""

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torchdtw import dtw, dtw_path

from .conftest import HIGH_MINUS_LOW, LOW, assert_equal, make_tensor
from .reference import reference_cost, reference_path

PATH_DIM = st.integers(1, 64)

HAND_PATHS = [
    ([[5.0]], [(0, 0)]),
    ([[1.0, 2.0], [3.0, 4.0]], [(0, 0), (1, 1)]),
    ([[1.0, 2.0, 3.0]], [(0, 0), (0, 1), (0, 2)]),
    ([[1.0], [2.0], [3.0]], [(0, 0), (1, 0), (2, 0)]),
    ([[1.0, 3.0, 3.0], [3.0, 1.0, 3.0], [3.0, 3.0, 1.0]], [(0, 0), (1, 1), (2, 2)]),
]


@pytest.mark.parametrize(("matrix", "expected"), HAND_PATHS)
def test_dtw_path_hand_examples(matrix: list[list[float]], expected: list[tuple[int, int]]) -> None:
    """dtw_path matches paths computed by hand on tiny matrices."""
    path = dtw_path(torch.tensor(matrix, dtype=torch.float64))
    assert_equal(path, torch.tensor(expected, dtype=torch.long))


@given(x=PATH_DIM, y=PATH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_path_contract(x: int, y: int, low: float, high_minus_low: float) -> None:
    """dtw_path returns a (*, 2) long tensor that is a valid monotonic warping path."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low)
    path = dtw_path(d)

    assert path.dtype == torch.long
    assert path.ndim == 2
    assert path.size(1) == 2

    p = path.numpy()
    assert tuple(p[0]) == (0, 0)
    assert tuple(p[-1]) == (x - 1, y - 1)
    steps = np.diff(p, axis=0)
    # Every move is right, down, or diagonal; no move stays in place or goes backwards.
    allowed = {(0, 1), (1, 0), (1, 1)}
    assert all(tuple(step) in allowed for step in steps)


@given(x=PATH_DIM, y=PATH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_path_matches_reference(x: int, y: int, low: float, high_minus_low: float) -> None:
    """dtw_path matches the independent reference traceback."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low)
    expected = torch.tensor(reference_path(reference_cost(d.numpy())), dtype=torch.long)
    assert_equal(dtw_path(d), expected)


@given(x=PATH_DIM, y=PATH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_path_consistent_with_dtw(x: int, y: int, low: float, high_minus_low: float) -> None:
    """Summing distances along the path equals the corner cost, i.e. dtw == sum / path length."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low)
    path = dtw_path(d)
    along_path = d[path[:, 0], path[:, 1]].sum()
    torch.testing.assert_close(dtw(d), along_path / path.size(0))


@pytest.mark.requires_gpu
@given(x=PATH_DIM, y=PATH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_path_cuda_round_trip(x: int, y: int, low: float, high_minus_low: float) -> None:
    """dtw_path on a CUDA tensor returns the same path on the input device."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low)
    path = dtw_path(d.cuda())
    assert path.is_cuda
    assert_equal(path.cpu(), dtw_path(d))
