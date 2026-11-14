"""Tests for dtw_path.

dtw_path has a CPU-only kernel (csrc/dtw.cpp::dtw_path_cpu) and a device round-trip in the Python wrapper. These
tests cover its output contract (shape/dtype/device), that the returned indices form a valid monotonic warping path,
that it matches the independent reference traceback, ties and +inf masks included, that it is consistent with the
dtw value, and the CUDA round-trip.
"""

import sys

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torchdtw import dtw, dtw_path

from .conftest import HIGH_MINUS_LOW, LOW, assert_equal, make_tensor
from .reference import reference_cost, reference_path

PATH_DIM = st.integers(1, 64)
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
INF = float("inf")

HAND_PATHS = [
    ([[5.0]], [(0, 0)]),
    ([[1.0, 2.0], [3.0, 4.0]], [(0, 0), (1, 1)]),
    ([[1.0, 2.0, 3.0]], [(0, 0), (0, 1), (0, 2)]),
    ([[1.0], [2.0], [3.0]], [(0, 0), (1, 0), (2, 0)]),
    ([[1.0, 3.0, 3.0], [3.0, 1.0, 3.0], [3.0, 3.0, 1.0]], [(0, 0), (1, 1), (2, 2)]),
]


def reference(d: torch.Tensor) -> torch.Tensor:
    """Reference traceback of ``d``, on the CPU."""
    return torch.tensor(reference_path(reference_cost(d.cpu().double().numpy())), dtype=torch.long)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("matrix", "expected"), HAND_PATHS)
def test_dtw_path_hand_examples(matrix: list[list[float]], expected: list[tuple[int, int]], device: str) -> None:
    """dtw_path matches paths computed by hand on tiny matrices."""
    path = dtw_path(torch.tensor(matrix, dtype=torch.float64, device=device))
    assert_equal(path.cpu(), torch.tensor(expected, dtype=torch.long))


@pytest.mark.parametrize("device", DEVICES)
@given(x=PATH_DIM, y=PATH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_path_contract(x: int, y: int, low: float, high_minus_low: float, device: str) -> None:
    """dtw_path returns a (*, 2) long tensor on the input device that is a valid monotonic warping path."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device=device)
    path = dtw_path(d)

    assert path.dtype == torch.long
    assert path.device == d.device
    assert path.ndim == 2
    assert path.size(1) == 2

    p = path.cpu().numpy()
    assert tuple(p[0]) == (0, 0)
    assert tuple(p[-1]) == (x - 1, y - 1)
    steps = np.diff(p, axis=0)
    # Every move is right, down, or diagonal; no move stays in place or goes backwards.
    allowed = {(0, 1), (1, 0), (1, 1)}
    assert all(tuple(step) in allowed for step in steps)


@pytest.mark.parametrize("device", DEVICES)
@given(x=PATH_DIM, y=PATH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_path_matches_reference(x: int, y: int, low: float, high_minus_low: float, device: str) -> None:
    """dtw_path matches the independent reference traceback."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device=device)
    assert_equal(dtw_path(d).cpu(), reference(d))


@pytest.mark.parametrize("device", DEVICES)
@given(x=PATH_DIM, y=PATH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_path_consistent_with_dtw(x: int, y: int, low: float, high_minus_low: float, device: str) -> None:
    """Summing distances along the path equals the corner cost, i.e. dtw == sum / path length."""
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device=device)
    path = dtw_path(d)
    along_path = d[path[:, 0], path[:, 1]].sum()
    torch.testing.assert_close(dtw(d), along_path / path.size(0))


@pytest.mark.parametrize("device", DEVICES)
@given(x=PATH_DIM, y=PATH_DIM, high=st.integers(1, 3))
def test_dtw_path_integer_ties(x: int, y: int, high: int, device: str) -> None:
    """Small integer distances give many tied costs, broken as diag <= left <= up, like the reference."""
    d = torch.randint(0, high + 1, (x, y), dtype=torch.int32, device=device)
    assert_equal(dtw_path(d).cpu(), reference(d))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value", [0.0, 1.0])
@given(x=PATH_DIM, y=PATH_DIM)
def test_dtw_path_constant_ties(x: int, y: int, value: float, device: str) -> None:
    """Constant distances tie everywhere: with 0 every cost is equal, with 1 the diagonal moves are cheaper."""
    d = torch.full((x, y), value, dtype=torch.float32, device=device)
    assert_equal(dtw_path(d).cpu(), reference(d))


@pytest.mark.parametrize("device", DEVICES)
@given(x=PATH_DIM, y=PATH_DIM, p=st.floats(0.0, 0.5), seed=st.integers(0, 2**31 - 1))
def test_dtw_path_inf_mask(x: int, y: int, p: float, seed: int, device: str) -> None:
    """dtw_path routes around a random +inf mask like the reference, finite path or not."""
    g = torch.Generator().manual_seed(seed)
    d = torch.rand((x, y), generator=g, dtype=torch.float64)
    d = d.masked_fill(torch.rand((x, y), generator=g) < p, INF)
    assert_equal(dtw_path(d.to(device)).cpu(), reference(d))


@pytest.mark.parametrize("device", DEVICES)
def test_dtw_path_noncontiguous(device: str) -> None:
    """dtw_path reads strided distances."""
    d = make_tensor((37, 23), dtype=torch.float32, low=0, high=1, device=device, noncontiguous=True)
    assert not d.is_contiguous()
    assert_equal(dtw_path(d).cpu(), reference(d))
    assert_equal(dtw_path(d.t()).cpu(), reference(d.t()))


@pytest.mark.parametrize(
    "device",
    [
        # Inductor compiles C++ for the CPU with cl, which is not on the PATH outside a MSVC developer prompt.
        pytest.param("cpu", marks=pytest.mark.skipif(sys.platform == "win32", reason="Inductor needs cl on Windows")),
        pytest.param("cuda", marks=pytest.mark.requires_gpu),
    ],
)
def test_dtw_path_compile(device: str) -> None:
    """dtw_path compiles with fullgraph=True, its data-dependent output size included."""
    d = make_tensor((17, 29), dtype=torch.float32, low=0, high=1, device=device)
    compiled = torch.compile(dtw_path, fullgraph=True)
    assert_equal(compiled(d), dtw_path(d))
