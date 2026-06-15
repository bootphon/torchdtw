"""Multi-GPU: the kernels run correctly on a non-default CUDA device.

The CUDA launch derives its device index and stream from getCurrentDeviceIndex() / the current
stream (see csrc/cuda/dtw.cu), relying on the dispatcher's device guard to set the current device
to the input tensor's device. If that assumption broke, a tensor on a non-default device would be
read on the wrong device's stream -- an illegal access or silently wrong result. These tests put
data on a non-default device (also under a side stream there, and on two devices at once) and check
the results against the independent reference. Skipped unless at least 2 GPUs are visible.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torchdtw import dtw, dtw_batch

from .conftest import HIGH_MINUS_LOW, LOW, assert_equal, fixed_batch, make_tensor
from .reference import reference_dtw, reference_dtw_batch

MG_DIM = st.integers(1, 64)
MG_BATCH = st.integers(1, 3)
MG_BATCH_DIM = st.integers(1, 24)


def _other_device() -> torch.device:
    """Non-default CUDA device (the last visible one), distinct from the default cuda:0."""
    return torch.device("cuda", torch.cuda.device_count() - 1)


@pytest.mark.requires_multi_gpu
@given(x=MG_DIM, y=MG_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_non_default_device(x: int, y: int, low: float, high_minus_low: float) -> None:
    """Dtw on a non-default device returns on that device and matches the reference."""
    other = _other_device()
    d = make_tensor((x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device=str(other))
    out = dtw(d)
    assert out.device == other
    expected = torch.tensor(reference_dtw(d.cpu().numpy()), dtype=torch.float64)
    assert_equal(out.cpu(), expected)


@pytest.mark.requires_multi_gpu
@settings(max_examples=50)
@given(n=MG_BATCH, m=MG_BATCH, x=MG_BATCH_DIM, y=MG_BATCH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_non_default_device(n: int, m: int, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Asymmetric dtw_batch on a non-default device returns on that device and matches the reference."""
    other = _other_device()
    dev = str(other)
    d = make_tensor((n, m, x, y), dtype=torch.float64, low=low, high=high_minus_low + low, device=dev)
    sx = make_tensor((n,), dtype=torch.long, low=0, high=x + 1, device=dev)
    sy = make_tensor((m,), dtype=torch.long, low=0, high=y + 1, device=dev)
    out = dtw_batch(d, sx, sy, symmetric=False)
    assert out.device == other
    expected = torch.from_numpy(
        reference_dtw_batch(d.cpu().numpy(), sx.cpu().numpy(), sy.cpu().numpy(), symmetric=False)
    )
    assert_equal(out.cpu(), expected)


@pytest.mark.requires_multi_gpu
@settings(max_examples=50)
@given(n=MG_BATCH, x=MG_BATCH_DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_symmetric_non_default_device(n: int, x: int, low: float, high_minus_low: float) -> None:
    """Symmetric dtw_batch (triangular index path) on a non-default device matches the reference."""
    other = _other_device()
    dev = str(other)
    d = make_tensor((n, n, x, x), dtype=torch.float64, low=low, high=high_minus_low + low, device=dev)
    sx = make_tensor((n,), dtype=torch.long, low=0, high=x + 1, device=dev)
    out = dtw_batch(d, sx, sx, symmetric=True)
    assert out.device == other
    expected = torch.from_numpy(
        reference_dtw_batch(d.cpu().numpy(), sx.cpu().numpy(), sx.cpu().numpy(), symmetric=True)
    )
    assert_equal(out.cpu(), expected)


@pytest.mark.requires_multi_gpu
def test_dtw_batch_lengths_must_match_distances_device() -> None:
    """sx/sy on a different CUDA device than distances are rejected with a clear message."""
    other = _other_device()
    d, sx = fixed_batch()
    d = d.to("cuda:0")
    sx_other = sx.to(other)
    with pytest.raises(RuntimeError, match="same CUDA device"):
        dtw_batch(d, sx_other, sx_other, symmetric=False)


@pytest.mark.requires_multi_gpu
def test_dtw_batch_non_default_device_side_stream() -> None:
    """dtw_batch on a side stream of a non-default device must honour both the device and the stream."""
    other = _other_device()
    d, sx = fixed_batch()
    expected = dtw_batch(d, sx, sx, symmetric=False)

    d, sx = d.to(other), sx.to(other)
    stream = torch.cuda.Stream(device=other)
    with torch.cuda.stream(stream):
        out = dtw_batch(d, sx, sx, symmetric=False)
    stream.synchronize()

    assert out.device == other
    assert_equal(out.cpu(), expected)


@pytest.mark.requires_multi_gpu
def test_dtw_batch_two_devices_at_once() -> None:
    """Running on cuda:0 and a non-default device back to back yields correct, independent results."""
    d, sx = fixed_batch()
    expected = dtw_batch(d, sx, sx, symmetric=False)

    dev0 = torch.device("cuda", 0)
    dev1 = _other_device()
    out0 = dtw_batch(d.to(dev0), sx.to(dev0), sx.to(dev0), symmetric=False)
    out1 = dtw_batch(d.to(dev1), sx.to(dev1), sx.to(dev1), symmetric=False)
    torch.cuda.synchronize(dev0)
    torch.cuda.synchronize(dev1)

    assert out0.device == dev0
    assert out1.device == dev1
    assert_equal(out0.cpu(), expected)
    assert_equal(out1.cpu(), expected)
