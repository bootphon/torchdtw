"""Exercise the CUDA dtw kernels on non-default streams and under CUDA graph capture.

These guard against launching the kernel on the wrong CUDA stream. Deriving the launch
stream from ``torch::stable::accelerator::getCurrentStream(device).id()`` is a bug:
``id()`` returns an opaque, packed ``c10::StreamId``, not a ``cudaStream_t``. The two
only coincide for the default stream (both 0), so the cast silently works in ordinary
eager execution but yields a garbage stream handle on any side stream -- which makes the
kernel launch fail outright and makes CUDA-graph capture (the intended way to amortize
launch overhead for many small calls) impossible.
"""

import pytest
import torch

from torchdtw import dtw, dtw_batch

from .conftest import assert_equal, fixed_batch


@pytest.mark.requires_gpu
def test_dtw_batch_side_stream() -> None:
    """dtw_batch launched on a non-default stream must match the default-stream result."""
    d, sx = fixed_batch()
    d, sx = d.cuda(), sx.cuda()
    expected = dtw_batch(d, sx, sx, symmetric=False)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = dtw_batch(d, sx, sx, symmetric=False)
    stream.synchronize()
    assert_equal(actual, expected)


@pytest.mark.requires_gpu
def test_dtw_single_side_stream() -> None:
    """The 2D dtw entry point shares the launch path and must also honour the stream."""
    g = torch.Generator().manual_seed(0)
    d = torch.rand((48, 64), generator=g, dtype=torch.float32).cuda()
    expected = dtw(d)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = dtw(d)
    stream.synchronize()
    assert_equal(actual, expected)


@pytest.mark.requires_gpu
def test_dtw_batch_cuda_graph() -> None:
    """dtw_batch must be capturable into a CUDA graph and replay to the right result."""
    d, sx = fixed_batch()
    d, sx = d.cuda(), sx.cuda()
    expected = dtw_batch(d, sx, sx, symmetric=False)
    torch.cuda.synchronize()

    # Capture requires a prior warmup on a side stream.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            dtw_batch(d, sx, sx, symmetric=False)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = dtw_batch(d, sx, sx, symmetric=False)
    graph.replay()
    torch.cuda.synchronize()
    assert_equal(out, expected)


@pytest.mark.requires_gpu
def test_dtw_batch_compile_cuda_graph() -> None:
    """dtw_batch must compose with torch.compile(mode="reduce-overhead").

    That mode wraps the compiled region in CUDA graphs (capture on the warmup iterations,
    replay afterwards), so it exercises the same capturability as the manual graph test
    through the torch.compile path -- and requires the registered fake kernel to trace.
    Each replay must return the eager result.
    """
    d, sx = fixed_batch()
    d, sx = d.cuda(), sx.cuda()
    expected = dtw_batch(d, sx, sx, symmetric=False)
    torch.cuda.synchronize()

    torch.compiler.reset()
    compiled = torch.compile(dtw_batch, mode="reduce-overhead", fullgraph=True)
    # The first iterations warm up and capture; later ones replay the graph. reduce-overhead
    # reuses a static output buffer, so clone before the next call overwrites it.
    actual = torch.zeros_like(expected)
    for _ in range(4):
        actual = compiled(d, sx, sx, symmetric=False).clone()
    torch.cuda.synchronize()
    assert_equal(actual, expected)
