"""Soft-DTW: values and gradients against independent references, CPU/CUDA parity, and input handling."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torchdtw import soft_dtw, soft_dtw_batch

from .conftest import assert_equal
from .reference import brute_force_soft_dtw, reference_cost, reference_in_band, reference_soft_dtw

GAMMAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# (n, m, bandwidth): square and rectangular, with and without the band.
SHAPES = [(5, 7, -1), (8, 6, 3), (4, 8, 0), (8, 4, 1), (7, 4, 2), (3, 10, 1), (1, 5, -1), (5, 1, 0), (1, 1, -1)]
# Largest distances.size(-1) fitting three diagonals of the accumulation dtype in 48 KiB of shared memory.
CAPACITY = {torch.float64: 2048, torch.float32: 4096, torch.float16: 4096, torch.bfloat16: 4096}
HALF_DTYPES = [torch.float16, torch.bfloat16]


def gradcheck(fn: object, *inputs: torch.Tensor) -> None:
    """Gradcheck with tolerances suited to the log-space backward."""
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)


def cpu_and_cuda(*tensors: torch.Tensor | None) -> tuple[list, list]:
    """Detached copies of ``tensors`` on CPU and CUDA, floating ones requiring grad."""

    def copy(t: torch.Tensor | None, device: str) -> torch.Tensor | None:
        if t is None:
            return None
        t = t.detach().to(device)
        return t.clone().requires_grad_() if t.is_floating_point() else t

    return [copy(t, "cpu") for t in tensors], [copy(t, "cuda") for t in tensors]


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize(("n", "m", "bandwidth"), SHAPES)
def test_soft_dtw_matches_reference(n: int, m: int, bandwidth: int, gamma: float) -> None:
    """Costs and gradients match the cell-by-cell recursion, differentiated by autograd."""
    d = torch.rand(2, n, m, dtype=torch.float64, generator=torch.Generator().manual_seed(n * m), requires_grad=True)
    d_ref = d.detach().clone().requires_grad_()
    cost = soft_dtw(d, gamma=gamma, bandwidth=bandwidth)
    expected = torch.stack([reference_soft_dtw(d_ref[b], gamma, bandwidth) for b in range(2)])
    torch.testing.assert_close(cost, expected)
    cost.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(d.grad, d_ref.grad)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize(("n", "m"), [(3, 4), (4, 4), (2, 5), (5, 2), (1, 3)])
def test_soft_dtw_matches_brute_force(n: int, m: int, gamma: float) -> None:
    """Costs match soft-DTW's definition, the soft minimum over every alignment path."""
    d = torch.rand(1, n, m, dtype=torch.float64, generator=torch.Generator().manual_seed(n + m))
    expected = torch.tensor([brute_force_soft_dtw(d[0], gamma)], dtype=torch.float64)
    torch.testing.assert_close(soft_dtw(d, gamma=gamma), expected)


def test_soft_dtw_single_path() -> None:
    """With a single alignment (one row, one column, or a zero band), the cost is the sum along it, exactly."""
    row = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)
    assert_equal(soft_dtw(row), torch.tensor([6.0], dtype=torch.float64))
    assert_equal(soft_dtw(row.transpose(1, 2)), torch.tensor([6.0], dtype=torch.float64))
    square = torch.rand(1, 5, 5, dtype=torch.float64)
    torch.testing.assert_close(soft_dtw(square, bandwidth=0), square[0].diagonal().sum().reshape(1))


def test_band_square_is_diagonal_band() -> None:
    """On square inputs, the band is |i - j| <= bandwidth."""
    for bandwidth in range(6):
        for i in range(6):
            for j in range(6):
                assert reference_in_band(i, j, 6, 6, bandwidth) == (abs(i - j) <= bandwidth)


def test_band_constrains() -> None:
    """Restricting the alignments can only increase the cost."""
    d = torch.rand(1, 6, 15, dtype=torch.float64)
    assert soft_dtw(d, gamma=0.5, bandwidth=1) >= soft_dtw(d, gamma=0.5)


def test_small_gamma_approaches_dtw() -> None:
    """As gamma tends to 0, soft-DTW tends to the unnormalized DTW cost."""
    d = torch.rand(1, 6, 8, dtype=torch.float64) * 10
    expected = reference_cost(d[0].numpy())[-1, -1]
    assert abs(soft_dtw(d, gamma=1e-3).item() - expected) < 0.1  # noqa: PLR2004


def test_soft_dtw_transpose_invariant() -> None:
    """Swapping the two sequences gives the same cost."""
    d = torch.rand(2, 5, 8, dtype=torch.float64)
    torch.testing.assert_close(soft_dtw(d, gamma=0.5), soft_dtw(d.transpose(1, 2), gamma=0.5))


@pytest.mark.parametrize("bandwidth", [-1, 2])
def test_padding_ignored(bandwidth: int) -> None:
    """Only the (sx[b], sy[b]) sub-block is read, and the padding gets no gradient."""
    d = torch.rand(3, 10, 10, dtype=torch.float64, requires_grad=True)
    sx, sy = torch.tensor([6, 10, 3]), torch.tensor([7, 6, 9])
    cost = soft_dtw(d, sx, sy, gamma=0.5, bandwidth=bandwidth)
    cost.sum().backward()
    for b in range(3):
        sub = d[b, : sx[b], : sy[b]].detach().unsqueeze(0).requires_grad_()
        cost_b = soft_dtw(sub, gamma=0.5, bandwidth=bandwidth)
        assert_equal(cost[b : b + 1], cost_b)
        cost_b.backward()
        assert_equal(d.grad[b, : sx[b], : sy[b]], sub.grad[0])
        assert (d.grad[b, sx[b] :] == 0).all()
        assert (d.grad[b, :, sy[b] :] == 0).all()


def test_zero_lengths() -> None:
    """Pairs with an empty sequence have a zero cost and no gradient."""
    d = torch.rand(3, 4, 5, dtype=torch.float64, requires_grad=True)
    cost = soft_dtw(d, torch.tensor([0, 4, 2]), torch.tensor([5, 5, 0]))
    assert cost[0] == 0
    assert cost[2] == 0
    cost.sum().backward()
    assert (d.grad[0] == 0).all()
    assert (d.grad[2] == 0).all()


def test_soft_dtw_batch_matches_paired() -> None:
    """All pairs match the paired soft_dtw of each pair, in value and gradient."""
    d = torch.rand(3, 4, 6, 5, dtype=torch.float64, requires_grad=True)
    sx, sy = torch.tensor([6, 3, 4]), torch.tensor([5, 2, 4, 1])
    cost = soft_dtw_batch(d, sx, sy, gamma=0.7, bandwidth=2)
    weights = torch.rand_like(cost)
    (cost * weights).sum().backward()
    for i in range(3):
        for j in range(4):
            pair = d[i, j].detach().unsqueeze(0).requires_grad_()
            cost_ij = soft_dtw(pair, sx[i : i + 1], sy[j : j + 1], gamma=0.7, bandwidth=2)
            assert_equal(cost[i, j], cost_ij[0])
            (cost_ij * weights[i, j]).sum().backward()
            assert_equal(d.grad[i, j], pair.grad[0])


def test_gradcheck() -> None:
    """Analytical gradients match finite differences, with padding, band and all pairs."""
    d = torch.rand(2, 6, 8, dtype=torch.float64, requires_grad=True)
    sx, sy = torch.tensor([4, 6]), torch.tensor([5, 8])
    gradcheck(lambda t: soft_dtw(t, sx, sy, gamma=1.0, bandwidth=2), d)
    gradcheck(lambda t: soft_dtw(t, gamma=0.01), torch.rand(1, 6, 6, dtype=torch.float64, requires_grad=True))
    d4 = torch.rand(2, 3, 4, 9, dtype=torch.float64, requires_grad=True)
    gradcheck(lambda t: soft_dtw_batch(t, torch.tensor([4, 3]), torch.tensor([9, 7, 2]), gamma=0.5, bandwidth=1), d4)


def test_small_gamma_gradients_finite() -> None:
    """The log-space backward stays finite with a small gamma on larger inputs."""
    d = torch.rand(2, 64, 64, dtype=torch.float64, requires_grad=True)
    soft_dtw(d, gamma=1e-3).sum().backward()
    assert torch.isfinite(d.grad).all()
    assert (d.grad != 0).any()


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_gradcheck_through_distances(p: float) -> None:
    """Gradients flow from the sequences through the distances between their frames."""
    x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 7, 4, dtype=torch.float64)
    sx, sy = torch.tensor([5, 3]), torch.tensor([6, 7])
    gradcheck(lambda t: soft_dtw(torch.cdist(t, y, p=p) ** p, sx, sy, gamma=0.5, bandwidth=2), x)


@pytest.mark.parametrize("dtype", [torch.float32, *HALF_DTYPES])
def test_reduced_precision(dtype: torch.dtype) -> None:
    """Reduced-precision inputs accumulate in float and stay close to float64, in their own dtype."""
    d = torch.rand(2, 5, 7, dtype=torch.float64)
    d_low = d.to(dtype).requires_grad_()
    cost = soft_dtw(d_low)
    assert cost.dtype == dtype
    tolerance = 1e-5 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(cost.double(), soft_dtw(d_low.detach().double()), atol=tolerance, rtol=tolerance)
    cost.sum().backward()
    assert d_low.grad.dtype == dtype
    assert torch.isfinite(d_low.grad).all()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32])
def test_length_dtypes(dtype: torch.dtype) -> None:
    """Lengths of any integer dtype give the same costs as int64 lengths."""
    d = torch.rand(3, 6, 7, dtype=torch.float64)
    sx, sy = torch.tensor([6, 2, 4]), torch.tensor([7, 7, 1])
    assert_equal(soft_dtw(d, sx.to(dtype), sy.to(dtype)), soft_dtw(d, sx, sy))
    assert_equal(soft_dtw(d, sx.to(dtype)), soft_dtw(d, sx, torch.full((3,), 7)))


def test_noncontiguous() -> None:
    """Strided distances and lengths give the same costs and gradients as contiguous ones."""
    base = torch.rand(6, 10, 12, dtype=torch.float64)
    strided = base[::2, ::2, ::3].requires_grad_()
    contiguous = strided.detach().contiguous().requires_grad_()
    lengths = torch.tensor([5, 0, 3, 0, 4, 0])[::2]
    assert not strided.is_contiguous()
    assert not lengths.is_contiguous()
    cost = soft_dtw(strided, lengths, gamma=0.5)
    expected = soft_dtw(contiguous, lengths.contiguous(), gamma=0.5)
    assert_equal(cost, expected)
    cost.sum().backward()
    expected.sum().backward()
    assert_equal(strided.grad, contiguous.grad)


def test_r_kept_only_for_backward() -> None:
    """The accumulated cost matrix R is only returned, and allocated, when keep_R is set."""
    d = torch.rand(2, 3, 4)
    _, R = torch.ops.torchdtw.soft_dtw(d, None, None, 1.0, -1, False)  # noqa: FBT003
    assert R.shape == (0,)
    _, R = torch.ops.torchdtw.soft_dtw(d.half(), None, None, 1.0, -1, True)  # noqa: FBT003
    assert R.shape == d.shape
    assert R.dtype == torch.float32
    with pytest.raises(RuntimeError, match="keep_R=True"):
        torch.ops.torchdtw.soft_dtw_backward(torch.ones(2), d, torch.empty(0), None, None, 1.0)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((torch.rand(5, 5), None, None, 1.0), "3D"),
        ((torch.rand(1, 5, 5), None, None, 0.0), "gamma"),
        ((torch.randint(0, 9, (1, 5, 5)), None, None, 1.0), "float"),
        ((torch.rand(1, 5, 5), torch.tensor([[5]]), None, 1.0), "1D"),
        ((torch.rand(2, 5, 5), torch.tensor([5, 5, 5]), None, 1.0), "sizes must match"),
        ((torch.rand(1, 5, 5), torch.tensor([5.0]), None, 1.0), "integer dtype"),
        ((torch.rand(1, 5, 5), torch.tensor([5]), torch.tensor([5], dtype=torch.int32), 1.0), "dtypes do not match"),
        ((torch.rand(1, 5, 5), torch.tensor([6]), None, 1.0), "sx values must not exceed"),
        ((torch.rand(2, 3, 5, 5), None, torch.tensor([5, 5, 6]), 1.0), "sy values must not exceed"),
    ],
)
def test_validation(args: tuple, match: str) -> None:
    """Invalid inputs are rejected with an explicit message."""
    with pytest.raises(RuntimeError, match=match):
        torch.ops.torchdtw.soft_dtw(*args, -1, False)  # noqa: FBT003


def test_python_rank_checks() -> None:
    """soft_dtw takes 3D distances and soft_dtw_batch 4D ones."""
    with pytest.raises(RuntimeError, match="3D"):
        soft_dtw(torch.rand(1, 1, 2, 2))
    with pytest.raises(RuntimeError, match="4D"):
        soft_dtw_batch(torch.rand(1, 2, 2))


@pytest.mark.requires_gpu
@settings(max_examples=30)
@given(
    b=st.integers(1, 3),
    n=st.integers(1, 300),
    m=st.integers(1, 300),
    bandwidth=st.integers(-1, 8),
    gamma=st.sampled_from([0.01, 0.1, 1.0]),
    seed=st.integers(0, 2**16),
)
def test_cpu_cuda_parity(b: int, n: int, m: int, bandwidth: int, gamma: float, seed: int) -> None:
    """CUDA matches CPU on costs and gradients, with padding and band."""
    g = torch.Generator().manual_seed(seed)
    d = torch.rand(b, n, m, dtype=torch.float64, generator=g)
    sx = torch.randint(0, n + 1, (b,), generator=g, dtype=torch.int32)
    sy = torch.randint(1, m + 1, (b,), generator=g, dtype=torch.int32)
    (d_cpu, sx_cpu, sy_cpu), (d_cuda, sx_cuda, sy_cuda) = cpu_and_cuda(d, sx, sy)
    cost_cpu = soft_dtw(d_cpu, sx_cpu, sy_cpu, gamma=gamma, bandwidth=bandwidth)
    cost_cuda = soft_dtw(d_cuda, sx_cuda, sy_cuda, gamma=gamma, bandwidth=bandwidth)
    torch.testing.assert_close(cost_cuda.cpu(), cost_cpu, rtol=1e-6, atol=1e-8)
    weights = torch.rand(b, dtype=torch.float64, generator=g)
    (cost_cpu * weights).sum().backward()
    (cost_cuda * weights.cuda()).sum().backward()
    torch.testing.assert_close(d_cuda.grad.cpu(), d_cpu.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.requires_gpu
def test_cpu_cuda_parity_batch() -> None:
    """CUDA matches CPU on all pairs, with default lengths and non-contiguous distances."""
    d = torch.rand(5, 3, 40, 33, dtype=torch.float64).transpose(2, 3).contiguous().transpose(2, 3)
    for sx, sy in [(None, None), (torch.randint(0, 41, (5,)), torch.randint(1, 34, (3,)))]:
        (d_cpu, sx_cpu, sy_cpu), (d_cuda, sx_cuda, sy_cuda) = cpu_and_cuda(d, sx, sy)
        cost_cpu = soft_dtw_batch(d_cpu, sx_cpu, sy_cpu, gamma=0.3, bandwidth=4)
        cost_cuda = soft_dtw_batch(d_cuda, sx_cuda, sy_cuda, gamma=0.3, bandwidth=4)
        torch.testing.assert_close(cost_cuda.cpu(), cost_cpu, rtol=1e-6, atol=1e-8)
        cost_cpu.sum().backward()
        cost_cuda.sum().backward()
        torch.testing.assert_close(d_cuda.grad.cpu(), d_cpu.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, *HALF_DTYPES])
def test_cpu_cuda_parity_reduced_precision(dtype: torch.dtype) -> None:
    """Reduced-precision CUDA matches CPU up to the rounding of the output."""
    (d_cpu,), (d_cuda,) = cpu_and_cuda(torch.rand(3, 20, 30).to(dtype))
    cost_cpu, cost_cuda = soft_dtw(d_cpu), soft_dtw(d_cuda)
    torch.testing.assert_close(cost_cuda.cpu(), cost_cpu)
    cost_cpu.sum().backward()
    cost_cuda.sum().backward()
    torch.testing.assert_close(d_cuda.grad.cpu(), d_cpu.grad)


@pytest.mark.requires_gpu
@pytest.mark.parametrize(("dtype", "capacity"), list(CAPACITY.items()))
def test_shared_memory_boundary(dtype: torch.dtype, capacity: int) -> None:
    """distances.size(-1) == capacity runs on CUDA like on CPU, capacity + 1 is rejected."""
    (d_cpu,), (d_cuda,) = cpu_and_cuda(torch.rand(1, 3, capacity).to(dtype))
    cost_cpu, cost_cuda = soft_dtw(d_cpu, gamma=10.0), soft_dtw(d_cuda, gamma=10.0)
    torch.testing.assert_close(cost_cuda.cpu(), cost_cpu)
    cost_cpu.sum().backward()
    cost_cuda.sum().backward()
    torch.testing.assert_close(d_cuda.grad.cpu(), d_cpu.grad)
    with pytest.raises(RuntimeError, match="shared memory"):
        soft_dtw(torch.rand(1, 3, capacity + 1, device="cuda").to(dtype))


@pytest.mark.requires_gpu
def test_cuda_device_checks() -> None:
    """Lengths must be on the device of the distances."""
    with pytest.raises(RuntimeError, match="same device"):
        soft_dtw(torch.rand(1, 4, 4, device="cuda"), torch.tensor([4]))
