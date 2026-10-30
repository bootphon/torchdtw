"""Soft-DTW (Cuturi & Blondel, 2017): a differentiable DTW, with CPU and GPU backends."""

import torch

from ._windows import check_no_cuda_on_windows

__all__ = ["soft_dtw", "soft_dtw_batch"]


class RankError(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"distances must be a {expected}D tensor, got {actual}D")


def _soft_dtw(
    distances: torch.Tensor,
    ndim: int,
    sx: torch.Tensor | None,
    sy: torch.Tensor | None,
    *,
    gamma: float,
    bandwidth: int,
) -> torch.Tensor:
    if distances.ndim != ndim:
        raise RankError(ndim, distances.ndim)
    check_no_cuda_on_windows(distances)
    # Keep the accumulated cost matrix R only if the backward pass, its only reader, can run.
    keep_r = torch.is_grad_enabled() and distances.requires_grad
    cost, _ = torch.ops.torchdtw.soft_dtw.default(distances, sx, sy, gamma, bandwidth, keep_r)
    return cost


def soft_dtw(
    distances: torch.Tensor,
    sx: torch.Tensor | None = None,
    sy: torch.Tensor | None = None,
    *,
    gamma: float = 1.0,
    bandwidth: int = -1,
) -> torch.Tensor:
    """Compute the soft-DTW cost of each pair of sequences in the ``distances`` 3D tensor.

    Soft-DTW replaces the minimum over alignments of DTW by the soft minimum
    ``-gamma * log(sum(exp(-cost / gamma)))``, so it is differentiable with respect to ``distances``. Unlike
    ``dtw``, the cost is not normalized by the path length. Only the ``(sx[b], sy[b])`` sub-block of each pair is
    read, and the padding beyond it gets no gradient. Every ``sx[b]`` must be ``<= N`` and every ``sy[b] <= M``:
    the CPU backend validates this, but the CUDA backend assumes it and reads out of bounds if violated.

    :param distances: A 3D tensor of shape (B, N, M) with the pairwise distances between the frames of B pairs of
        sequences, e.g. ``torch.cdist(x, y) ** 2`` for sequences x of shape (B, N, D) and y of shape (B, M, D).
        Must have a floating dtype.
    :param sx: An optional 1D integer tensor of shape (B,) with the lengths of the first sequences. Defaults to N.
    :param sy: An optional 1D integer tensor of shape (B,) with the lengths of the second sequences. Defaults to M.
    :param gamma: The smoothing parameter, positive. Soft-DTW tends to the unnormalized DTW cost as it tends to 0.
    :param bandwidth: Width of the Sakoe-Chiba band around the diagonal joining the first and last frames of each
        pair, in steps along the longer sequence. It is floored at half the slope of that diagonal so that an
        alignment always exists. Negative values disable the band.
    :returns: A 1D tensor of shape (B,) with the costs.
    """
    return _soft_dtw(distances, 3, sx, sy, gamma=gamma, bandwidth=bandwidth)


def soft_dtw_batch(
    distances: torch.Tensor,
    sx: torch.Tensor | None = None,
    sy: torch.Tensor | None = None,
    *,
    gamma: float = 1.0,
    bandwidth: int = -1,
) -> torch.Tensor:
    """Compute the soft-DTW cost between all pairs of sequences of two batches, on the ``distances`` 4D tensor.

    Same as ``soft_dtw``, for every pair ``(i, j)`` instead of B independent pairs. Only the ``(sx[i], sy[j])``
    sub-block of each pair is read, and the padding beyond it gets no gradient. Every ``sx[i]`` must be ``<= s1``
    and every ``sy[j] <= s2``: the CPU backend validates this, but the CUDA backend assumes it.

    :param distances: A 4D tensor of shape (n1, n2, s1, s2) with the pairwise distances between the frames of two
        batches of sequences. Must have a floating dtype.
    :param sx: An optional 1D integer tensor of shape (n1,) with the lengths of the first batch. Defaults to s1.
    :param sy: An optional 1D integer tensor of shape (n2,) with the lengths of the second batch. Defaults to s2.
    :param gamma: The smoothing parameter, positive.
    :param bandwidth: Width of the Sakoe-Chiba band, as in ``soft_dtw``. Negative values disable the band.
    :returns: A 2D tensor of shape (n1, n2) with the costs.
    """
    return _soft_dtw(distances, 4, sx, sy, gamma=gamma, bandwidth=bandwidth)


@torch.library.register_fake("torchdtw::soft_dtw")
def _(
    distances: torch.Tensor,
    sx: torch.Tensor | None,  # noqa: ARG001
    sy: torch.Tensor | None,  # noqa: ARG001
    gamma: float,  # noqa: ARG001
    bandwidth: int,  # noqa: ARG001
    keep_R: bool,  # noqa: FBT001
) -> tuple[torch.Tensor, torch.Tensor]:
    """Register the FakeTensor kernel for soft_dtw, for compatibility with torch.compile."""
    torch._check(distances.ndim in {3, 4})
    acc_dtype = torch.float32 if distances.dtype in {torch.float16, torch.bfloat16} else distances.dtype
    cost = distances.new_empty(distances.shape[: distances.ndim - 2])
    R = distances.new_empty(distances.shape if keep_R else (0,), dtype=acc_dtype)
    return cost, R


@torch.library.register_fake("torchdtw::soft_dtw_backward")
def _(
    grad: torch.Tensor,  # noqa: ARG001
    distances: torch.Tensor,
    R: torch.Tensor,  # noqa: ARG001
    sx: torch.Tensor | None,  # noqa: ARG001
    sy: torch.Tensor | None,  # noqa: ARG001
    gamma: float,  # noqa: ARG001
) -> torch.Tensor:
    """Register the FakeTensor kernel for soft_dtw_backward, for compatibility with torch.compile."""
    return distances.new_empty(distances.shape)


def _setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: tuple) -> None:
    distances, sx, sy, gamma, _, _ = inputs
    _, R = output
    ctx.mark_non_differentiable(R)
    # R has no gradient: do not let autograd allocate and fill one with zeros.
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(distances, R, sx, sy)
    ctx.gamma = gamma


def _backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor | None, _: torch.Tensor | None) -> tuple:
    if grad is None:
        return (None,) * 6
    distances, R, sx, sy = ctx.saved_tensors
    grad_distances = torch.ops.torchdtw.soft_dtw_backward.default(grad, distances, R, sx, sy, ctx.gamma)
    return grad_distances, *(None,) * 5


torch.library.register_autograd("torchdtw::soft_dtw", _backward, setup_context=_setup_context)
