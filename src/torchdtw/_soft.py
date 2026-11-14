"""Soft-DTW (Cuturi & Blondel, 2017): a differentiable DTW, with CPU and GPU backends."""

import torch

from ._windows import check_no_cuda_on_windows

__all__ = ["soft_dtw", "soft_dtw_batch"]


class RankError(RuntimeError):
    def __init__(self, expected: str, actual: int) -> None:
        super().__init__(f"distances must be a {expected} tensor, got {actual}D")


class SinglePairLengthsError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("sx and sy are only accepted with 3D distances: a 2D tensor is a single pair, read whole")


def _soft_dtw(
    distances: torch.Tensor,
    ndim: int,
    sx: torch.Tensor | None,
    sy: torch.Tensor | None,
    *,
    gamma: float,
    bandwidth: int,
    symmetric: bool,
    normalize: bool,
) -> torch.Tensor:
    if distances.ndim != ndim:
        raise RankError(f"{ndim}D", distances.ndim)
    check_no_cuda_on_windows(distances)
    if torch.is_grad_enabled() and distances.requires_grad:
        return _SoftDTW.apply(distances, sx, sy, gamma, bandwidth, symmetric, normalize)
    cost, _ = torch.ops.torchdtw.soft_dtw.default(
        distances,
        sx,
        sy,
        gamma,
        bandwidth,
        symmetric,
        keep_R=False,
        normalize=normalize,
    )
    return cost


def soft_dtw(
    distances: torch.Tensor,
    sx: torch.Tensor | None = None,
    sy: torch.Tensor | None = None,
    *,
    gamma: float = 1.0,
    bandwidth: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute the soft-DTW cost of a pair of sequences, or of each pair in a batch of independent pairs.

    Soft-DTW replaces the minimum over alignments of DTW by the soft minimum
    ``-gamma * log(sum(exp(-cost / gamma)))``, so it is differentiable with respect to ``distances``. Unlike
    ``dtw``, the cost is not normalized by the path length. With 3D ``distances``, only the ``(sx[b], sy[b])``
    sub-block of each pair is read, and the padding beyond it gets no gradient. Every ``sx[b]`` must be ``<= N``
    and every ``sy[b] <= M``.

    :param distances: A 2D tensor of shape (N, M) with the pairwise distances between the frames of two sequences,
        or a 3D tensor of shape (B, N, M) for B independent pairs, e.g. ``torch.cdist(x, y) ** 2`` for sequences x
        of shape (B, N, D) and y of shape (B, M, D). Must have a floating dtype.
    :param sx: An optional 1D integer tensor of shape (B,) with the lengths of the first sequences. Defaults to N.
        Only with 3D ``distances``.
    :param sy: An optional 1D integer tensor of shape (B,) with the lengths of the second sequences. Defaults to M.
        Only with 3D ``distances``.
    :param gamma: The smoothing parameter, positive. Soft-DTW tends to the unnormalized DTW cost as it tends to 0.
    :param bandwidth: Width of the Sakoe-Chiba band around the diagonal joining the first and last frames of each
        pair, in steps along the longer sequence. It is floored at half the slope of that diagonal so that an
        alignment always exists. Negative values disable the band.
    :param normalize: Whether to divide the cost of each pair by ``sx[b] + sy[b] - 1``, the length of its longest
        alignment path.
    :returns: A scalar tensor with the cost for 2D ``distances``, a 1D tensor of shape (B,) with the costs for 3D
        ones.
    """
    if distances.ndim == 2:
        if sx is not None or sy is not None:
            raise SinglePairLengthsError
        return _soft_dtw(
            distances.unsqueeze(0),
            3,
            None,
            None,
            gamma=gamma,
            bandwidth=bandwidth,
            symmetric=False,
            normalize=normalize,
        )[0]
    if distances.ndim != 3:
        raise RankError("2D or 3D", distances.ndim)
    return _soft_dtw(distances, 3, sx, sy, gamma=gamma, bandwidth=bandwidth, symmetric=False, normalize=normalize)


def soft_dtw_batch(
    distances: torch.Tensor,
    sx: torch.Tensor | None = None,
    sy: torch.Tensor | None = None,
    *,
    gamma: float = 1.0,
    bandwidth: int = -1,
    symmetric: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute the soft-DTW cost between all pairs of sequences of two batches, on the ``distances`` 4D tensor.

    Same as ``soft_dtw``, for every pair ``(i, j)`` instead of B independent pairs. Only the ``(sx[i], sy[j])``
    sub-block of each pair is read, and the padding beyond it gets no gradient. Every ``sx[i]`` must be ``<= s1``
    and every ``sy[j] <= s2``.

    :param distances: A 4D tensor of shape (n1, n2, s1, s2) with the pairwise distances between the frames of two
        batches of sequences. Must have a floating dtype.
    :param sx: An optional 1D integer tensor of shape (n1,) with the lengths of the first batch. Defaults to s1.
    :param sy: An optional 1D integer tensor of shape (n2,) with the lengths of the second batch. Defaults to s2.
    :param gamma: The smoothing parameter, positive.
    :param bandwidth: Width of the Sakoe-Chiba band, as in ``soft_dtw``. Negative values disable the band.
    :param symmetric: Whether the two batches are the same: only the pairs ``i <= j`` are computed, from the blocks
        ``distances[i, j]``, and mirrored. The blocks ``i > j`` are not read and get no gradient. Unlike in
        ``dtw_batch``, the diagonal is computed: the soft-DTW cost of a sequence with itself is not 0. Requires
        ``n1 == n2``.
    :param normalize: Whether to divide the cost of each pair by ``sx[i] + sy[j] - 1``, as in ``soft_dtw``.
    :returns: A 2D tensor of shape (n1, n2) with the costs.
    """
    return _soft_dtw(distances, 4, sx, sy, gamma=gamma, bandwidth=bandwidth, symmetric=symmetric, normalize=normalize)


@torch.library.register_fake("torchdtw::soft_dtw")
def _(
    distances: torch.Tensor,
    sx: torch.Tensor | None,
    sy: torch.Tensor | None,
    gamma: float,
    bandwidth: int,
    symmetric: bool,
    keep_R: bool,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Register the FakeTensor kernel for soft_dtw, for compatibility with torch.compile."""
    torch._check(distances.ndim in {3, 4})
    acc_dtype = torch.float32 if distances.dtype in {torch.float16, torch.bfloat16} else distances.dtype
    cost = distances.new_empty(distances.shape[: distances.ndim - 2])
    R = distances.new_empty(distances.shape if keep_R else (0,), dtype=acc_dtype)
    return cost, R


@torch.library.register_fake("torchdtw::soft_dtw_backward")
def _(
    grad: torch.Tensor,
    distances: torch.Tensor,
    R: torch.Tensor,
    sx: torch.Tensor | None,
    sy: torch.Tensor | None,
    gamma: float,
    symmetric: bool,
    normalize: bool = False,
) -> torch.Tensor:
    """Register the FakeTensor kernel for soft_dtw_backward, for compatibility with torch.compile."""
    return distances.new_empty(distances.shape)


class _SoftDTW(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        distances: torch.Tensor,
        sx: torch.Tensor | None,
        sy: torch.Tensor | None,
        gamma: float,
        bandwidth: int,
        symmetric: bool,
        normalize: bool,
    ) -> torch.Tensor:
        cost, R = torch.ops.torchdtw.soft_dtw.default(
            distances,
            sx,
            sy,
            gamma,
            bandwidth,
            symmetric,
            keep_R=True,
            normalize=normalize,
        )
        ctx.save_for_backward(distances, R, sx, sy)
        ctx.gamma = gamma
        ctx.symmetric = symmetric
        ctx.normalize = normalize
        return cost

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> tuple:
        distances, R, sx, sy = ctx.saved_tensors
        grad_distances = torch.ops.torchdtw.soft_dtw_backward.default(
            grad,
            distances,
            R,
            sx,
            sy,
            ctx.gamma,
            ctx.symmetric,
            ctx.normalize,
        )
        return grad_distances, None, None, None, None, None, None
