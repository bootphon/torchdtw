"""DTW cost and path, with CPU and GPU backends."""

import torch

from ._windows import check_no_cuda_on_windows

__all__ = ["dtw", "dtw_batch", "dtw_path"]


def dtw(distances: torch.Tensor) -> torch.Tensor:
    """Compute the DTW cost of the given ``distances`` 2D tensor.

    Use ``+inf`` to mask forbidden alignments. NaN distances are unsupported: the result is
    unspecified and may differ between the CPU and CUDA backends. Integer ``distances`` accumulate
    the cost in their own dtype and may overflow on long sequences; use a wide enough integer dtype
    or a floating dtype.

    :param distances: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
    :returns: A scalar tensor with the cost.
    """
    check_no_cuda_on_windows(distances)
    return torch.ops.torchdtw.dtw.default(distances)


def dtw_path(distances: torch.Tensor) -> torch.Tensor:
    """Compute the DTW path of the given ``distances`` 2D tensor.

    No CUDA variant or batched implementation are provided for now.
    Use ``+inf`` to mask forbidden alignments. NaN distances are unsupported and give an
    unspecified path.
    :param distances: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
    :returns: A 2D tensor of shape (*, 2) with the path indices.
    """
    return torch.ops.torchdtw.dtw_path.default(distances.cpu()).to(distances.device)


def dtw_batch(
    distances: torch.Tensor,
    sx: torch.Tensor | None = None,
    sy: torch.Tensor | None = None,
    *,
    symmetric: bool = False,
) -> torch.Tensor:
    """Compute the batched DTW cost on the ``distances`` 4D tensor.

    Only the ``(sx[i], sy[j])`` sub-block of each pair is read, so padding beyond the sequence
    lengths is ignored. Every ``sx[i]`` must be ``<= s1`` and every ``sy[j] <= s2``: the CPU backend
    raises an error, and the CUDA backend a device-side assert, if violated. Use ``+inf``
    to mask forbidden alignments. NaN distances are unsupported: the result is unspecified and may
    differ between the CPU and CUDA backends. Integer ``distances`` accumulate the cost in their own
    dtype and may overflow on long sequences; use a wide enough integer dtype or a floating dtype.

    :param distances: A 4D tensor of shape (n1, n2, s1, s2) representing the pairwise distances between two
        batches of sequences.
    :param sx: An optional 1D integer tensor of shape (n1,) with the lengths of the sequences in the first batch.
        Defaults to s1.
    :param sy: An optional 1D integer tensor of shape (n2,) with the lengths of the sequences in the second batch.
        Defaults to s2.
    :param symmetric: Whether the two batches are the same: only the pairs ``i < j`` are computed, and mirrored,
        and the diagonal is 0. Requires ``n1 == n2``.
    :returns: A 2D tensor of shape (n1, n2) with the costs.
    """
    check_no_cuda_on_windows(distances)
    if sx is not None and sy is not None:
        return torch.ops.torchdtw.dtw_batch.default(distances, sx, sy, symmetric)
    return torch.ops.torchdtw.dtw_batch.optional_lengths(distances, sx, sy, symmetric)


@torch.library.register_fake("torchdtw::dtw")
def _(distances: torch.Tensor) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw, for compatibility with torch.compile."""
    torch._check(distances.ndim == 2)
    return torch.empty((), dtype=distances.dtype, layout=distances.layout, device=distances.device)


@torch.library.register_fake("torchdtw::dtw_batch")
@torch.library.register_fake("torchdtw::dtw_batch.optional_lengths")
def _(
    distances: torch.Tensor,
    sx: torch.Tensor | None,
    sy: torch.Tensor | None,
    symmetric: bool,
) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw_batch, for compatibility with torch.compile."""
    torch._check(distances.ndim == 4)
    for lengths in (sx, sy):
        if lengths is not None:
            torch._check(lengths.ndim == 1)
            torch._check(not (lengths.dtype.is_complex or lengths.dtype.is_floating_point))
    if sx is not None and sy is not None:
        torch._check(sx.dtype == sy.dtype)
    torch._check(isinstance(symmetric, bool))
    nx, ny, _, _ = distances.shape
    return torch.empty((nx, ny), dtype=distances.dtype, layout=distances.layout, device=distances.device)
