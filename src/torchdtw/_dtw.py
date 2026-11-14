"""DTW cost and path, with CPU and GPU backends."""

import torch

from ._windows import check_no_cuda_on_windows

__all__ = ["dtw", "dtw_batch", "dtw_path"]


def dtw(distances: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Compute the DTW cost of the given ``distances`` 2D tensor.

    Use ``+inf`` to mask forbidden alignments. NaN distances are unsupported: the result is
    unspecified and may differ between the CPU and CUDA backends.

    :param distances: A 2D tensor of shape (n, m) with the pairwise distances between two sequences.
    :param normalize: Whether to divide the cost by the length of the warping path.
    :returns: A scalar tensor with the cost.
    """
    check_no_cuda_on_windows(distances)
    if distances.requires_grad:
        distances = distances.detach()
    return torch.ops.torchdtw.dtw.default(distances, normalize)


def dtw_path(distances: torch.Tensor) -> torch.Tensor:
    """Compute the DTW path of the given ``distances`` 2D tensor.

    No CUDA variant or batched implementation are provided for now. Use ``+inf`` to mask forbidden alignments.
    NaN distances are unsupported and give an unspecified path.

    :param distances: A 2D tensor of shape (n, m) with the pairwise distances between two sequences.
    :returns: A 2D tensor of shape (*, 2) with the indices of the path, from (0, 0) to (n - 1, m - 1).
    """
    return torch.ops.torchdtw.dtw_path.default(distances.cpu()).to(distances.device)


def dtw_batch(
    distances: torch.Tensor,
    sx: torch.Tensor | None = None,
    sy: torch.Tensor | None = None,
    *,
    symmetric: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute the DTW cost between all pairs of sequences of two batches, on the ``distances`` 4D tensor.

    Only the ``(sx[i], sy[j])`` sub-block of each pair is read, so the padding beyond the sequence lengths is ignored.
    Use ``+inf`` to mask forbidden alignments. NaN distances are unsupported: the result is unspecified and may
    differ between the CPU and CUDA backends.

    :param distances: A 4D tensor of shape (n1, n2, s1, s2) with the pairwise distances between the frames of two
        batches of sequences.
    :param sx: An optional 1D integer tensor of shape (n1,) with the lengths of the first batch, at most s1.
        Defaults to s1.
    :param sy: An optional 1D integer tensor of shape (n2,) with the lengths of the second batch, at most s2.
        Defaults to s2.
    :param symmetric: Whether the two batches are the same: only the pairs ``i < j`` are computed, and mirrored,
        and the diagonal is 0. Requires ``n1 == n2``.
    :param normalize: Whether to divide each cost by the length of its warping path.
    :returns: A 2D tensor of shape (n1, n2) with the costs.
    """
    check_no_cuda_on_windows(distances)
    if distances.requires_grad:
        distances = distances.detach()
    if sx is not None and sy is not None:
        return torch.ops.torchdtw.dtw_batch.default(distances, sx, sy, symmetric, normalize)
    return torch.ops.torchdtw.dtw_batch.optional_lengths(distances, sx, sy, symmetric, normalize)


_DISTANCES_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_LENGTHS_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)


def check_lengths_tensors(
    distances: torch.Tensor, sx: torch.Tensor | None, sy: torch.Tensor | None, nx: int, ny: int
) -> None:
    """Check ``sx`` and ``sy`` in a fake kernel, as ``check_lengths_tensors`` in csrc/common.h."""
    for lengths, n in ((sx, nx), (sy, ny)):
        if lengths is not None:
            torch._check(lengths.ndim == 1, lambda: "sx and sy must be 1D tensors")
            torch._check(lengths.shape[0] == n, lambda: "sx and sy sizes must match the batch dimensions of distances")
            torch._check(lengths.dtype in _LENGTHS_DTYPES, lambda: "sx and sy must have an integer dtype")
            torch._check(
                lengths.device == distances.device, lambda: "sx and sy must be on the same device as distances"
            )
    if sx is not None and sy is not None:
        torch._check(sx.dtype == sy.dtype, lambda: "sx and sy dtypes do not match")


@torch.library.register_fake("torchdtw::dtw")
def _(distances: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw, for compatibility with torch.compile."""
    torch._check(distances.ndim == 2, lambda: "distances must be a 2D tensor")
    torch._check(distances.dtype in _DISTANCES_DTYPES, lambda: f"dtw not implemented for {distances.dtype}")
    torch._check(distances.shape[0] > 0, lambda: "Empty input tensor")
    torch._check(distances.shape[1] > 0, lambda: "Empty input tensor")
    return torch.empty((), dtype=distances.dtype, layout=distances.layout, device=distances.device)


@torch.library.register_fake("torchdtw::dtw_path")
def _(distances: torch.Tensor) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw_path, for compatibility with torch.compile."""
    torch._check(distances.ndim == 2, lambda: "distances must be a 2D tensor")
    torch._check(distances.dtype in _DISTANCES_DTYPES, lambda: f"dtw not implemented for {distances.dtype}")
    n, m = distances.shape
    torch._check(n > 0 and m > 0, lambda: "Empty input tensor")
    ctx = torch.library.get_ctx()
    if isinstance(n, int) and isinstance(m, int):
        path_len = ctx.new_dynamic_size(min=max(n, m), max=n + m - 1)
    else:  # new_dynamic_size only takes static bounds
        path_len = ctx.new_dynamic_size()
        torch._check(path_len >= n)
        torch._check(path_len >= m)
        torch._check(path_len <= n + m - 1)
    return distances.new_empty((path_len, 2), dtype=torch.long)


@torch.library.register_fake("torchdtw::dtw_batch")
@torch.library.register_fake("torchdtw::dtw_batch.optional_lengths")
def _(
    distances: torch.Tensor,
    sx: torch.Tensor | None,
    sy: torch.Tensor | None,
    symmetric: bool,
    normalize: bool = True,
) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw_batch, for compatibility with torch.compile."""
    torch._check(distances.ndim == 4, lambda: "distances must be a 4D tensor")
    torch._check(distances.dtype in _DISTANCES_DTYPES, lambda: f"dtw not implemented for {distances.dtype}")
    nx, ny, _, _ = distances.shape
    check_lengths_tensors(distances, sx, sy, nx, ny)
    torch._check(isinstance(symmetric, bool))
    torch._check(
        not symmetric or nx == ny, lambda: "symmetric dtw_batch requires distances.size(0) == distances.size(1)"
    )
    return torch.empty((nx, ny), dtype=distances.dtype, layout=distances.layout, device=distances.device)
