"""DTW implementation using PyTorch C++ extensions, with CPU and CUDA backends."""

import torch

from . import _C  # noqa: F401 # ty: ignore[unresolved-import]

__all__ = ["dtw", "dtw_batch"]


def dtw(distances: torch.Tensor) -> torch.Tensor:
    """Compute the DTW cost of the given ``distances`` 2D tensor.

    :param distances: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
    :returns: A scalar tensor with the cost.
    """
    return torch.ops.torchdtw.dtw.default(distances)


def dtw_path(distances: torch.Tensor) -> torch.Tensor:
    """Compute the DTW path of the given ``distances`` 2D tensor.

    No CUDA variant or batched implementation are provided for now.
    :param distances: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
    :returns: A 2D tensor of shape (*, 2) with the path indices.
    """
    return torch.ops.torchdtw.dtw_path.default(distances.cpu()).to(distances.device)


def dtw_batch(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
    """Compute the batched DTW cost on the ``distances`` 4D tensor.

    :param distances: A 4D tensor of shape (n1, n2, s1, s2) representing the pairwise distances between two
        batches of sequences.
    :param sx: A 1D tensor of shape (n1,) representing the lengths of the sequences in the first batch.
    :param sy: A 1D tensor of shape (n2,) representing the lengths of the sequences in the second batch.
    :param symmetric: Whether or not the DTW is symmetric (i.e., the two batches are the same).
    :returns: A 2D tensor of shape (n1, n2) with the costs.
    """
    return torch.ops.torchdtw.dtw_batch.default(distances, sx, sy, symmetric)


@torch.library.register_fake("torchdtw::dtw")
def _(distances: torch.Tensor) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw, for compatibility with torch.compile."""
    torch._check(distances.ndim == 2)
    torch._check(distances.dtype == torch.float32)
    return torch.empty((), dtype=torch.float32, layout=distances.layout, device=distances.device)


@torch.library.register_fake("torchdtw::dtw_batch")
def _(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, symmetric: bool) -> torch.Tensor:  # noqa: FBT001
    """Register the FakeTensor kernel for dtw_batch, for compatibility with torch.compile."""
    torch._check(distances.ndim == 4)
    torch._check(sx.ndim == 1)
    torch._check(sy.ndim == 1)
    torch._check(distances.dtype == torch.float32)
    torch._check(sx.dtype == torch.long)
    torch._check(sy.dtype == torch.long)
    torch._check(isinstance(symmetric, bool))
    nx, ny, _, _ = distances.shape
    return torch.empty((nx, ny), dtype=torch.float32, layout=distances.layout, device=distances.device)
