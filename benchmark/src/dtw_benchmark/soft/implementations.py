"""Uniform ``(x, y, sx, sy, gamma) -> costs`` wrappers around each soft-DTW implementation.

Only torchdtw's ``soft_dtw`` takes per-sequence lengths: padded frames never enter the alignment, however uneven the
lengths of a batch. The other implementations take rectangular (B, N, D) / (B, M, D) batches, so a single call would
treat the padding as real frames. To keep the comparison fair, they are called once per pair of sequences, each
sliced to its true lengths, whenever the lengths of the batch are not uniform.
"""

from collections.abc import Callable

import torch
from numba import cuda

from torchdtw import soft_dtw

from .soft_dtw_cuda import SoftDTW as MaghoumiSoftDTW

try:
    from softdtw_cuda import SoftDTW as SdtwCudaTorchSoftDTW
except ImportError:
    SdtwCudaTorchSoftDTW = None


def _batched_or_per_pair(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
) -> torch.Tensor:
    """Call ``fn`` on the whole batch if its lengths are uniform, else once per pair sliced to its lengths."""
    if bool((sx == x.size(1)).all()) and bool((sy == y.size(1)).all()):
        return fn(x, y)
    return torch.cat([fn(x[b : b + 1, : int(sx[b])], y[b : b + 1, : int(sy[b])]) for b in range(x.size(0))])


def torchdtw_soft_dtw(
    x: torch.Tensor, y: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Soft-DTW of torchdtw, on the real lengths."""
    return soft_dtw(torch.cdist(x, y) ** 2, sx, sy, gamma=gamma)


def maghoumi(x: torch.Tensor, y: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, gamma: float) -> torch.Tensor:
    """Soft-DTW of pytorch-softdtw-cuda (Maghoumi): Numba on CPU, Numba CUDA kernels on GPU."""
    return _batched_or_per_pair(MaghoumiSoftDTW(use_cuda=x.is_cuda, gamma=gamma), x, y, sx, sy)


def sdtw_cuda_torch(
    x: torch.Tensor, y: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Soft-DTW of sdtw-cuda-torch (BGU-CS-VIL), a C++/CUDA extension."""
    if SdtwCudaTorchSoftDTW is None:
        msg = "sdtw-cuda-torch is not installed"
        raise ImportError(msg)
    return _batched_or_per_pair(SdtwCudaTorchSoftDTW(gamma=gamma), x, y, sx, sy)


def available(device: torch.device) -> dict[str, Callable[..., torch.Tensor]]:
    """Return the implementations runnable on ``device``, by name, starting with torchdtw: the reference."""
    impls: dict[str, Callable[..., torch.Tensor]] = {"torchdtw": torchdtw_soft_dtw}
    if device.type == "cpu" or cuda.is_available():
        impls["pytorch-softdtw-cuda (Maghoumi)"] = maghoumi
    if SdtwCudaTorchSoftDTW is not None:
        impls["sdtw-cuda-torch"] = sdtw_cuda_torch
    return impls
