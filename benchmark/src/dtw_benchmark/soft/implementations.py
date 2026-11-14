"""The soft-DTW implementations compared against torchdtw, and ``available`` to pick those that run on a device.

Each one is wrapped as ``(x, y, sx, sy, gamma) -> costs``, with the squared Euclidean distance between frames, from
the sequences ``x`` of shape (B, N, D) and ``y`` of shape (B, M, D). Only torchdtw takes per-sequence lengths: padded
frames never enter its alignments. The other implementations take rectangular batches, so a single call would treat
the padding as real frames. To keep the comparison fair, they are called once per pair of sequences, each sliced to
its lengths, whenever the lengths of the batch are not uniform.
"""

import warnings
from collections.abc import Callable

import torch
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from softdtw_cuda import SoftDTW as SdtwCudaTorchSoftDTW
from tslearn.metrics import SoftDTWLossPyTorch

import torchdtw

from .soft_dtw_cuda import SoftDTW as MaghoumiSoftDTW

REFERENCE = "torchdtw"

# Small batches give small grids; the vendored code builds its scalars with the legacy torch.cuda.FloatTensor.
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", message=r"The torch\.cuda\.\*DtypeTensor constructors", category=UserWarning)


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
    """Soft-DTW of torchdtw, on the real lengths, from the distances of ``torch.cdist``."""
    return torchdtw.soft_dtw(torch.cdist(x, y) ** 2, sx, sy, gamma=gamma)


def maghoumi(x: torch.Tensor, y: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, gamma: float) -> torch.Tensor:
    """Soft-DTW of pytorch-softdtw-cuda (Maghoumi, vendored): Numba on CPU, Numba CUDA kernels on GPU."""
    return _batched_or_per_pair(MaghoumiSoftDTW(use_cuda=x.is_cuda, gamma=gamma), x, y, sx, sy)


def sdtw_cuda_torch(
    x: torch.Tensor, y: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Soft-DTW of sdtw-cuda-torch (BGU-CS-VIL): Numba on CPU, fused Numba CUDA kernels on GPU."""
    return _batched_or_per_pair(SdtwCudaTorchSoftDTW(gamma=gamma), x, y, sx, sy)


def tslearn(x: torch.Tensor, y: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, gamma: float) -> torch.Tensor:
    """Soft-DTW of tslearn's ``SoftDTWLossPyTorch``: Numba, on CPU."""
    return _batched_or_per_pair(SoftDTWLossPyTorch(gamma=gamma), x, y, sx, sy)


def available(device: torch.device) -> dict[str, Callable[..., torch.Tensor]]:
    """Return the implementations runnable on ``device``, by label, in display order, ending with torchdtw.

    torchdtw is the correctness reference, under ``REFERENCE``. On CUDA, the Numba implementations need a working
    Numba CUDA target (the numba-cuda package and a CUDA toolkit), and tslearn only runs on the CPU.
    """
    impls: dict[str, Callable[..., torch.Tensor]] = {}
    if device.type == "cpu" or cuda.is_available():
        impls["pytorch-softdtw-cuda"] = maghoumi
        impls["sdtw-cuda-torch"] = sdtw_cuda_torch
    if device.type == "cpu":
        impls["tslearn"] = tslearn
    impls[REFERENCE] = torchdtw_soft_dtw
    return impls
