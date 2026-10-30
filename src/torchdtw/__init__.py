"""DTW implementation using PyTorch C++ extensions, with CPU and GPU backends."""

from importlib.util import find_spec

import torch

from . import _C as _C
from ._dtw import dtw, dtw_batch, dtw_path
from ._soft import soft_dtw, soft_dtw_batch

if torch.version.cuda is not None and find_spec(f"{__name__}._C_cuda") is not None:
    from . import _C_cuda as _C_cuda
if torch.version.hip is not None and find_spec(f"{__name__}._C_rocm") is not None:
    from . import _C_rocm as _C_rocm

__all__ = ["dtw", "dtw_batch", "dtw_path", "soft_dtw", "soft_dtw_batch"]
