"""Guard against CUDA tensors on Windows, where torchdtw is built without its CUDA extension."""

import sys

import torch


class CUDAOnWindowsError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("torchdtw was built without CUDA support on Windows. Move your tensors to CPU first.")


def check_no_cuda_on_windows(tensor: torch.Tensor) -> None:
    if sys.platform == "win32" and tensor.is_cuda:
        raise CUDAOnWindowsError
