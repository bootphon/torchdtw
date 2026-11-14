"""Exceptions raised by the Python wrappers of both DTW and soft-DTW."""

import sys

import torch


class CUDAOnWindowsError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("torchdtw was built without CUDA support on Windows. Move your tensors to CPU first.")


def check_no_cuda_on_windows(tensor: torch.Tensor) -> None:
    if sys.platform == "win32" and tensor.is_cuda:
        raise CUDAOnWindowsError


class RankError(RuntimeError):
    def __init__(self, expected: str, actual: int) -> None:
        super().__init__(f"distances must be a {expected} tensor, got {actual}D")


class SinglePairLengthsError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("sx and sy are only accepted with 3D distances: a 2D tensor is a single pair, read whole")
