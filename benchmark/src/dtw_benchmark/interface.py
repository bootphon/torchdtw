from collections.abc import Callable
from typing import Literal

import torch

from .impl import (
    dtw_cython,
    dtw_cython_batch,
    dtw_numba,
    dtw_numba_batch,
    dtw_torch,
    dtw_torch_batch,
    dtw_triton,
    dtw_triton_batch,
)

type Backend = Literal["cython", "numba", "torch", "triton"]
type Device = Literal["cpu", "cuda"]
type Dtw = Callable[[torch.Tensor], torch.Tensor]
type DtwBatch = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]

MAPPING: dict[tuple[Backend, Device], tuple[Dtw, DtwBatch]] = {
    ("cython", "cpu"): (dtw_cython, dtw_cython_batch),
    ("numba", "cpu"): (dtw_numba, dtw_numba_batch),
    ("torch", "cpu"): (dtw_torch, dtw_torch_batch),
    ("torch", "cuda"): (dtw_torch, dtw_torch_batch),
    ("triton", "cuda"): (dtw_triton, dtw_triton_batch),
}


def available_implementations() -> list[tuple[Backend, Device]]:
    return list(MAPPING.keys())


def foo(i: Implementation):
    return True
