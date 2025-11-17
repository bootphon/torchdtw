"""Build the DTW PyTorch C++ extension."""

import os
import sys

from setuptools import Extension, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension

TORCH_CUDA_ARCH_LIST = "Volta;Turing;Ampere;Ada;Hopper;Blackwell"
TORCH_TARGET_VERSION = 0x020A000000000000


def get_flags() -> tuple[list[str], list[str]]:
    """Return the compiler and linker flags."""
    match sys.platform:
        case "linux":
            return ["-fdiagnostics-color=always", "-O3", "-fopenmp"], ["-fopenmp"]
        case "win32":
            return ["/O2", "/openmp"], []
        case "darwin":  # On MacOS, we use the OpenMP version vendored by PyTorch
            return ["-fdiagnostics-color=always", "-O3"], []
    raise RuntimeError(sys.platform)


def get_extension() -> Extension:
    """Either CUDA or CPU extension."""
    use_cuda = CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    compiler_flags, linker_flags = get_flags()
    extra_compile_args = {"cxx": [f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}", *compiler_flags], "nvcc": ["-O3"]}
    sources = ["src/torchdtw/csrc/dtw.cpp"]
    if use_cuda:
        os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST
        sources.append("src/torchdtw/csrc/cuda/dtw.cu")
    return extension(
        "torchdtw._C",
        sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=linker_flags,
        py_limited_api=True,
    )


if __name__ == "__main__":
    setup(
        ext_modules=[get_extension()],
        cmdclass={"build_ext": BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp312"}},
    )
