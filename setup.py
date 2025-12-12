"""Build the DTW PyTorch C++ extension."""

import os
import sys

from setuptools import Extension, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension


def get_flags() -> tuple[list[str], list[str]]:
    """Return the compiler and linker flags."""
    match sys.platform:
        case "linux":
            return ["-Werror", "-fdiagnostics-color=always", "-O3", "-fopenmp"], ["-fopenmp"]
        case "win32":
            return ["/WX", "/O2", "/openmp"], []
        case "darwin":  # On MacOS, we use the OpenMP version vendored by PyTorch
            return ["-Werror", "-fdiagnostics-color=always", "-O3"], []
    raise RuntimeError(sys.platform)


def get_extension() -> Extension:
    """Either CUDA or CPU extension."""
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "Volta;Turing;Ampere;Ada;Hopper;Blackwell"
    use_cuda = CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    sources = ["src/torchdtw/csrc/dtw.cpp"] + (["src/torchdtw/csrc/cuda/dtw.cu"] if use_cuda else [])
    compiler_flags, linker_flags = get_flags()
    extra_compile_args = {"cxx": ["-DTORCH_TARGET_VERSION=0x020A000000000000", *compiler_flags], "nvcc": ["-O3"]}
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
