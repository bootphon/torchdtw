"""Build the DTW PyTorch C++ extension."""

import os
import sys

from setuptools import Extension, setup
from torch.utils.cpp_extension import CUDA_HOME, IS_HIP_EXTENSION, BuildExtension, CppExtension, CUDAExtension


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


class CUDAArchListError(RuntimeError):
    """To raise if CUDA is found and TORCH_CUDA_ARCH_LIST is not set."""

    def __init__(self) -> None:
        super().__init__(
            "You must explicitly set TORCH_CUDA_ARCH_LIST to build from source if CUDA is found.\n"
            "Check you supported gpu architectures beforehand.\n"
            "For example: TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX'"
        )


class ROCmArchListError(RuntimeError):
    """To raise if PyTorch is built with ROCm and PYTORCH_ROCM_ARCH is not set."""

    def __init__(self) -> None:
        super().__init__(
            "You must explicitly set PYTORCH_ROCM_ARCH to build from source with a ROCm build of PyTorch.\n"
            "Check you supported gpu architectures beforehand.\n"
            "For example: PYTORCH_ROCM_ARCH='gfx90a;gfx942;gfx1100'"
        )


def get_extensions() -> list[Extension]:
    """Return the CPU extension, plus the CUDA or ROCm extension if a GPU toolkit is found.

    The GPU kernels live in a separate shared object so that the CPU one does not link against
    libtorch_cuda, libtorch_hip or the GPU runtime, and can be loaded with CPU-only builds of PyTorch.
    """
    use_rocm = IS_HIP_EXTENSION and sys.platform == "linux"
    use_cuda = not use_rocm and CUDA_HOME is not None and sys.platform != "win32"
    if use_cuda and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        raise CUDAArchListError
    if use_rocm and "PYTORCH_ROCM_ARCH" not in os.environ:
        raise ROCmArchListError
    compiler_flags, linker_flags = get_flags()
    extra_compile_args = {
        "cxx": ["-DTORCH_TARGET_VERSION=0x020A000000000000", "-DTORCH_STABLE_ONLY", *compiler_flags],
        "nvcc": ["-DTORCH_TARGET_VERSION=0x020A000000000000", "-O3"],
    }
    extensions = [
        CppExtension(
            "torchdtw._C",
            ["src/torchdtw/csrc/dtw.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=linker_flags,
            py_limited_api=True,
        )
    ]
    if use_cuda:
        cuda_extension = CUDAExtension(
            "torchdtw._C_cuda",
            ["src/torchdtw/csrc/cuda/dtw.cu"],
            extra_compile_args=extra_compile_args,
            py_limited_api=True,
        )
        # Remove cudart so it does not appear in the .so's dependencies.
        # Cudart symbols are resolved at runtime from the cudart already loaded by PyTorch,
        # making the wheel compatible across CUDA major versions.
        cuda_extension.libraries = [lib for lib in cuda_extension.libraries if "cudart" not in lib]
        extensions.append(cuda_extension)
    if use_rocm:
        extensions.append(
            CUDAExtension(
                "torchdtw._C_rocm",
                ["src/torchdtw/csrc/cuda/dtw.cu"],
                extra_compile_args=extra_compile_args,
                py_limited_api=True,
            )
        )
    return extensions


if __name__ == "__main__":
    setup(
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp312"}},
    )
