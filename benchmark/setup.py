# ruff: noqa: INP001
"""Build Cython DTW extension."""

import numpy as np
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "dtw_benchmark.dtw_cython",
            sources=["src/dtw_benchmark/dtw_cython.pyx"],
            define_macros=[("Py_LIMITED_API", 0x030C0000)],
            extra_compile_args=["-O3", "-DNPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION"],
            include_dirs=[np.get_include()],
            py_limited_api=True,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
