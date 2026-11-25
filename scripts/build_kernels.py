"""Build the optional CUDA extension into the editable source package."""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).resolve().parents[1]
os.chdir(root)
source = root / "rover/kernels/csrc"
setup(
    name="rover-kernels",
    script_args=["build_ext", "--inplace"],
    packages=["rover", "rover.kernels"],
    ext_modules=[
        CUDAExtension(
            name="rover.kernels._kernels",
            sources=[str(source / "pybind.cpp"), str(source / "gemv.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
