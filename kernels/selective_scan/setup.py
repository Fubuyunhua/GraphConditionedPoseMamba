"""Build the CUDA selective-scan extension used by this release.

Default: compile ``selective_scan_cuda_core``.  Set
``SELECTIVE_SCAN_MODES=core,oflex`` to build the optional oflex extension too.
Architecture flags are delegated to PyTorch and ``TORCH_CUDA_ARCH_LIST``.
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
MODES = tuple(
    item.strip()
    for item in os.getenv("SELECTIVE_SCAN_MODES", "core").split(",")
    if item.strip()
)
SOURCES = {
    "core": (
        "csrc/selective_scan/cus/selective_scan.cpp",
        "csrc/selective_scan/cus/selective_scan_core_fwd.cu",
        "csrc/selective_scan/cus/selective_scan_core_bwd.cu",
    ),
    "oflex": (
        "csrc/selective_scan/cusoflex/selective_scan_oflex.cpp",
        "csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu",
        "csrc/selective_scan/cusoflex/selective_scan_core_bwd.cu",
    ),
}
NAMES = {
    "core": "selective_scan_cuda_core",
    "oflex": "selective_scan_cuda_oflex",
}

unknown = sorted(set(MODES) - set(SOURCES))
if unknown:
    raise ValueError(f"Unsupported SELECTIVE_SCAN_MODES: {unknown}")

common_nvcc = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]

extensions = [
    CUDAExtension(
        name=NAMES[mode],
        sources=[str(ROOT / source) for source in SOURCES[mode]],
        include_dirs=[str(ROOT / "csrc/selective_scan")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": common_nvcc,
        },
    )
    for mode in MODES
]

setup(
    name="graph-posemamba-selective-scan",
    version="0.1.0",
    description="CUDA selective scan for GraphConditionedPoseMamba",
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
