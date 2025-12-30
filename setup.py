import os
import sys
import subprocess
import sysconfig
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

REPO_PATH = Path(__file__).resolve().parent

python_include_path = sysconfig.get_path("include")

CUDA_HOME = "/usr/local/cuda"
CUDA_INCLUDE_DIR = os.path.join(CUDA_HOME, "include")
CUDA_LIB_DIR = os.path.join(CUDA_HOME, "lib64")
PACKAGE_NAME = "matmul"

INCLUDE_DIRS = [
    "cuda/include",
    CUDA_INCLUDE_DIR,
    python_include_path,
    pybind11.get_include(),
]

LIBRARY_DIRS = [CUDA_LIB_DIR]
LIBRARIES = ["cudart", "cublas"]

CXX_FLAGS = ["-std=c++17", "-O3"]
NVCC_FLAGS = [
    "-std=c++17",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-arch=sm_80",
]

SRC_FILES = [
    "src/bindings.cpp",
    "cuda/src/tiledMultiply.cu",
    "cuda/src/cuBLASMultiply.cu",
]

# Make sure nvcc is available
try:
    subprocess.check_call(["nvcc", "--version"])
except Exception as e:
    print(f"nvcc compiler for CUDA not found: {e}; exiting")
    sys.exit(1)


class BuildExtCUDA(build_ext):
    """Compile .cu files with nvcc, others with the normal C++ compiler."""

    def build_extensions(self):
        from distutils.sysconfig import customize_compiler

        compiler = self.compiler
        customize_compiler(compiler)

        # Let distutils know about .cu files
        if ".cu" not in compiler.src_extensions:
            compiler.src_extensions.append(".cu")

        default_compile = compiler._compile
        nvcc = "nvcc"

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                # nvcc compile
                cmd = [nvcc, "-c", src, "-o", obj] + NVCC_FLAGS
                for inc in INCLUDE_DIRS:
                    cmd.append(f"-I{inc}")
                print("NVCC:", " ".join(cmd))
                self.spawn(cmd)
            else:
                # normal C++ compile
                extra_postargs = list(extra_postargs or []) + CXX_FLAGS
                default_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        compiler._compile = _compile
        super().build_extensions()


ext_modules = [
    Extension(
        PACKAGE_NAME,
        sources=SRC_FILES,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES,
        language="c++",
    )
]

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    description="CUDA tiled matrix multiplication exposed to Python via pybind11",
    author="Sebastia Agramunt Puig",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtCUDA},
    zip_safe=False,
    install_requires=[
        "numpy",
    ],
)
