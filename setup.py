# TODO
import time

import re

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


__version__ = re.search(r"VERSION (\d+\.\d+\.\d+)", open("CMakeLists.txt").read()).group(1)
__lib_name__ = "multivariate_isotonic_regression"


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(
            name,
            sources=[],
            runtime_library_dirs=["./build/lib.linux-x86_64-cpython-310"],
            extra_objects=["./build/lib.linux-x86_64-cpython-310/libhighs.so.1.6.0"],
        )
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        build_args = []

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_GIR_PYTHON=ON",
            "-D{__lib_name__.upper()}_VERSION_INFO={__version__}", # version
        ]

        # Add any flags set via environmnet variable
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # TODO
        print("THIS PATH: ", build_temp)
        print("THIS PATH: ", cmake_args)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cp", os.path.join(build_temp, "_deps", "highs-build", "lib", "libhighs.so.1.6.0"),
                   os.path.join("build", "lib.linux-x86_64-cpython-310")]
        )
        # time.sleep(60*5)

setup(
    name=__lib_name__,
    version=__version__,
    author="Edward Wall",
    description="Provides methods for Isotonic Regression in arbitrary dimensions and on arbitrary graphs structures.",
    ext_modules=[CMakeExtension("GeneralisedIsotonicRegression")],
    cmdclass={"build_ext": CMakeBuild},
    # zip_safe=False,
    python_requires=">=3.7",
)
