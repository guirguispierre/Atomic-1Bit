"""Minimal setuptools packaging for Atomic-1Bit.

The build also compiles the C++ ternary kernel (atomic_1bit/core ->
libatomic.so) so that `pip install .` leaves a usable install. Machines
without make/g++ get a warning, not a failed install; the kernel can be
built later with `make -C atomic_1bit/core` and the Python wrapper also
auto-builds it on first import when a toolchain is present.
"""

import os
import re
import subprocess
import warnings

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py

ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT, "atomic_1bit", "core")


def read_version():
    init = os.path.join(ROOT, "atomic_1bit", "__init__.py")
    with open(init) as f:
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read())
    if not match:
        raise RuntimeError("could not read __version__ from atomic_1bit/__init__.py")
    return match.group(1)


def read_requirements():
    reqs = []
    with open(os.path.join(ROOT, "requirements.txt")) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                reqs.append(line)
    return reqs


def build_kernel():
    """Compile libatomic.so. Warns instead of failing when no toolchain."""
    try:
        subprocess.check_call(["make", "-C", CORE_DIR])
    except (OSError, subprocess.CalledProcessError) as e:
        warnings.warn(
            f"could not build libatomic.so during install ({e}). "
            "The C++ kernel is required for Python inference; build it "
            "later with 'make -C atomic_1bit/core'."
        )


class BuildPy(build_py):
    """Runs the kernel build before copying package files into the wheel."""

    def run(self):
        build_kernel()
        super().run()


with open(os.path.join(ROOT, "README.md")) as f:
    long_description = f.read()

setup(
    name="atomic-1bit",
    version=read_version(),
    description="Bare-metal inference engine for 1.58-bit ternary language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    # most subdirs (model/, nn/, python/, utils/, core/) have no __init__.py
    # and are reached through sys.path, so namespace discovery is required or
    # the wheel ships an incomplete package
    packages=find_namespace_packages(include=["atomic_1bit", "atomic_1bit.*"]),
    # ship the kernel sources so an installed wrapper can still auto-build,
    # and the built libatomic.so itself when the toolchain is present
    package_data={
        "atomic_1bit": [
            "core/libatomic.so",
            "core/Makefile",
            "core/kernel.h",
            "core/backends/*",
        ]
    },
    python_requires=">=3.8",
    install_requires=read_requirements(),
    cmdclass={"build_py": BuildPy},
)
