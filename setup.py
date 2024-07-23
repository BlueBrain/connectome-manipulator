#!/usr/bin/env python

# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import importlib.util
from setuptools import setup, find_packages

# read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()

spec = importlib.util.spec_from_file_location(
    "connectome_manipulator.version",
    "connectome_manipulator/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__


setup(
    name="connectome-manipulator",
    author="Blue Brain Project, EPFL",
    version=VERSION,
    description="A connectome manipulation framework for SONATA circuits",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/BlueBrain/connectome-manipulator.git",
    license="Apache-2",
    install_requires=[
        "bluepysnap>=3.0.1",
        "numpy>=1.24.3",
        "pandas>=1.5.3",
        "progressbar>=2.5",
        "pyarrow>=10.0.1",
        "scipy>=1.10.1",
        "scikit-learn>=1.3.2",
        "voxcell>=3.1.5",
        "tables>=3.8.0",  # Optional dependency of pandas.DataFrame.to_hdf()
        "distributed>=2023.4.1",  # Dask
        "dask-mpi>=2022.4.0",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    extras_require={"docs": ["sphinx", "sphinx-bluebrain-theme"]},
    entry_points={
        "console_scripts": [
            "connectome-manipulator=connectome_manipulator.cli:app",
            "parallel-manipulator=connectome_manipulator.cli_parallel:app",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
