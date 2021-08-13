#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

# read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()

VERSION = imp.load_source("", "connectome_manipulator/version.py").__version__

setup(
    name="connectome_manipulator",
    author="Christoph Pokorny",
    author_email="christoph.pokorny@epfl.ch",
    version=VERSION,
    description="A tool to perform structural manipulations on a SONATA circuit connectome",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpgitlab.epfl.ch/conn/structural/connectome_manipulator.git",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/SSCXDIS/issues",
        "Source": "https://bbpgitlab.epfl.ch/conn/structural/connectome_manipulator.git",
    },
    license="BBP-internal-confidential",
    install_requires=[
        "bluepysnap",
        "numpy",
        "progressbar",
        "scipy",
        "sklearn",
        "voxcell",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    extras_require={"docs": ["sphinx", "sphinx-bluebrain-theme"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
