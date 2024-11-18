# Copyright © 2023 Quantum Intelligence Research Group
#
# Distributed under terms of the GNU Affero General Public License.

from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent

name = 'yaqq'
version = '0.13.7'
description = 'Yet Another Quantum Quantizer - Design Space Exploration of Quantum Gate Sets using Novelty Search'
long_description = (this_directory / "README.md").read_text()
url = 'https://github.com/Advanced-Research-Centre/YAQQ'
author="Quantum Intelligence Research Group"

setup(
    name=name,
    version=version,
    install_requires=["numpy >= 1.26.4", "qiskit >= 1.2.4", "astropy >= 6.1.6", "matplotlib >= 3.9.2", "scipy >= 1.14.1", "tqdm >= 4.65.0", "qutip >= 5.0.4", "scikit-learn >= 1.5.2", "weylchamber >= 0.6.0"],
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=url,
    author=author,
    license="AGPL v3",
    project_urls={
        "Source Code": url,
    },
    python_requires=">=3.12",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords="quantum compiler",

)