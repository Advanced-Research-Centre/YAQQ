# Copyright © 2023 Quantum Intelligence Research Group
#
# Distributed under terms of the GNU Affero General Public License.

from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent

name = 'yaqq'
version = '0.13.5'
description = 'Yet Another Quantum Quantizer - Design Space Exploration of Quantum Gate Sets using Novelty Search'
long_description = (this_directory / "README.md").read_text()
url = 'https://github.com/Advanced-Research-Centre/YAQQ'
author="Quantum Intelligence Research Group"

setup(
    name=name,
    version=version,
    install_requires=["numpy >= 1.23.5", "qiskit >= 0.43.3", "astropy >= 5.3.1", "matplotlib >= 3.7.2", "scipy >= 1.11.1", "tqdm >= 4.65.0", "qutip >= 4.7.2", "scikit-learn >= 1.3.0", "weylchamber >= 0.4.0"],
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=url,
    author=author,
    license="AGPL v3",
    project_urls={
        "Source Code": url,
    },
    python_requires=">=3.11",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    keywords="quantum compiler",

)