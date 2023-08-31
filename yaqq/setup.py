# Copyright © 2023 Aritra Sarkar
#
# Distributed under terms of the GNU Affero General Public License.

from setuptools import setup

setup(
    name='yaqq',
    version='0.13.2',
    install_requires=["numpy <= 1.23.5", "qiskit <= 0.43.3", "astropy <= 5.3.1", "matplotlib <= 3.7.2", "scipy <= 1.11.1", "tqdm <= 4.65.0", "qutip <= 4.7.2", "scikit-learn <= 1.3.0"],
)