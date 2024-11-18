[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![PyPI version](https://badge.fury.io/py/yaqq.svg)](https://badge.fury.io/py/yaqq)

Copyright © 2023 Quantum Intelligence Research Group

Source code available at: https://github.com/Advanced-Research-Centre/YAQQ

Contact: https://www.linkedin.com/in/sarkararitra/

**YAQQ: Yet Another Quantum Quantizer - Design Space Exploration of Quantum Gate Sets using Novelty Search**

The standard model of quantum computation is based on quantum circuits, where the number and quality of the quantum gates composing the circuit influence the runtime and fidelity of the computation. The fidelity of the decomposition of quantum algorithms, represented as unitary matrices, to bounded depth quantum circuits depends strongly on the set of gates available for the decomposition routine. To investigate this dependence, we explore the design space of discrete quantum gate sets and present a software tool for comparative analysis of quantum processing units and control protocols based on their native gates. The evaluation is conditioned on a set of unitary transformations representing target use cases on the quantum processors. The cost function considers three key factors: (i) the statistical distribution of the decomposed circuits' depth, (ii) the statistical distribution of process fidelities for the approximate decomposition, and (iii) the relative novelty of a gate set compared to other gate sets in terms of the aforementioned properties. The developed software, called YAQQ (Yet Another Quantum Quantizer), enables the discovery of an optimized set of quantum gates through this tunable joint cost function. To identify these gate sets, we use the novelty search algorithm, circuit decomposition techniques (like Solovay-Kitaev, Cartan, and quantum Shannon decomposition), and stochastic optimization to implement YAQQ within the Qiskit quantum simulator environment. YAQQ exploits reachability tradeoffs conceptually derived from quantum algorithmic information theory. Our results demonstrate the pragmatic application of identifying gate sets that are advantageous to popularly used quantum gate sets in representing quantum algorithms. Consequently, we demonstrate pragmatic use cases of YAQQ in comparing transversal logical gate sets in quantum error correction codes, designing optimal quantum instruction sets, and compiling to specific quantum processors.


This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

### Install:

```
pip install --upgrade yaqq
```

### Usage:

#### Manual mode
```
import yaqq
qq = yaqq()
qq.yaqq_manual()
```
#### API mode
```
import yaqq
qq = yaqq(<CONFIG_FPATH>)
qq.yaqq_cfg(<CONFIG_FNAME>)
```

### Configuration Files:
The configuration file specified in the API mode automates the Manual mode's CLI. We have provided some example configuration files for various modes.

### Contributors:

Aritra Sarkar (project lead, development) Akash Kundu (development, test suite integration)

### Citation:
If you find the repository useful, please consider citing our [article](https://arxiv.org/abs/2406.17610):
```
@article{sarkar2024yaqq,
  title={YAQQ: Yet Another Quantum Quantizer - Design Space Exploration of Quantum Gate Sets using Novelty Search},
  author={Sarkar, Aritra and Kundu, Akash and Steinberg, Matthew and Mishra, Sibasish and Fauquenot, Sebastiaan and Acharya, Tamal and Miszczak, Jaros{\l}aw A and Feld, Sebastian},
  journal={arXiv preprint arXiv:2406.17610},
  year={2024}
}
```