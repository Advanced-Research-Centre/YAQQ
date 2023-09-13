[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![PyPI version](https://badge.fury.io/py/yaqq.svg)](https://badge.fury.io/py/yaqq)

Copyright © 2023 Quantum Intelligence Research Group

Source code available at: https://github.com/Advanced-Research-Centre/YAQQ

Contact: https://www.linkedin.com/in/sarkararitra/

YAQQ: Yet Another Quantum Quantizer - Design Space Exploration of Quantum Gate Sets using Novelty Search

The YAQQ (Yaqq Another Quantum Quantizer) is an agent that searches for novel quantum gate sets. Given a gate set, it can find a complementary gate that performs better for a particular set of unitary transformations than the original gate set. It is possible theoretically because (a) there are an infinite number of ways of creating universal quantum computing gate sets - the ubiquity of quantum universality, (b) for each discrete gate set, there are certain quantum states that are easy to express, but many other quantum states which are exponentially costly - universal distribution for quantum automata. The cost, or the performance of a gate set, considers the fidelity when the gate set is used to decompose the target set of quantum transformations and the circuit complexity of the decomposition.

The theoretical foundation of this package, higher- and multi-order network models, was developed in the following works:
1. [RuliadTrotter: meta-modeling Metamathematical observers](https://community.wolfram.com/groups/-/m/t/2575951)
2. [Visualizing Quantum Circuit Probability: Estimating Quantum State Complexity for Quantum Program Synthesis](https://www.mdpi.com/1099-4300/25/5/763)
3. [Automated Quantum Software Engineering: why? what? how?](https://arxiv.org/abs/2212.00619)
4. [Discovering Quantum Circuit Components with Program Synthesis](https://arxiv.org/abs/2305.01707)
5. [Automated Gadget Discovery in Science](https://arxiv.org/abs/2212.12743)
6. The YAQQ name is inspired by the [YACC](https://en.wikipedia.org/wiki/Yacc) - a compiler-compiler, or a compiler-generator. In a similar way YAQQ provides the set of decompositions for the generated gate set.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

### Install:

```
pip install --upgrade yaqq
```

### Usage:

```
import yaqq
yaqq.run()
```

### Contributors:

Aritra Sarkar (project lead, development) Akash Kundu (development, test suite integration)

### Citation:
If you find the repository useful, please consider citing:

```
@misc{YAQQ,
  author={Sarkar, Aritra and Kundu, Akash},
  title={YAQQ: Yet Another Quantum Quantizer},
  howpublished={\url{[https://github.com/Advanced-Research-Centre/YAQQ](https://github.com/Advanced-Research-Centre/YAQQ)}},
  year={2023}
}
```