Welcome to YAQQ's documentation!
===================================

**YAQQ** (Yet Another Quantum Quantizer) is a Python library for Design Space Exploration of Quantum Gate Sets using Novelty Search.

The standard model of quantum computation is based on quantum circuits, where the number and quality of the quantum gates composing the circuit influence the runtime and fidelity of the computation. The fidelity of the decomposition of quantum algorithms, represented as unitary matrices, to bounded depth quantum circuits depends strongly on the set of gates available for the decomposition routine. To investigate this dependence, we explore the design space of discrete quantum gate sets and present a software tool for comparative analysis of quantum processing units and control protocols based on their native gates. The evaluation is conditioned on a set of unitary transformations representing target use cases on the quantum processors. The cost function considers three key factors: (i) the statistical distribution of the decomposed circuits' depth, (ii) the statistical distribution of process fidelities for the approximate decomposition, and (iii) the relative novelty of a gate set compared to other gate sets in terms of the aforementioned properties. The developed software, called YAQQ (Yet Another Quantum Quantizer), enables the discovery of an optimized set of quantum gates through this tunable joint cost function. To identify these gate sets, we use the novelty search algorithm, circuit decomposition techniques (like Solovay-Kitaev, Cartan, and quantum Shannon decomposition), and stochastic optimization to implement YAQQ within the Qiskit quantum simulator environment. YAQQ exploits reachability tradeoffs conceptually derived from quantum algorithmic information theory. Our results demonstrate the pragmatic application of identifying gate sets that are advantageous to popularly used quantum gate sets in representing quantum algorithms. Consequently, we demonstrate pragmatic use cases of YAQQ in comparing transversal logical gate sets in quantum error correction codes, designing optimal quantum instruction sets, and compiling to specific quantum processors.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api
