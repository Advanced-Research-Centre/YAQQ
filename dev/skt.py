from __future__ import annotations
import warnings
import collections
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence
import numpy as np
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.quantum_info import process_fidelity, Choi
from astropy.coordinates import cartesian_to_spherical
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import math
from qiskit.extensions import UnitaryGate, U3Gate
from qiskit.circuit import Gate
from qiskit.quantum_info import random_unitary
import scipy.linalg as la
from qiskit.circuit.gate import Gate
from tqdm import tqdm
from scipy.optimize import minimize 

from typing import Optional, Union
from qiskit.circuit.controlledgate import ControlledGate

class UGate(Gate):
    U = np.identity(2)
    label = ""
    def __init__(self, label, unitary):
        self.U = unitary
        self.label = label
        """Create new gate."""
        super().__init__(label, 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return UdgGate(self.label+'dg', self.U)  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the U gate."""
        return self.U
   
class UdgGate(Gate):
    U = np.identity(2)
    label = ""
    def __init__(self, label, unitary):
        self.U = unitary
        self.label = label
        """Create new gate."""
        super().__init__(label, 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return UGate(self.label[:-2], self.U)  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the Udg gate."""
        return la.inv(self.U)

# class XGate(Gate):
#     def __init__(self, label: Optional[str] = None):
#         """Create new X gate."""
#         super().__init__("x", 1, [], label=label)
#     def inverse(self):
#         r"""Return inverted X gate (itself)."""
#         return XGate()  # self-inverse
#     def __array__(self, dtype=None):
#         """Return a numpy.array for the X gate."""
#         return np.array([[0, 1], [1, 0]], dtype=dtype)

# class CXGate(ControlledGate):
#     def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None):
#         """Create new CX gate."""
#         super().__init__(
#             "cx", 2, [], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=XGate()
#         )
#     def inverse(self):
#         """Return inverted CX gate (itself)."""
#         return CXGate(ctrl_state=self.ctrl_state)  # self-inverse
#     def __array__(self, dtype=None):
#         """Return a numpy.array for the CX gate."""
#         if self.ctrl_state:
#             return np.array(
#                 [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=dtype
#             )
#         else:
#             return np.array(
#                 [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=dtype
#             )

# from qiskit.synthesis.discrete_basis.generate_basis_approximations import generate_basic_approximations
####################################################################################################
# Updated generalized generate_basic_approximations of qiskit
####################################################################################################

class gen_basis_seq:

    Node = collections.namedtuple("Node", ("labels", "sequence", "children"))
    _1q_gates = {}

    def _check_candidate_kdtree(self, candidate, existing_sequences, tol=1e-10):
        from sklearn.neighbors import KDTree
        # do a quick, string-based check if the same sequence already exists
        if any(candidate.name == existing.name for existing in existing_sequences):
            return False
        points = np.array([sequence.product.flatten() for sequence in existing_sequences])
        candidate = np.array([candidate.product.flatten()])
        kdtree = KDTree(points)
        dist, _ = kdtree.query(candidate)
        return dist[0][0] > tol

    def _process_node(self, node: self.Node, basis: list[str], sequences: list[GateSequence]):  
        for label in basis:
            sequence = node.sequence.copy()
            sequence.append(self._1q_gates[label])
            if self._check_candidate_kdtree(sequence, sequences):
                sequences.append(sequence)
                node.children.append(self.Node(node.labels + (label,), sequence, []))
        return node.children

    def generate_basic_approximations(self, basis_gates: list[Gate], depth: int = 3) -> list[GateSequence]:
        """Generates a list of ``GateSequence``s with the gates in ``basic_gates``.
        Args:
            basis_gates: The gates from which to create the sequences of gates.
            depth: The maximum depth of the approximations.
            filename: If provided, the basic approximations are stored in this file.
        Returns:
            List of ``GateSequences`` using the gates in ``basic_gates``.
        """
        
        basis = []
        for gate in basis_gates:
            basis.append(gate.name)
            self._1q_gates[gate.label] = gate

        tree = self.Node((), GateSequence(), [])
        cur_level = [tree]
        sequences = [tree.sequence]
        for _ in [None] * depth:
            next_level = []
            for node in cur_level:
                next_level.extend(self._process_node(node, basis, sequences))
            cur_level = next_level
        return sequences


# from qiskit.transpiler.passes.synthesis import SolovayKitaev

# gbs = gen_basis_seq()
# max_depth = 2
# agent1_gateseq = gbs.generate_basic_approximations(agent1_gateset, max_depth)

# # recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition
# # skd1 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent1_gateseq)   

# print(len(agent1_gateseq))
# for i in agent1_gateseq:
#     print(i.name)