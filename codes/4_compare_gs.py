#from qiskit.synthesis.discrete_basis.generate_basis_approximations import generate_basic_approximations

# Updated generalized generate_basic_approximations of qiskit
####################################################################################################

from __future__ import annotations
import warnings
import collections
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.utils import optionals
from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence

Node = collections.namedtuple("Node", ("labels", "sequence", "children"))

_1q_gates = {}

def _check_candidate(candidate, existing_sequences, tol=1e-10):
    if optionals.HAS_SKLEARN:
        return _check_candidate_kdtree(candidate, existing_sequences, tol)

    warnings.warn(
        "The SolovayKitaev algorithm relies on scikit-learn's KDTree for a "
        "fast search over the basis approximations. Without this, we fallback onto a "
        "greedy search with is significantly slower. We highly suggest to install "
        "scikit-learn to use this feature.",
        category=RuntimeWarning,
    )
    return _check_candidate_greedy(candidate, existing_sequences, tol)

def _check_candidate_greedy(candidate, existing_sequences, tol=1e-10):
    # do a quick, string-based check if the same sequence already exists
    if any(candidate.name == existing.name for existing in existing_sequences):
        return False

    for existing in existing_sequences:
        if matrix_equal(existing.product_su2, candidate.product_su2, ignore_phase=True, atol=tol):
            # is the new sequence less or more efficient?
            return len(candidate.gates) < len(existing.gates)
    return True

@optionals.HAS_SKLEARN.require_in_call
def _check_candidate_kdtree(candidate, existing_sequences, tol=1e-10):
    """Check if there's a candidate implementing the same matrix up to ``tol``.
    This uses a k-d tree search and is much faster than the greedy, list-based search.
    """
    from sklearn.neighbors import KDTree

    # do a quick, string-based check if the same sequence already exists
    if any(candidate.name == existing.name for existing in existing_sequences):
        return False

    points = np.array([sequence.product.flatten() for sequence in existing_sequences])
    candidate = np.array([candidate.product.flatten()])

    kdtree = KDTree(points)
    dist, _ = kdtree.query(candidate)

    return dist[0][0] > tol

def _process_node(node: Node, basis: list[str], sequences: list[GateSequence]):  
    for label in basis:
        sequence = node.sequence.copy()
        sequence.append(_1q_gates[label])
        if _check_candidate(sequence, sequences):
            sequences.append(sequence)
            node.children.append(Node(node.labels + (label,), sequence, []))
    return node.children

def generate_basic_approximations(basis_gates: list[Gate], depth: int) -> list[GateSequence]:
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
        _1q_gates[gate.label] = gate
    tree = Node((), GateSequence(), [])
    cur_level = [tree]
    sequences = [tree.sequence]
    for _ in [None] * depth:
        next_level = []
        for node in cur_level:
            next_level.extend(_process_node(node, basis, sequences))
        cur_level = next_level
    return sequences
        
####################################################################################################

import numpy as np
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.quantum_info import process_fidelity, Choi
from astropy.coordinates import cartesian_to_spherical
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import math
from qiskit.extensions import UnitaryGate
from qiskit.circuit import Gate

"""
Decompose QuantumCircuit qc with gateset gs
Ref: qiskit\transpiler\passes\synthesis\solovay_kitaev_synthesis.py
"""
def equivalent_decomposition(qc,gs):

    max_depth = 3 # maximum number of same consequetive gates allowed
    recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition
    agent_approximations = generate_basic_approximations(gs, max_depth) 
    #print(gs)
    #for i in agent_approximations:
    #    print(i.name)
    skd = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent_approximations)
    return skd(qc)

"""
Evenly distributing n points on a Bloch sphere
Ref: stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
"""
def fibo_bloch(samples):
    rz_angle, rx_angle = [],[]
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        sphe_coor = cartesian_to_spherical(x, y, z)
        rz_angle.append(sphe_coor[1].radian+math.pi/2)
        rx_angle.append(sphe_coor[2].radian)
    return rz_angle, rx_angle

from qiskit.circuit.gate import Gate
            
class TGate(Gate):
    def __init__(self, label):
        """Create new gate."""
        super().__init__("t", 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return TdgGate(label="tdg")  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the b gate."""
        return np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=dtype)
        
class TdgGate(Gate):
    def __init__(self, label):
        """Create new gate."""
        super().__init__("tdg", 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return TGate(label="t")  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the b gate."""
        return np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=dtype)
        
class HGate(Gate):
    def __init__(self, label):
        """Create new gate."""
        super().__init__("h", 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return HGate(label="h")  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the b gate."""
        return np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2)
 
class BGate(Gate):
    def __init__(self, label):
        """Create new gate."""
        super().__init__("b", 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return BGate(label="b")  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the b gate."""
        return np.array([[0, 1], [1, 0]], dtype=dtype)
        
if __name__ == "__main__":

    # Make gates

    t = TGate(label="t")
    tdg = TdgGate(label="tdg")
    h = HGate(label="h")
    b = BGate(label="b")
    
    # Make gatesets
    
    agent1_gateset = [t, h, tdg]
    agent2_gateset = [b, h, tdg]

    points = 50
    rz_ang_list, rx_ang_list = fibo_bloch(points)[0], fibo_bloch(points)[1]
    
    result_db = []
    for p in range(points):
        qc0 = QuantumCircuit(1)
        qc0.rz(rz_ang_list[p],0)
        qc0.rx(rx_ang_list[p],0)
        qc01 = equivalent_decomposition(qc0,agent1_gateset)
        qc02 = equivalent_decomposition(qc0,agent2_gateset)
        choi0 = Choi(qc0)
        choi01 = Choi(qc01)
        choi02 = Choi(qc02)
        pf01 = process_fidelity(choi0,choi01)
        pf02 = process_fidelity(choi0,choi02)
        result_db.append([qc01.depth(),qc02.depth(),pf01,pf02])
    
    y1,y2 = [],[]
    for i in result_db:
        y1.append(i[2])
        y2.append(i[3])
        # print(i,'\n')

    plt.plot(y1, '-x', label = "[t, h, tdg]")
    plt.plot(y2, '-o', label = "[b, h, tdg]")
    plt.ylim((0,1))
    plt.legend()
    plt.show()