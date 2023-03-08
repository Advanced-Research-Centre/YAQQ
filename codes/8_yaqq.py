"""
Program to:
    Option 1: compare two user defined gate sets.
    Option 2: generate novel gateset
"""


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
from qiskit.extensions import UnitaryGate
from qiskit.circuit import Gate
from qiskit.quantum_info import random_unitary
import scipy.linalg as la
from qiskit.circuit.gate import Gate

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

    def generate_basic_approximations(self, basis_gates: list[Gate], depth: int) -> list[GateSequence]:
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
        
####################################################################################################

class HGate(Gate):
    def __init__(self, label):
        """Create new gate."""
        super().__init__("h", 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return HGate(label="h")  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the b gate."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
class TGate(Gate):
    def __init__(self, label):
        """Create new gate."""
        super().__init__("t", 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return TdgGate(label="tdg")  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the b gate."""
        return np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
        
class TdgGate(Gate):
    def __init__(self, label):
        """Create new gate."""
        super().__init__("tdg", 1, [], label=label)
    def inverse(self):
        """Invert this gate."""
        return TGate(label="t")  # self-inverse
    def __array__(self, dtype=None):
        """Return a numpy.array for the b gate."""
        return np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
        
####################################################################################################

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
    
####################################################################################################

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

####################################################################################################

def compare_gs():
        
    # ===> Define gateset GS1 (standard gates)
    
    t = TGate(label="t")
    tdg = TdgGate(label="tdg")
    h = HGate(label="h")
    agent1_gateset = [h, t, tdg]
    
    # ===> Define gateset GS2 (standard gates via U-gate)
    
    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    tdg_U_mat = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
    Uh = UGate("Uh", h_U_mat)
    Ut = UGate("Ut", t_U_mat)
    Utdg = UGate("Utdg", tdg_U_mat)
    agent2_gateset = [Uh, Ut, Utdg]

    # ===> Generate sequences
    
    # print("H gate:")
    # print(agent1_gateset[0].__array__())
    # print(agent2_gateset[0].__array__())
    # print("T gate:")
    # print(agent1_gateset[1].__array__())
    # print(agent2_gateset[1].__array__())
    # print("Tdg gate:")
    # print(agent1_gateset[2].__array__())
    # print(agent2_gateset[2].__array__())

    gbs = gen_basis_seq()
    max_depth = 3 # maximum number of same consecutive gates allowed
    agent1_gateseq = gbs.generate_basic_approximations(agent1_gateset, max_depth) 
    agent2_gateseq = gbs.generate_basic_approximations(agent2_gateset, max_depth)
    # for i in agent1_gateseq:
    #    print(i.name)

    # ===> Declare SKT object
    
    recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition
    skd1 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent1_gateseq)    
    skd2 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent2_gateseq) 
    
    # ===> Make trial states on Bloch sphere
    
    points = 10
    rz_ang_list, rx_ang_list = fibo_bloch(points)[0], fibo_bloch(points)[1]   

    # ===> SK Decompose trail states for both gate sets and calculate fidelity/length
    
    fid_gs01 = []
    fid_gs02 = []
    len_gs01 = []
    len_gs02 = []

    for p in range(points):
        qc0 = QuantumCircuit(1)
        qc0.rz(rz_ang_list[p],0)
        qc0.rx(rx_ang_list[p],0)
        qc01 = skd1(qc0)
        qc02 = skd2(qc0)
        choi0 = Choi(qc0)
        choi01 = Choi(qc01)
        choi02 = Choi(qc02)
        fid_gs01.append(process_fidelity(choi0,choi01))
        fid_gs02.append(process_fidelity(choi0,choi02))
        len_gs01.append(qc01.depth())
        len_gs02.append(qc02.depth())

    # ===> Plot results
    
    _, ax = plt.subplots(1, 2)

    ax[0].plot(fid_gs01, '-x', label = "[h, t, tdg]")
    ax[0].plot(fid_gs02, '-o', label = "[uh, ut, utdg]")
    ax[1].plot(len_gs01, '-x', label = "[h, t, tdg]")
    ax[1].plot(len_gs02, '-o', label = "[uh, ut, utdg]")
    
    ax[0].set_ylabel("Process Fidelity")
    ax[1].set_ylabel("Decomposed Circuit Length")
    ax[0].set_ylim(bottom=0,top=1)
    ax[1].set_ylim(bottom=0,top=None)
    ax[0].legend()
    ax[1].legend()
    plt.show()

    return

####################################################################################################
 
def novel_gs():

    # ===> Define gateset GS1 (standard gates via U-gate)
    
    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    tdg_U_mat = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
    Uh = UGate("Uh", h_U_mat)
    Ut = UGate("Ut", t_U_mat)
    Utdg = UGate("Utdg", tdg_U_mat)
    agent1_gateset = [Uh, Ut, Utdg]

    # ===> Generate sequences and declare SKT object for GS1

    gbs = gen_basis_seq()
    max_depth = 3 # maximum number of same consecutive gates allowed
    agent1_gateseq = gbs.generate_basic_approximations(agent1_gateset, max_depth) 
    recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition
    skd1 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent1_gateseq)    
    
    # ===> Make trial states on Bloch sphere
    
    points = 20
    rz_ang_list, rx_ang_list = fibo_bloch(points)[0], fibo_bloch(points)[1]   

    # ===> Find complementary gate set via random search

    trials = 50
    gateset_list, total_fid_list, total_depth_list = [], [], []
    for _ in range(trials):

        # ===> Generate random unitaries, sequences and declare SKT object for GS2
        
        unitary1, unitary2, unitary3 = random_unitary(2).data, random_unitary(2).data, random_unitary(2).data
        U1 = UGate("U1", unitary1)
        U2 = UGate("U2", unitary2)
        U3 = UGate("U3", unitary3)
        agent2_gateset = [U1, U2, U3]
        agent2_gateseq = gbs.generate_basic_approximations(agent2_gateset, max_depth) 
        skd2 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent2_gateseq)    

        # ===> Decompose each state on Bloch sphere and compare GS1 and GS2

        fid_list, depth_list = [], []
        for p in range(points):
            qc0 = QuantumCircuit(1)
            qc0.rz(rz_ang_list[p],0)
            qc0.rx(rx_ang_list[p],0)
            qc01 = skd1(qc0)
            qc02 = skd2(qc0)
            choi0 = Choi(qc0)
            choi01 = Choi(qc01)
            choi02 = Choi(qc02)
            pf01 = process_fidelity(choi0,choi01)
            pf02 = process_fidelity(choi0,choi02)
            fid_list.append([pf01,pf02])
            depth_list.append([qc01.depth(),qc02.depth()])
        
        gateset_list.append([unitary1, unitary2, unitary3])
        total_fid_list.append(fid_list)
        total_depth_list.append(depth_list)

    save_data = input("Save data? [y/n]: ")
    if save_data == 'y':
        np.save( 'data/total_depth_list', total_depth_list )
        np.save( 'data/total_fid_list', total_fid_list )
        np.save( 'data/gateset_list', gateset_list )

    plot_data = input("Plot data? [y/n]: ")
    if plot_data == 'y':
        x1_list, x2_list = [], []
        for i in total_fid_list:
            x1, x2 = 0, 0
            for fid in i:
                x1 += fid[0]
                x2 += fid[1]
            x1_list.append(x1 / points)
            x2_list.append(x2 / points)
        plt.plot(x1_list, '-x' , label = '[uh, ut, utdg]')
        plt.plot(x2_list, '-o', label = '[u1, u2, u3]')
        plt.legend()
        plt.show()

    return

####################################################################################################        

if __name__ == "__main__":

    print("Option 1: Compare gate sets GS1 and GS2")
    print("Option 2: Generate complementary gate set of GS1")
    yaqq_mode = input("Enter choice: ")
    match yaqq_mode:
        case '1': compare_gs()
        case '2': novel_gs()
        case _  : print("Invalid option")
    print("Thank you for using YAQQ.")