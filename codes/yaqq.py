"""
https://github.com/Advanced-Research-Centre/YAQQ
"""

########################################################################################################################################################################################################

from __future__ import annotations
import collections
import os
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize  
from astropy.coordinates import cartesian_to_spherical

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.gate import Gate
from qiskit.extensions import UnitaryGate, UGate
from qiskit.quantum_info import random_statevector, random_unitary, Statevector, Operator, process_fidelity, Choi
from qiskit.quantum_info.operators.random import random_unitary
from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence
from qiskit.transpiler.passes.synthesis import SolovayKitaev

import qutip as qt
from qutip.measurement import measurement_statistics


########################################################################################################################################################################################################

"""
Data Set Generation
"""

# ------------------------------------------------------------------------------------------------ #

"""
Data Set Generation: Evenly distributed states (using golden mean)
Ref: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
"""

def gen_ds_fiboS(samples = 100, dimensions = 1):
    ds = []
    phi = math.pi * (3. - math.sqrt(5.))        # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2    # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)           # radius at y
        theta = phi * i                         # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        sphe_coor = cartesian_to_spherical(x, y, z)
        z_ang = sphe_coor[1].radian+math.pi/2
        x_ang = sphe_coor[2].radian
        qc = QuantumCircuit(dimensions)       
        qc.ry(z_ang,0)  # To rotate state z_ang from |0>, rotate about Y
        qc.rz(x_ang,0)  # To rotate state x_ang from |+>, rotate about Z
        fiboU_0 = Operator.from_circuit(qc)      
        ds.append(UnitaryGate(fiboU_0,label='FiboU'+str(i)))
    return ds

# ------------------------------------------------------------------------------------------------ #

"""
Data Set Generation: Haar random states
Ref: https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_statevector.html
"""

def gen_ds_randS(samples = 100, dimensions = 1):
    ds = []
    for i in range(samples):
        qc = QuantumCircuit(dimensions)
        randS = random_statevector(2**dimensions).data
        qc.prepare_state(randS, list(range(0, dimensions)))   
        randU_0 = Operator.from_circuit(qc)
        ds.append(UnitaryGate(randU_0,label='RndU'+str(i)))
    return ds

# ------------------------------------------------------------------------------------------------ #

"""
Data Set Generation: Haar random unitaries
Ref: https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_unitary.html
"""

def gen_ds_randU(samples = 100, dimensions = 1):
    ds = []
    for i in range(samples):
        ds.append(UnitaryGate(random_unitary(2**dimensions),label='RndU'+str(i)))
    return ds

# ------------------------------------------------------------------------------------------------ #

########################################################################################################################################################################################################

"""
Data Set Visualization and Result Plotting
"""

# ------------------------------------------------------------------------------------------------ #

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# ------------------------------------------------------------------------------------------------ #

def vis_ds_randU(ds):
    b = qt.Bloch()
    b.point_marker = ['o']
    b.point_size = [20]
    samples = len(ds)
    color = []
    for i in range(samples):
        qc = QuantumCircuit(1)
        qc.append(ds[i], [0])
        sv = Statevector(qc).data
        b.add_states(qt.Qobj(sv), kind='point')
        _, _, pX = measurement_statistics(qt.Qobj(sv), qt.sigmax())
        _, _, pY = measurement_statistics(qt.Qobj(sv), qt.sigmay())
        _, _, pZ = measurement_statistics(qt.Qobj(sv), qt.sigmaz())
        color.append(rgb_to_hex(int(pX[0]*255),int(pY[0]*255),int(pZ[0]*255)))
        
    b.point_color = color
    b.render()
    plt.show()

# ------------------------------------------------------------------------------------------------ #

def vis_dsU_gs(ds, pf, cd):
    b = qt.Bloch()
    b.point_marker = ['o']
    b.point_size = [20]
    samples = len(ds)
    color = []
    cd_max = max(cd)
    for i in range(samples):
        qc = QuantumCircuit(1)
        qc.append(ds[i], [0])
        sv = Statevector(qc).data
        b.add_states(qt.Qobj(sv), kind='point')
        _, _, pX = measurement_statistics(qt.Qobj(sv), qt.sigmax())
        _, _, pY = measurement_statistics(qt.Qobj(sv), qt.sigmay())
        _, _, pZ = measurement_statistics(qt.Qobj(sv), qt.sigmaz())
        color.append(rgb_to_hex(int(pf[i]*255),int(cd[i]*255/cd_max),int(pf[i]*255)))
        
    b.point_color = color
    b.render()
    plt.show()

# ------------------------------------------------------------------------------------------------ #

def plot_compare_gs(gs1,gs2,pf1,pf2,cd1,cd2,pfivt=False):
        
        avg_fid_gs01 = np.mean(pf1)
        avg_fid_gs02 = np.mean(pf2)
        avg_dep_gs01 = np.mean(cd1)
        avg_dep_gs02 = np.mean(cd2) 
        
        ivt_fid_gs01 = np.subtract(1,pf1)

        _, ax = plt.subplots(1, 2)
        ax[0].plot(pf1, '-x', color = 'r', label = 'PF ['+gs1+']')
        ax[0].plot(pf2, '-o', color = 'b', label = 'PF ['+gs2+']')
        if pfivt:
            ax[0].plot(ivt_fid_gs01, '-x', color = 'g', label = 'target PF trend')

        ax[0].axhline(y=avg_fid_gs01, linestyle='-.', color = 'r' , label = 'avg.PF ['+gs1+']')
        ax[0].axhline(y=avg_fid_gs02, linestyle='-.', color = 'b' , label = 'avg.PF ['+gs2+']')

        ax[1].plot(cd1, '-x', color = 'r', label = 'CD ['+gs1+']')
        ax[1].plot(cd2, '-o', color = 'b', label = 'CD ['+gs2+']')

        ax[1].axhline(y=avg_dep_gs01, linestyle='-.', color = 'r', label = 'avg.CD ['+gs1+']')
        ax[1].axhline(y=avg_dep_gs02, linestyle='-.', color = 'b', label = 'avg.CD ['+gs2+']')

        ax[0].set_ylabel("Process Fidelity")
        ax[1].set_ylabel("Circuit Depth")
        ax[0].set_ylim(bottom=0,top=1)
        ax[1].set_ylim(bottom=0,top=None)
        ax[0].legend()
        ax[1].legend()
        # _, ax = plt.subplots(1, 2, figsize = (7,3.5), sharex=True, layout="constrained")
        # ax[0].set_xlabel("Equidistant Points")
        # ax[1].set_xlabel("Equidistant Points")
        # plt.legend(ncol = 2, bbox_to_anchor = (1, 1.13))

        plot_save = input("Save plots and data? [Y/N] (def.: N): ") or 'N'
        if plot_save == 'Y':
            plt.savefig('figures/lok_dekhano_plot.pdf')
            plt.savefig('figures/lok_dekhano_plot.png')      
            # np.save('data/gateset_list', gs2)     # TBD:  GS2 has only names [U1,U2,U3], not matrix

        plt.show()

# ------------------------------------------------------------------------------------------------ #

########################################################################################################################################################################################################

"""
Unitary Decomposition Methods using given Gate Set 
"""

# ------------------------------------------------------------------------------------------------ #

"""
Given an unitary and a gate set, use trials to find a circuit using the gate set that is close to the unitary
"""

def dcmp_rand(randU, gs, trials = 100, max_depth= 100):

    qc0 = QuantumCircuit(1)
    qc0.append(Operator(randU),[0])
    choi0 = Choi(qc0)
    
    pfi_best = 0
    dep_best = 0
    qcirc_best = []
    for i in range(trials):
        dep = random.randrange(1,max_depth)
        seq = random.choices(list(gs.keys()), k=dep)
        qc0 = QuantumCircuit(1)
        for i in seq:
            qc0.append(gs[i], [0])
        choi01 = Choi(qc0)
        pfi = process_fidelity(choi0,choi01)
        if pfi > pfi_best:
            pfi_best = pfi
            qcirc_best = qc0
            dep_best = dep

    return pfi_best, dep_best, qcirc_best

# ------------------------------------------------------------------------------------------------ #

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

# ------------------------------------------------------------------------------------------------ #

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

# ------------------------------------------------------------------------------------------------ #

"""
Updated generalized generate_basic_approximations of Qiskit's Solovay-Kitaev theorem
Ref: from qiskit.synthesis.discrete_basis.generate_basis_approximations import generate_basic_approximations
Ref: https://en.wikipedia.org/wiki/Solovay%E2%80%93Kitaev_theorem
"""

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
          
# ------------------------------------------------------------------------------------------------ #

########################################################################################################################################################################################################

"""
Utilities
"""

# ------------------------------------------------------------------------------------------------ #

"""
Qiskit U3 gate as unitary matrix
"""

def qiskit_U3(theta, lamda, phi):
    mat = np.asarray(   [[np.cos(theta/2), -np.exp(1j*lamda)*np.sin(theta/2)],
                        [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(lamda+phi))*np.cos(theta/2)]])
    return mat

# ------------------------------------------------------------------------------------------------ #

def Cartesian_to_BlochRzRx(samples):
    rz_angle, rx_angle = [],[]
    for i in samples:
        sphe_coor = cartesian_to_spherical(i[0],i[1],i[2])
        rz_angle.append(sphe_coor[1].radian+math.pi/2)
        rx_angle.append(sphe_coor[2].radian)       
    return rz_angle, rx_angle

# ------------------------------------------------------------------------------------------------ #

def def_GS1(yaqq_dcmp):

    # ===> Define gateset GS1 (standard gates)    
    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    tdg_U_mat = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
    if yaqq_dcmp == 1:  # skt
        # h = HGate(label="h")          # legacy code
        # t = TGate(label="t")          # legacy code
        # tdg = TdgGate(label="tdg")    # legacy code
        Uh = UGate("Uh", h_U_mat)
        Ut = UGate("Ut", t_U_mat)
        Utdg = UGate("Utdg", tdg_U_mat)
        gs1 = [Uh, Ut, Utdg]
        gs1_gates = 'Uh,Ut,Utdg'
    else:               # rand
        gs1 = {}    
        gs1['H'] = UnitaryGate(h_U_mat,label='H')
        gs1['T'] = UnitaryGate(t_U_mat,label='T')
        gs1['Tdg'] = UnitaryGate(tdg_U_mat,label='Tdg')
        gs1_gates = ','.join(list(gs1.keys()))
    return gs1, gs1_gates

# ------------------------------------------------------------------------------------------------ #

def def_GS2_rand(yaqq_dcmp):

    # ===> Define gateset GS2 (random gates)    
    U1_mat = random_unitary(2).data
    U2_mat = random_unitary(2).data
    U3_mat = random_unitary(2).data    
    if yaqq_dcmp == 1:  # skt
        U1 = UGate("U1", U1_mat)
        U2 = UGate("U2", U2_mat)
        U3 = UGate("U3", U3_mat)
        gs2 = [U1, U2, U3]
        gs2_gates = 'U1,U2,U3'
    else:               # rand
        gs2 = {}   
        gs2['U1'] = UnitaryGate(U1_mat,label='U1') 
        gs2['U2'] = UnitaryGate(U2_mat,label='U2')
        gs2['U3'] = UnitaryGate(U3_mat,label='U3')
        gs2_gates = ','.join(list(gs2.keys()))
    return gs2, gs2_gates

# ------------------------------------------------------------------------------------------------ #

def def_GS2_param(thetas, yaqq_dcmp):

    # ===> Define gateset GS2 (parametric gates)      
    if yaqq_dcmp == 1:  # skt
        U3a = UGate("U3a", qiskit_U3(thetas[0], thetas[1], thetas[2]))
        U3b = UGate("U3b", qiskit_U3(thetas[3], thetas[4], thetas[5]))
        gs2 = [U3a, U3b]
    else:               # rand
        gs2 = {}   
        gs2['U3a'] = UnitaryGate(qiskit_U3(thetas[0], thetas[1], thetas[2]),label='U3a') 
        gs2['U3b'] = UnitaryGate(qiskit_U3(thetas[3], thetas[4], thetas[5]),label='U3b')
    return gs2

# ------------------------------------------------------------------------------------------------ #

########################################################################################################################################################################################################

"""
Mode 1: Compare GS2 (in code) w.r.t. GS1 (in code)
"""

# ------------------------------------------------------------------------------------------------ #

def compare_gs(ds, yaqq_dcmp):

    # ===> Define gateset GS1 (standard gates) 
    gs1, gs1_gates = def_GS1(yaqq_dcmp)

    if yaqq_dcmp == 1:  # skt
        gbs = gen_basis_seq()
        gseq1 = gbs.generate_basic_approximations(gs1)    # default 3 max_depth
        recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition 
        dcmp_skt1 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=gseq1)  # declare SKT object 

    # ===> Load U3 angles from novel_gs_scipy and Define gateset GS2
    path = os.path.realpath(__file__) # The path of this file
    dir = os.path.dirname(path)
    opt_angle_novel_gs = np.load(dir+'\\results\opt_ang.npy')
    if yaqq_dcmp == 1:  # skt
        Ua = UGate("Ua", np.array(qiskit_U3(opt_angle_novel_gs[3], opt_angle_novel_gs[4], opt_angle_novel_gs[5]), dtype=complex))
        Ub = UGate("Ub", np.array(qiskit_U3(opt_angle_novel_gs[0], opt_angle_novel_gs[1], opt_angle_novel_gs[2]), dtype=complex))
        gs2 = [Ua, Ub]
        gs2_gates = 'Ua,Ub'
        gseq2 = gbs.generate_basic_approximations(gs2)    # default 3 max_depth
        dcmp_skt2 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=gseq2) 
    else:               # rand
        gs2 = {}    
        gs2['U3a'] = UnitaryGate(qiskit_U3(opt_angle_novel_gs[0], opt_angle_novel_gs[1], opt_angle_novel_gs[2]),label='U3a')
        gs2['U3b'] = UnitaryGate(qiskit_U3( opt_angle_novel_gs[3], opt_angle_novel_gs[4], opt_angle_novel_gs[5]),label='U3b')
        gs2_gates = ','.join(list(gs2.keys()))

    # ===> Compare GS1 and GS2
    samples = len(ds)
    pf01_db, pf02_db = [], []
    dep01_db, dep02_db = [], []

    for i in tqdm(range(samples)):   
        qc0 = QuantumCircuit(1)
        qc0.append(ds[i], [0])
        if yaqq_dcmp == 1:          # Solovay-Kitaev Decomposition
            qc01 = dcmp_skt1(qc0)
            qc02 = dcmp_skt2(qc0)
        elif yaqq_dcmp == 2:        # Random Decomposition
            _, _, qc01 = dcmp_rand(qc0,gs1)    # default 100 trials and 100 max_depth
            _, _, qc02 = dcmp_rand(qc0,gs2)    # default 100 trials and 100 max_depth
        choi0 = Choi(qc0)
        choi01 = Choi(qc01)
        choi02 = Choi(qc02)
        pf01_db.append(process_fidelity(choi0,choi01))
        pf02_db.append(process_fidelity(choi0,choi02))
        dep01_db.append(qc01.depth())
        dep02_db.append(qc02.depth())

    # ===> Plot results
    plot_data = input("Plot data? [Y/N] (def.: Y): ") or 'Y'
    if plot_data == 'Y':
        # vis_dsU_gs(ds, pf01_db, dep01_db)
        plot_compare_gs(gs1_gates,gs2_gates,pf01_db,pf02_db,dep01_db,dep02_db)

########################################################################################################################################################################################################

"""
Mode 2: Generative novel GS2 w.r.t. GS1 (in code)
"""

# ------------------------------------------------------------------------------------------------ #

"""
Calculate cost function based on distribution of process fidelity differences and gate depths of two gate sets
"""

def cfn_calc(pf01_db,cd01_db,pf02_db,cd02_db):
    ivt_pf_gs01 = np.subtract(1,pf01_db)
    dist_pf_novelty = np.mean(abs(np.subtract(ivt_pf_gs01,pf02_db)))
    ivt_cd_gs01 = np.subtract(max(cd01_db),cd01_db)
    dist_cd_novelty = np.mean(abs(np.subtract(ivt_cd_gs01,cd02_db)))
    dist_pf_avg = - np.mean(pf02_db)
    dist_cd_avg = np.mean(cd02_db)
    w_pf_trend, w_cd_trend, w_pf_avg, w_cd_avg = 100, 100, 1000, 1
    cfn = w_pf_trend*dist_pf_novelty + w_cd_trend*dist_cd_novelty + w_pf_avg*dist_pf_avg + w_cd_avg*dist_cd_avg
    return cfn

# ------------------------------------------------------------------------------------------------ #

def generate_gs_random(ds, yaqq_dcmp, trials = 20):
    
    # ===> Define gateset GS1 (standard gates) 
    gs1, gs1_gates = def_GS1(yaqq_dcmp)

    if yaqq_dcmp == 1:  # skt
        gbs = gen_basis_seq()
        gseq1 = gbs.generate_basic_approximations(gs1)    # default 3 max_depth
        recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition 
        dcmp_skt1 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=gseq1)  # declare SKT object 

    # ===> Evaluate GS1 (standard gates) 
    samples = len(ds)
    pf01_db, dep01_db = [], []
    qc0_db, choi0_db = [], []
    for i in range(samples):     
        qc0 = QuantumCircuit(1)
        qc0.append(ds[i], [0])
        qc0_db.append(qc0)
        if yaqq_dcmp == 1:          # Solovay-Kitaev Decomposition
            qc01 = dcmp_skt1(qc0)
        elif yaqq_dcmp == 2:        # Random Decomposition
            _, _, qc01 = dcmp_rand(qc0,gs1)    # default 100 trials and 100 max_depth
        choi0 = Choi(qc0)
        choi0_db.append(choi0)
        choi01 = Choi(qc01)
        pf01_db.append(process_fidelity(choi0,choi01))
        dep01_db.append(qc01.depth())

    cfn_best, cfn_best_db = 100000, []
    for t in tqdm(range(trials)):
        
        # ===> Define gateset GS1 (standard gates) 
        gs2, gs2_gates = def_GS2_rand(yaqq_dcmp)

        # ===> Generate sequences and declare SKT object for GS2
        if yaqq_dcmp == 1:  # skt
            gseq2 = gbs.generate_basic_approximations(gs2)    # default 3 max_depth
            dcmp_skt2 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=gseq2)  # declare SKT object 
        
        # ===> Decompose data set with GS2
        pf02_db, dep02_db = [], []
        for i in range(samples):     

            if yaqq_dcmp == 1:          # Solovay-Kitaev Decomposition
                qc02 = dcmp_skt2(qc0_db[i])
            elif yaqq_dcmp == 2:        # Random Decomposition
                _, _, qc02 = dcmp_rand(qc0_db[i],gs2)    # default 100 trials and 100 max_depth
            choi02 = Choi(qc02)
            pf02_db.append(process_fidelity(choi0_db[i],choi02))
            dep02_db.append(qc02.depth())

        # ===> Evaluate GS2 based on cost function
        cfn = cfn_calc(pf01_db, dep01_db, pf02_db, dep02_db)
        if cfn <= cfn_best:
            cfn_best = cfn
            cfn_best_db = [gs2, pf02_db, dep02_db]   

    print("\nBest settings found:",cfn_best_db[0])
    
    # ===> Plot results
    plot_data = input("Plot data? [Y/N] (def.: Y): ") or 'Y'
    if plot_data == 'Y':
        plot_compare_gs(gs1_gates,gs2_gates,pf01_db,pf02_db,dep01_db,dep02_db,pfivt=True)

    return

# ------------------------------------------------------------------------------------------------ #

def eval_cfn(ds, qc0_db, choi0_db, pf01_db, cd01_db, thetas, yaqq_dcmp):
    
    gs2 = def_GS2_param(thetas, yaqq_dcmp)
    if yaqq_dcmp == 1:  # skt
        gbs = gen_basis_seq()
        gseq2 = gbs.generate_basic_approximations(gs2)    # default 3 max_depth
        recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition 
        dcmp_skt2 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=gseq2)  # declare SKT object 
    samples = len(ds)
    pf02_db, cd02_db = [], []
    for i in range(samples):  
        if yaqq_dcmp == 1:          # Solovay-Kitaev Decomposition
            qc02 = dcmp_skt2(qc0_db[i])
        elif yaqq_dcmp == 2:        # Random Decomposition
            _, _, qc02 = dcmp_rand(qc0_db[i],gs2)    # default 100 trials and 100 max_depth
        choi02 = Choi(qc02)
        pf02_db.append(process_fidelity(choi0_db[i],choi02))
        cd02_db.append(qc02.depth())
    cfn = cfn_calc(pf01_db, cd01_db, pf02_db, cd02_db)
    return cfn
    
# ------------------------------------------------------------------------------------------------ #

def generate_gs_optimize(ds, yaqq_dcmp, trials = 2, method = 'COBYLA', maxiter = 20):

    # ===> Define gateset GS1 (standard gates) 
    gs1, gs1_gates = def_GS1(yaqq_dcmp)

    if yaqq_dcmp == 1:  # skt
        gbs = gen_basis_seq()
        gseq1 = gbs.generate_basic_approximations(gs1)    # default 3 max_depth
        recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition 
        dcmp_skt1 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=gseq1)  # declare SKT object 

    # ===> Evaluate GS1 (standard gates) 
    samples = len(ds)
    pf01_db, cd01_db = [], []
    qc0_db, choi0_db = [], []
    for i in range(samples):     
        qc0 = QuantumCircuit(1)
        qc0.append(ds[i], [0])
        qc0_db.append(qc0)
        if yaqq_dcmp == 1:          # Solovay-Kitaev Decomposition
            qc01 = dcmp_skt1(qc0)
        elif yaqq_dcmp == 2:        # Random Decomposition
            _, _, qc01 = dcmp_rand(qc0,gs1)    # default 100 trials and 100 max_depth
        choi0 = Choi(qc0)
        choi0_db.append(choi0)
        choi01 = Choi(qc01)
        pf01_db.append(process_fidelity(choi0,choi01))
        cd01_db.append(qc01.depth())

    def cost_to_optimize(thetas):
        return eval_cfn(ds, qc0_db, choi0_db, pf01_db, cd01_db, thetas, yaqq_dcmp)
    
    cost_list, ang_list = [], []
    for t in tqdm(range(trials)):
        initial_guess = np.random.random(2*3)
        res = minimize(cost_to_optimize, initial_guess, method = method, options={'maxiter': maxiter})
        cost_list.append(res['fun'])
        ang_list.append(res['x'])
        # print('Cost of initial random guess of GS:', initial_cost)
        # print('Cost of optimized GS:', res.fun)

    best_GS = cost_list.index(np.min(cost_list))
    sort_opt_ang = ang_list[best_GS]
    # np.save('opt_ang_rand', sort_opt_ang)
    print("\nAngles of best GS found:",sort_opt_ang, "with cost",cost_list[best_GS])

    # ===> Evaluate GS2 (novel gates) 
    # TBD: Choose which method to use for decomposition, and if Rand, does not mean it will have same cost now, as inside SciPy
    U3a_mat = qiskit_U3(sort_opt_ang[0], sort_opt_ang[1], sort_opt_ang[2])
    U3b_mat = qiskit_U3(sort_opt_ang[3], sort_opt_ang[4], sort_opt_ang[5])
    gs2 = {}    
    gs2['u3a'] = UnitaryGate(U3a_mat,label='U3a')
    gs2['u3b'] = UnitaryGate(U3b_mat,label='U3b')
    gs2_gates = ','.join(list(gs2.keys()))
    pf02_db, cd02_db = [], []

    for i in range(samples):     
        pf02, cd02, qc01 = dcmp_rand(qc0_db[i],gs2)    # default 100 trials and 100 max_depth
        pf02_db.append(pf02)
        cd02_db.append(cd02)
    
    # ===> Plot results
    plot_data = input("Plot data? [Y/N] (def.: Y): ") or 'Y'
    if plot_data == 'Y':
        plot_compare_gs(gs1_gates,gs2_gates,pf01_db,pf02_db,cd01_db,cd02_db,pfivt=True) 

    return

# ------------------------------------------------------------------------------------------------ #

########################################################################################################################################################################################################

if __name__ == "__main__":

    devmode = input("\n  ===> Run Default Configuration? [Y/N] (def.: Y): ") or 'Y'

    if devmode == 'Y':
        # compare_gs(ds=gen_ds_fiboS(samples=10),yaqq_dcmp=1)
        # compare_gs(ds=gen_ds_fiboS(samples=10),yaqq_dcmp=2)
        # generate_gs_random(ds=gen_ds_fiboS(samples=2),yaqq_dcmp=1)
        # generate_gs_random(ds=gen_ds_randU(samples=2),yaqq_dcmp=2)
        generate_gs_optimize(ds=gen_ds_randU(samples=2),yaqq_dcmp=1)
        # generate_gs_optimize(ds=gen_ds_randU(samples=2),yaqq_dcmp=2)

    else:

        yaqq_ds_size = int(input("\n  ===> Enter Data Set Size (def.: 500): ") or 500)
            
        print("\n  ===> Choose Data Set:")
        print("   Data Set 1 - Evenly distributed states (using golden mean)")
        print("   Data Set 2 - Haar random states")
        print("   Data Set 3 - Haar random unitaries")      # https://iopscience.iop.org/article/10.1088/1367-2630/ac37c8/meta
        yaqq_ds_type = int(input("   Option (def.: 1): ") or 1)
        match yaqq_ds_type:
            case 1: 
                ds = gen_ds_fiboS(samples=yaqq_ds_size)  # Returns list of unitary gate objects as the preparation for the state vectors from |0>
            case 2: 
                ds = gen_ds_randS(samples=yaqq_ds_size)  # Returns list of unitary gate objects as the preparation for the state vectors from |0>
            case 3: 
                ds = gen_ds_randU(samples=yaqq_ds_size)  # Returns list of unitary gate objects
            case _ : 
                print("Invalid option")
                exit(1)   

        yaqq_ds_show = input("\n  ===> Visualize Data Set? [Y/N] (def.: N): ") or 'N'
        if yaqq_ds_show == 'Y':
            vis_ds_randU(ds)    # Plots the states when the unitaries are applied to |0> state    

        print("\n  ===> Choose Gate Decomposition Method:")
        print("   Method 1 - Solovay-Kitaev Decomposition")
        print("   Method 2 - Random Decomposition")
        yaqq_dcmp = int(input("   Option (def.: 2): ") or 2)

        print("\n  ===> Choose YAQQ Mode:")
        print("   Mode 1 - Compare GS2 (in code) w.r.t. GS1 (in code)")
        print("   Mode 2 - Generative novel GS2 w.r.t. GS1 (in code)")
        yaqq_mode = int(input("   Option (def.: 2): ")) or 2
        match yaqq_mode:
            case 1: 
                compare_gs(ds, yaqq_dcmp)
            case 2: 
                print("\n  ===> Choose Search Method:")
                print("   Method 1 - Random Gate Set Search")
                print("   Method 2 - U3 Angles Optimize with Multiple Random Initialization")
                yaqq_search = int(input("   Option (def.: 1): ")) or 1
                print()

                match yaqq_search:
                    case 1: 
                        generate_gs_random(ds, yaqq_dcmp)
                    case 2: 
                        generate_gs_optimize(ds, yaqq_dcmp)
                    case _: 
                        print("Invalid option")
                        exit(1)   
            case _ : 
                print("Invalid option")
                exit(1)   

    print("\nThank you for using YAQQ.")

########################################################################################################################################################################################################