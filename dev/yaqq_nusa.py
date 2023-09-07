import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary, Choi, process_fidelity
from qiskit.extensions import UnitaryGate
import random
from skt import gen_basis_seq, UGate, UdgGate
from qiskit.transpiler.passes.synthesis import SolovayKitaev

class NovelUniversalitySearchAgent:

    # ------------------------------------------------------------------------------------------------ #

    """
    Configure the decomposition methods for 1, 2 and 3+ qubit gates
    """

    def cnfg_dcmp(self, dcmp):
        self.d_1q = dcmp[0]    # Decomposition method for 1-qubit gates             [rand, skd]
        self.d_2q = dcmp[1]    # Decomposition method for 2-qubit gates             [rand, cartan]
        self.d_nq = dcmp[2]    # Decomposition method for 3 or more qubit gates     [rand, qsd]
        return
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Configure the weights of the elements of the cost function
    """

    def cnfg_wgts(self, wgts):
        self.w_apf = wgts[0]   # Weight of average process fidelity
        self.w_npf = wgts[1]   # Weight of novelty of process fidelity
        self.w_acd = wgts[2]   # Weight of average circuit depth
        self.w_ncd = wgts[3]   # Weight of novelty of circuit depth
        self.w_agf = wgts[4]   # Weight of average gate fabrication
        return

    # ------------------------------------------------------------------------------------------------ #

    """
    Define standard 1-qubit gateset: H, T, Tdg
    """

    def def_GS_standard_1q(self):
        h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
        tdg_U_mat = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
        gs = {}    
        gs['H'] = UnitaryGate(h_U_mat,label='H')
        gs['T'] = UnitaryGate(t_U_mat,label='T')
        gs['Tdg'] = UnitaryGate(tdg_U_mat,label='Tdg')
        gs_gates = ','.join(list(gs.keys()))
        return gs, gs_gates

    # ------------------------------------------------------------------------------------------------ #

    """
    Define standard 2-qubit gateset: H, T, CX
    """

    def def_GS_standard_2q(self):
        h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
        cx_U_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        gs = {}    
        gs['H'] = UnitaryGate(h_U_mat,label='H')
        gs['T'] = UnitaryGate(t_U_mat,label='T')
        gs['CX'] = UnitaryGate(cx_U_mat,label='CX')
        gs_gates = ','.join(list(gs.keys()))
        return gs, gs_gates
      
    # ------------------------------------------------------------------------------------------------ #

    """
    Given an unitary and a gate set, use trials to find a circuit using the gate set that is close to the unitary
    """

    def dcmp_rand(self, U, gs, trials, max_depth):

        choi0 = Choi(U)
        dim = int(np.log2(choi0.dim[0]))
        pfi_best = 0
        dep_best = 0
        qcirc_best = []
        for i in range(trials):
            dep = random.randrange(1,max_depth)
            seq = random.choices(list(gs.keys()), k=dep)
            qc0 = QuantumCircuit(dim)
            for seqid in seq:
                g_order = list(range(dim))
                random.shuffle(g_order)
                qc0.append(gs[seqid], g_order[:gs[seqid].num_qubits])
            choi01 = Choi(qc0)
            pfi = process_fidelity(choi0,choi01)
            if pfi > pfi_best:
                pfi_best = pfi
                qcirc_best = qc0
                dep_best = dep

        return pfi_best, dep_best, qcirc_best

    # ------------------------------------------------------------------------------------------------ #

    def skt_gs(self, gs):
        gs_skt = []
        for g in gs.keys():
            gs_skt.append(UGate(g,gs[g].to_matrix()))
        return gs_skt
    
    # ------------------------------------------------------------------------------------------------ #

    def dcmp_skt(self, U, skt_obj):

        choi0 = Choi(U)
        dim = int(np.log2(choi0.dim[0]))
        qc0 = QuantumCircuit(dim)
        qc0.append(U,list(range(dim)))
        qc01 = skt_obj(qc0)
        # choi0 = Choi(qc0)
        choi01 = Choi(qc01)
        pfi = process_fidelity(choi0,choi01)
        dep = qc01.depth()

        return pfi, dep, qc01

    # ------------------------------------------------------------------------------------------------ #

    def decompose_u(self):

        # Define Unitary to decompose here
        dim = 1
        U = UnitaryGate(random_unitary(2**dim),label='RndU')
        
        # Define Gate Set to decompose into here
        gs, _ = self.def_GS_standard_1q()

        # Decompose Unitary into Gate Set

        if dim > 2 and self.d_nq == 1:      # Random n-qubit decomposition
            # Decompose using gates in gs1, no further decompositions required (this is the least analytical decomposition)
            pfi, dep, qcirc = self.dcmp_rand(U, gs, trials = 500, max_depth = 500)
            print(pfi, dep)
            print(qcirc)
            pass
        elif dim > 2 and self.d_nq == 2:    # QSD decomposition
            # Analytically decompose using CX, Ry, Rz. Need to further decompose both 2-qubit and 1-qubit gates
            if self.d_2q == 1:                  # Random 2-qubit decomposition
                pass
            elif self.d_2q == 2:                # Cartan decomposition
                # Analytically decompose using CAN 2-qubit gates in gs1, need to further decompose 1-qubit gates
                if self.d_1q == 1:                  # Random 1-qubit decomposition
                    pass
                elif self.d_1q == 2:                # Solovay-Kitaev decomposition (this is the most analytical decomposition)
                    pass
            if self.d_1q == 1:                  # Random 1-qubit decomposition
                pass
            elif self.d_1q == 2:                # Solovay-Kitaev decomposition
                # convert gs to SKT UGate
                pass

        if dim == 2 and self.d_2q == 1:     # Random 2-qubit decomposition
            # Decompose using gates in gs1, no further decompositions required
            pfi, dep, qcirc = self.dcmp_rand(U, gs, trials = 500, max_depth = 500)
            print(pfi, dep)
            print(qcirc)
            pass
        elif dim == 2 and self.d_2q == 2:   # Cartan decomposition
            # Analytically decompose using CAN 2-qubit gates in gs1, need to further decompose 1-qubit gates
            if self.d_1q == 1:                  # Random 1-qubit decomposition
                pass
            elif self.d_1q == 2:                # Solovay-Kitaev decomposition
                pass

        if dim == 1 and self.d_1q == 1:     # Random 1-qubit decomposition
            pfi, dep, qcirc = self.dcmp_rand(U, gs, trials = 500, max_depth = 500)
            print(pfi, dep)
            print(qcirc)
        elif dim == 1 and self.d_1q == 2:   # Solovay-Kitaev decomposition
            gbs = gen_basis_seq()
            skt_obj = SolovayKitaev(recursion_degree = 3, basic_approximations = gbs.generate_basic_approximations(self.skt_gs(gs)))  # declare SKT object, larger recursion depth increases the accuracy and length of the decomposition
            pfi, dep, qcirc = self.dcmp_skt(U, skt_obj)
            print(pfi, dep)
            print(qcirc)
            
        return
    
    # ------------------------------------------------------------------------------------------------ #
    
    def compare_gs():
        return
    
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
    
    def nusa():
        return
    
    # ------------------------------------------------------------------------------------------------ #