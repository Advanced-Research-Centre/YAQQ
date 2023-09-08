import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary, Choi, process_fidelity
from qiskit.extensions import UnitaryGate
import random
from skt import gen_basis_seq, UGate, UdgGate
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from tqdm import tqdm
import time

class NovelUniversalitySearch:

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

    def def_gs(self, gs_cfg):

        gs = {}
        gno = 0
        for g in gs_cfg:
            gno += 1
            match g:
                case 'R1':      # R1: Haar Random 1-qubit Unitary
                    U = random_unitary(2).data
                case 'P1':      # P1: Parametric 1-qubit Unitary (IBM U3)
                    U = random_unitary(2).data  # TBD
                case 'G1':      # G1: Golden 1-qubit Unitary
                    U = random_unitary(2).data  # TBD
                case 'SG1':     # SG1: Super Golden 1-qubit Unitary
                    U = random_unitary(2).data  # TBD
                case 'T1':      # T1: T Gate 1-qubit Unitary
                    U = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
                case 'TD1':     # TD1: T-dagger Gate 1-qubit Unitary
                    U = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
                case 'H1':      # H1: H (Hadamard) Gate 1-qubit Unitary
                    U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
                case 'R2':      # R2: Haar Random 2-qubit Unitary
                    U = random_unitary(4).data
                case 'NL2':     # NL2: Non-local 2-qubit Unitary
                    U = random_unitary(2).data  # TBD
                case 'CX2':     # CX2: CNOT Gate 2-qubit Unitary
                    U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
                case 'B2':      # B2: B (Berkeley) Gate 2-qubit Unitary
                    U = random_unitary(2).data  # TBD
                case 'PE2':     # PE2: Perfect Entangler 2-qubit Unitary
                    U = random_unitary(2).data  # TBD
                case 'SPE2':    # SPE2: Special Perfect Entangler 2-qubit Unitary
                    U = random_unitary(2).data  # TBD
                case _:
                    print("Invalid Gate Set Configuration")
                    exit()
            gs[str(gno)+g] = UnitaryGate(U, label=str(gno)+g) 
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

    """
    Convert gate set from Unitary Gate to UGate for Solovay-Kitaev decomposition of Qiskit
    """

    def skt_gs(self, gs):

        gs_skt = []
        for g in gs.keys():
            gs_skt.append(UGate(g,gs[g].to_matrix()))

        return gs_skt
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Given an unitary and a gate set, use Qiskit's Solovay-Kitaev decomposition to find a circuit using the gate set that is close to the unitary
    """

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

    """
    Given an unitary and a gate set, decompose the unitary into the gate set based on the decomposition methods
    """

    def dcmp_U_gs(self, U, gs):
        
        dim = int(np.log2(Choi(U).dim[0]))
        rand_trials = 5
        rand_max_depth = 500

        if dim > 2 and self.d_nq == 1:      # Random n-qubit decomposition
            # Decompose using gates in gs1, no further decompositions required (this is the least analytical decomposition)
            pfi, dep, qcirc = self.dcmp_rand(U, gs, trials = rand_trials, max_depth = rand_max_depth)
        elif dim > 2 and self.d_nq == 2:    # QSD decomposition
            # Analytically decompose using CX, Ry, Rz. Need to further decompose both 2-qubit and 1-qubit gates
            if self.d_2q == 1:                  # Random 2-qubit decomposition
                pass  # TBD
            elif self.d_2q == 2:                # Cartan decomposition
                # Analytically decompose using CAN 2-qubit gates in gs1, need to further decompose 1-qubit gates
                if self.d_1q == 1:                  # Random 1-qubit decomposition
                    pass  # TBD
                elif self.d_1q == 2:                # Solovay-Kitaev decomposition (this is the most analytical decomposition)
                    pass  # TBD
            if self.d_1q == 1:                  # Random 1-qubit decomposition
                pass  # TBD
            elif self.d_1q == 2:                # Solovay-Kitaev decomposition
                # convert gs to SKT UGate
                pass  # TBD

        if dim == 2 and self.d_2q == 1:     # Random 2-qubit decomposition
            # Decompose using gates in gs1, no further decompositions required
            pfi, dep, qcirc = self.dcmp_rand(U, gs, trials = rand_trials, max_depth = rand_max_depth)
        elif dim == 2 and self.d_2q == 2:   # Cartan decomposition
            # Analytically decompose using CAN 2-qubit gates in gs1, need to further decompose 1-qubit gates
            if self.d_1q == 1:                  # Random 1-qubit decomposition
                pass  # TBD
            elif self.d_1q == 2:                # Solovay-Kitaev decomposition
                pass  # TBD

        if dim == 1 and self.d_1q == 1:     # Random 1-qubit decomposition
            pfi, dep, qcirc = self.dcmp_rand(U, gs, trials = rand_trials, max_depth = rand_max_depth)
        elif dim == 1 and self.d_1q == 2:   # Solovay-Kitaev decomposition
            gbs = gen_basis_seq()
            skt_obj = SolovayKitaev(recursion_degree = 3, basic_approximations = gbs.generate_basic_approximations(self.skt_gs(gs)))  # declare SKT object, larger recursion depth increases the accuracy and length of the decomposition
            pfi, dep, qcirc = self.dcmp_skt(U, skt_obj)

        return pfi, dep, qcirc

    # ------------------------------------------------------------------------------------------------ #

    def decompose_u(self):

        # Define Unitary to decompose here
        dim = 3
        U = UnitaryGate(random_unitary(2**dim),label='RndU')
        
        # Define Gate Set to decompose into here
        # gs, _ = self.def_gs(['H1','T1','TD1'])
        gs, _ = self.def_gs(['H1','T1','CX2'])
        
        # Decompose Unitary into Gate Set
        pf, cd, qc = self.dcmp_U_gs(U, gs)
        print(pf, cd)
        print(qc)        
            
        return
    
    # ------------------------------------------------------------------------------------------------ #
    
    def compare_gs(self, ds):

        # Define Gate Sets here (TBD: Load saved gate set from file)

        # gs1, gs1_gates = self.def_gs(['H1','T1','TD1'])
        # gs2, gs2_gates = self.def_gs(['R1','R1','R1'])

        gs1, gs1_gates = self.def_gs(['H1','T1','CX2'])
        gs2, gs2_gates = self.def_gs(['R1','R1','R2']) 

        # Decompose Unitaries from Data Set into both Gate Set

        samples = len(ds)
        pf01_db, cd01_db = [], []
        pf02_db, cd02_db = [], []
        print("\n  Decomposing Data Set into Gate Set 1:["+gs1_gates+"] and Gate Set 2:["+gs2_gates+"] \n")
        for i in tqdm(range(samples)):   
            pfi, dep, _ = self.dcmp_U_gs(ds[i], gs1)
            pf01_db.append(pfi)
            cd01_db.append(dep)
            pfi, dep, _ = self.dcmp_U_gs(ds[i], gs2)
            pf02_db.append(pfi)
            cd02_db.append(dep)

        return gs1, gs1_gates, pf01_db, cd01_db, gs2, gs2_gates, pf02_db, cd02_db

    # ------------------------------------------------------------------------------------------------ #

    def cnfg_cost_fab(self):

        self.cost_agf = 0
        # Use this method to set fabrication cost of gates in the gate set

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
        
        self.cnfg_cost_fab()

        return
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Calculate cost function based on distribution of process fidelity differences and gate depths of two gate sets
    """

    def cfn_calc(self,pf01_db,cd01_db,pf02_db,cd02_db):

        ivt_pf_gs01 = np.subtract(1,pf01_db)
        dist_pf_novelty = np.mean(abs(np.subtract(ivt_pf_gs01,pf02_db)))
        ivt_cd_gs01 = np.subtract(max(cd01_db),cd01_db)
        dist_cd_novelty = np.mean(abs(np.subtract(ivt_cd_gs01,cd02_db)))
        dist_pf_avg = - np.mean(pf02_db)
        dist_cd_avg = np.mean(cd02_db)

        cfn = self.w_apf*dist_pf_avg + self.w_npf*dist_pf_novelty + self.w_acd*dist_cd_avg + self.w_ncd*dist_cd_novelty + self.w_agf*self.cost_agf

        return cfn

    # ------------------------------------------------------------------------------------------------ #
    
    def nusa(self, ds, ngs_cfg, search):

        # Define Gate Set 1 here (TBD: Load saved gate set from file)
        gs1, gs1_gates = self.def_gs(['H1','T1','TD1'])

        # Decompose Unitaries from Data Set into Gate Set 1
        samples = len(ds)
        pf01_db, cd01_db = [], []
        print("\n  Decomposing Data Set into Gate Set 1:["+gs1_gates+"] \n")
        for i in tqdm(range(samples)):   
            pf, cd, _ = self.dcmp_U_gs(ds[i], gs1)
            pf01_db.append(pf)
            cd01_db.append(cd)

        # Constraints: time, cost, trials
        cfn_best, cfn_best_db = np.inf, []
        max_time = 10
        trials = 0
        start = time.time()
        end = time.time()

        while (end - start) < max_time:
            # Define Gate Set 2 here
            gs2, gs2_gates = self.def_gs(ngs_cfg) 
            trials += 1
            print("\n  Decomposing Data Set into Gate Set 2:["+gs2_gates+"], trials = "+str(trials)+"\n") 
            pf02_db, cd02_db = [], []
            for i in range(samples):   
                pf, cd, _ = self.dcmp_U_gs(ds[i], gs1)
                pf02_db.append(pf)
                cd02_db.append(cd)
            cfn = self.cfn_calc(pf01_db, cd01_db, pf02_db, cd02_db)
            if cfn <= cfn_best:
                cfn_best = cfn
                cfn_best_db = [gs2, gs2_gates, pf02_db, cd02_db] 
            end = time.time()

        return gs1, gs1_gates, pf01_db, cd01_db, cfn_best_db[0], cfn_best_db[1], cfn_best_db[2], cfn_best_db[3]

    # ------------------------------------------------------------------------------------------------ #