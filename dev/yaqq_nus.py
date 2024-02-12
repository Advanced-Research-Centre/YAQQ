import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary, Choi, process_fidelity, Operator
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.synthesis import qsd
import random
from skt import gen_basis_seq, UGate, UdgGate
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from tqdm import tqdm
import time
import weylchamber
from scipy.optimize import minimize 
from qiskit.quantum_info import TwoQubitBasisDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
import warnings
from datetime import date
import configparser
import json

class NovelUniversalitySearch:

    # ------------------------------------------------------------------------------------------------ #

    """
    Configure the decomposition methods for 1, 2 and 3+ qubit gates
    """

    def cnfg_dcmp(self, dcmp_gs1, dcmp_gs2, autocfg, Config = None):
       
        self.d_1q = [dcmp_gs1[0]]             # Decomposition method for 1-qubit gates             [rand, skd]
        self.d_2q = [dcmp_gs1[1]]             # Decomposition method for 2-qubit gates             [rand, cartan]
        self.d_nq = [dcmp_gs1[2]]             # Decomposition method for 3 or more qubit gates     [rand, qsd]

        if len(dcmp_gs2) > 0:
            self.d_1q.append(dcmp_gs2[0])   # Decomposition method for 1-qubit gates             [rand, skd]
            self.d_2q.append(dcmp_gs2[1])   # Decomposition method for 2-qubit gates             [rand, cartan]
            self.d_nq.append(dcmp_gs2[2])   # Decomposition method for 3 or more qubit gates     [rand, qsd]           

        if autocfg and Config.has_option('experiment', 'skt_param'):
            self.skt_param = json.loads(Config['experiment']['skt_param'])
        else:
            self.skt_param = [3,3]

        if autocfg and Config.has_option('experiment', 'rnd_param'):
            self.rnd_param = json.loads(Config['experiment']['rnd_param'])
        else:
            self.rnd_param = [100,500]

        return
      
    # ------------------------------------------------------------------------------------------------ #

    def def_gs(self, gs_cfg, params = None):

        # All params are normalized to [0,1]
        gs = {}
        gno = 0
        param_ctr = 0
        for g in gs_cfg:
            gno += 1
            match g:
                case 'R1':      # R1: Haar Random 1-qubit Unitary
                    U = random_unitary(2).data
                case 'P1':      # P1: Parametric 1-qubit Unitary (Qiskit U3)
                    theta, phi, lam = params[param_ctr]*np.pi, params[param_ctr+1]*np.pi, params[param_ctr+2]*np.pi
                    U = np.asarray( [[np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
                                    [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(lam+phi))*np.cos(theta/2)]])
                    param_ctr += 3
                case 'G1':      # G1: Golden 1-qubit Unitary
                    U = random_unitary(2).data  # TBD Extension
                case 'SG1':     # SG1: Super Golden 1-qubit Unitary
                    # Ref: https://arxiv.org/abs/1704.02106
                    U = random_unitary(2).data  # TBD Extension
                case 'T1':      # T1: T Gate 1-qubit Unitary
                    U = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
                case 'X1':      # X1: X (Pauli-X) Gate 1-qubit Unitary
                    U = np.array([[0, 1], [1, 0]], dtype=complex)
                case 'Y1':      # Y1: Y (Pauli-Y) Gate 1-qubit Unitary
                    U = np.array([[0, -1j], [1j, 0]], dtype=complex)
                case 'Z1':      # Z1: Z (Pauli-Z) Gate 1-qubit Unitary
                    U = np.array([[1, 0], [0, -1]], dtype=complex)  
                case 'S1':      # S1: S Gate 1-qubit Unitary
                    U = np.array([[1, 0], [0, np.exp(1j*np.pi/2)]], dtype=complex)
                case 'M1':      # M1: Matt's pi/3 phase Gate 1-qubit Unitary for HQECC
                    U = np.array([[1, 0], [0, np.exp(1j*np.pi/3)]], dtype=complex)
                case 'TD1':     # TD1: T-dagger Gate 1-qubit Unitary
                    U = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
                case 'H1':      # H1: H (Hadamard) Gate 1-qubit Unitary
                    U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
                case 'F1':      # F1: Load 1-qubit Unitary Gate definition from File
                    fname = input("\n  ===> Enter Gate Configuration Filename: ")
                    U = np.load(fname, allow_pickle=True)
                case 'R2':      # R2: Haar Random 2-qubit Unitary
                    U = random_unitary(4).data
                case 'NL2':     # NL2: Non-local 2-qubit Unitary
                    U = weylchamber.canonical_gate(params[param_ctr],params[param_ctr+1],params[param_ctr+2])
                    param_ctr += 3
                case 'CX2':     # CX2: CNOT Gate 2-qubit Unitary
                    U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
                case 'B2':      # B2: B (Berkeley) Gate 2-qubit Unitary
                    U = weylchamber.canonical_gate(0.5,0.25,0)
                case 'PE2':     # PE2: Perfect Entangler 2-qubit Unitary
                    U = random_unitary(2).data  # TBD Extension
                case 'SPE2':    # SPE2: Special Perfect Entangler 2-qubit Unitary
                    U = weylchamber.canonical_gate(0.5,params[param_ctr],0)
                    param_ctr += 1
                case 'F2':      # F2: Load 2-qubit Unitary Gate definition from File
                    # ask user for file name
                    pass        # TBD Extension
                case _:
                    print("Invalid Gate Set Configuration")
                    exit()
            gs[str(gno)+g] = UnitaryGate(U, label=str(gno)+g) 
        gs_gates = ','.join(list(gs.keys()))

        return gs, gs_gates
    # ------------------------------------------------------------------------------------------------ #

    def gs_param_ctr(self, gs_cfg):

        param_ctr = 0
        for g in gs_cfg:
            match g:
                case 'P1':      # P1: Parametric 1-qubit Unitary (Qiskit U3)
                    param_ctr += 3
                case 'G1':      # G1: Golden 1-qubit Unitary
                    param_ctr += 0  # TBD Extension
                case 'SG1':     # SG1: Super Golden 1-qubit Unitary
                    param_ctr += 0  # TBD Extension
                case 'NL2':     # NL2: Non-local 2-qubit Unitary
                    param_ctr += 3
                case 'PE2':     # PE2: Perfect Entangler 2-qubit Unitary
                    param_ctr += 0  # TBD Extension
                case 'SPE2':    # SPE2: Special Perfect Entangler 2-qubit Unitary
                    param_ctr += 1

        return param_ctr
    
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
                if gs[seqid].num_qubits <= dim:
                    g_order = list(range(dim))
                    random.shuffle(g_order)
                    qc0.append(gs[seqid], g_order[:gs[seqid].num_qubits])
            choi01 = Choi(qc0)
            pfi = process_fidelity(choi0,choi01)
            if pfi > pfi_best:   # TBD: Should we consider the circuit with the highest fidelity or the one with the lowest depth?
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
            if gs[g].num_qubits == 1:
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
    Find a basis gate to use from the gate set for KAK decomposition
    """

    def kak_gs(self, gs):
        
        gnq = []
        for g in gs.keys():
            gnq.append(gs[g].num_qubits)

        if 2 not in gnq:
            print("No 2-qubit gate in the gate set. Cartan decomposition using 1-qubit gates not available.")
            # TBD Extension: Use 1-qubit gates to construct 2-qubit gates
            exit()
        elif gnq.count(2) > 1:
            print("Multiple choices of 2-qubit gate in the gate set. Cartan decomposition using multiple 2-qubit gates not available.")
            # TBD Extension: Use multiple 2-qubit gates
            exit()
        else:
            bg = gs[list(gs.keys())[gnq.index(2)]]

        return bg


    # ------------------------------------------------------------------------------------------------ #

    """
    Given an unitary and a gate set, decompose the unitary into the gate set based on the decomposition methods
    """

    def dcmp_U_gs(self, U, gs, gsid = 0):
        
        dim = int(np.log2(Choi(U).dim[0]))
        rand_trials = self.rnd_param[0]
        rand_max_depth = self.rnd_param[1]
        skt_recursion_degree = self.skt_param[0]
        skt_gbs_depth = self.skt_param[1]

        if dim > 2 and self.d_nq[gsid] == 1:      # Random n-qubit decomposition
            # Decompose using gates in gs1, no further decompositions required
            pf, cd, qc = self.dcmp_rand(U, gs, trials = rand_trials, max_depth = rand_max_depth)
        elif dim > 2 and self.d_nq[gsid] == 2:    # QSD decomposition
            # Need to further decompose both 2-qubit and 1-qubit gates
            qsd_circ = qsd.qs_decomposition(U.to_matrix())
            ds_qsd_1q, ds_qsd_2q = [], []
            for g in qsd_circ:
                if g.operation.num_qubits == 1:
                    ds_qsd_1q.append(UnitaryGate(Operator(g.operation),label='QSD_1q'))
                elif g.operation.num_qubits == 2:
                    ds_qsd_2q.append(UnitaryGate(Operator(g.operation),label='QSD_2q'))
            # Decompose 2 qubits gates in ds_qsd_2q           
            ds_qsd_2q_gs = []
            for U_qsd_2q in ds_qsd_2q:
                _, _, qcirc_QSD_2q = self.dcmp_U_gs(U_qsd_2q, gs, gsid)
                ds_qsd_2q_gs.append(qcirc_QSD_2q)
            # Decompose 1 qubit gates in ds_qsd_1q
            ds_qsd_1q_gs = []
            for U_qsd_1q in ds_qsd_1q:
                _, _, qcirc_QSD_1q = self.dcmp_U_gs(U_qsd_1q, gs, gsid)
                ds_qsd_1q_gs.append(qcirc_QSD_1q)
            # Replace the gates in QSD with decomposed circuits
            qc = QuantumCircuit(dim)
            for g in qsd_circ:
                if g.operation.num_qubits == 1:
                    U_gs = ds_qsd_1q_gs.pop(0)
                    tgt = [qsd_circ.find_bit(x).index for x in g.qubits]
                    for g_gs in U_gs:
                        qc.append(g_gs.operation, tgt)
                elif g.operation.num_qubits == 2:
                    U_gs = ds_qsd_2q_gs.pop(0)
                    tgt = [qsd_circ.find_bit(x).index for x in g.qubits]
                    for g_gs in U_gs:
                        if len(g_gs.qubits) == 1:
                            qc.append(g_gs.operation, [tgt[g_gs.qubits[0].index]])
                        elif len(g_gs.qubits) == 2:
                            qc.append(g_gs.operation, tgt)
                qc.barrier()
            pf = process_fidelity(Choi(U),Choi(qc))
            cd = qc.depth()

        if dim == 2 and self.d_2q[gsid] == 1:     # Random 2-qubit decomposition
            # Decompose using gates in gs1, no further decompositions required
            pf, cd, qc = self.dcmp_rand(U, gs, trials = rand_trials, max_depth = rand_max_depth)
        elif dim == 2 and self.d_2q[gsid] == 2:   # Cartan decomposition
            # Analytically decompose using CAN 2-qubit gates in gs1, need to further decompose 1-qubit gates
            bg = self.kak_gs(gs)
            # warnings.filterwarnings("ignore", category=UserWarning) # Ignore warning of non-perfect decomposition for non-supercontrolled gates
            kak_obj = TwoQubitBasisDecomposer(bg)
            ds_kak_1q = kak_obj.decomp3_supercontrolled(TwoQubitWeylDecomposition(U))
            # Decompose 1 qubit gates in ds_kak_1q
            ds_kak_1q_gs = []
            for U_kak_1q in ds_kak_1q:
                _, _, qcirc_KAK_1q = self.dcmp_U_gs(UnitaryGate(Operator(U_kak_1q),label='KAK_1q'), gs, gsid)
                ds_kak_1q_gs.append(qcirc_KAK_1q)
            # Replace the gates in KAK with decomposed circuits
            qc = QuantumCircuit(2)
            for g_gs in ds_kak_1q_gs[0]:
                qc.append(g_gs.operation, [0])
            for g_gs in ds_kak_1q_gs[1]:
                qc.append(g_gs.operation, [1])
            qc.append(bg,[0,1])
            for g_gs in ds_kak_1q_gs[2]:
                qc.append(g_gs.operation, [0])
            for g_gs in ds_kak_1q_gs[3]:
                qc.append(g_gs.operation, [1])
            qc.append(bg,[0,1])
            for g_gs in ds_kak_1q_gs[4]:
                qc.append(g_gs.operation, [0])
            for g_gs in ds_kak_1q_gs[5]:
                qc.append(g_gs.operation, [1])
            qc.append(bg,[0,1])
            for g_gs in ds_kak_1q_gs[6]:
                qc.append(g_gs.operation, [0])
            for g_gs in ds_kak_1q_gs[7]:
                qc.append(g_gs.operation, [1])
            pf = process_fidelity(Choi(U),Choi(qc))
            cd = qc.depth()

        if dim == 1 and self.d_1q[gsid] == 1:     # Random 1-qubit decomposition
            pf, cd, qc = self.dcmp_rand(U, gs, trials = rand_trials, max_depth = rand_max_depth)
        elif dim == 1 and self.d_1q[gsid] == 2:   # Solovay-Kitaev decomposition
            gbs = gen_basis_seq()
            skt_obj = SolovayKitaev(recursion_degree = skt_recursion_degree, basic_approximations = gbs.generate_basic_approximations(self.skt_gs(gs),depth=skt_gbs_depth))  # declare SKT object, larger recursion depth increases the accuracy and length of the decomposition
            pf, cd, qc = self.dcmp_skt(U, skt_obj)

        return pf, cd, qc

    # ------------------------------------------------------------------------------------------------ #

    def decompose_u(self, autocfg, Config = None):

        # Define Unitary to decompose here
        if autocfg:
            yaqq_ds_dim = int(Config['mode3']['yaqq_ds_dim'])
            if Config['mode3']['u_type'] == 'file':
                U_fname = Config['mode3']['u_fname']
                np_U = np.load('results/data/'+U_fname+'.npy', allow_pickle=True)
                if np_U.shape[0] != 2**yaqq_ds_dim:
                    print("Invalid Unitary Dimension")
                    exit()
                U = UnitaryGate(np_U,label='UsrU')
            else:
                print("Generating Haar random unitary of given dimension") 
                np_U = np.array(random_unitary(2**yaqq_ds_dim))
                print(np_U)
                if Config['mode3']['u_save'] == 'Y':
                    U_fname = Config['mode3']['u_fname']
                    np.save('results/data/'+U_fname+'.npy',np_U)
                U = UnitaryGate(np_U,label='RndU')
        else:
            yaqq_ds_dim = int(input("\n  ===> Enter Data Set Dimension (def.: 1): ") or 1)        
            load_U = input("\n  ===> Load Unitary to Decompose from File? [Y/N] (def.: N): ") or 'N'
            if load_U == 'Y':
                U_fname = input("\n  ===> Enter filename (def.: <enter>): ") or 'dcmp_U'
                np_U = np.load('results/data/'+U_fname+'.npy', allow_pickle=True)
                if np_U.shape[0] != 2**yaqq_ds_dim:
                    print("Invalid Unitary Dimension")
                    exit()
                U = UnitaryGate(np_U,label='UsrU')
            else:
                print("Generating Haar random unitary of given dimension") 
                np_U = np.array(random_unitary(2**yaqq_ds_dim))
                print(np_U)
                save_U = input("\n  ===> Save Generated Unitary? [Y/N] (def.: N): ") or 'N'
                if save_U == 'Y':
                    U_fname = input("\n  ===> Enter filename (def.: <enter>): ") or 'dcmp_U'
                    np.save('results/data/'+U_fname+'.npy',np_U)
                U = UnitaryGate(np_U,label='RndU')
       
        # Define Gate Set to decompose into here
        if autocfg:
            yaqq_cf_ngs = Config['mode3']['yaqq_cf_ngs'].split(',')
        else:
            print("\n Gate Set Composition:")                   # TBD: Currently only constant gates allowed
            print("   X1: X (Pauli-X) Gate 1-qubit Unitary")  
            print("   Y1: Y (Pauli-Y) Gate 1-qubit Unitary")  
            print("   Z1: Z (Pauli-Z) Gate 1-qubit Unitary")  
            print("   T1: T Gate 1-qubit Unitary")            
            print("   TD1: T-dagger Gate 1-qubit Unitary")    
            print("   H1: H (Hadamard) Gate 1-qubit Unitary") 
            print("   S1: S Gate 1-qubit Unitary") 
            print("   M1: Phase pi/3 Gate 1-qubit Unitary")     # for Matt's HQECC experiment
            if yaqq_ds_dim >= 2:
                print("   CX2: CNOT Gate 2-qubit Unitary")             
                print("   B2: B (Berkeley) Gate 2-qubit Unitary")      
            yaqq_cf_ngs = (input("\n  ===> Enter Gate Set (def.: [H1,T1,TD1]): ") or 'H1,T1,TD1').split(',')

        gs, gs_gates = self.def_gs(yaqq_cf_ngs)

        # Decompose Unitary into Gate Set
        pf, cd, qc = self.dcmp_U_gs(U, gs, gsid = 0)
        print("\nProcess Fidelity:",pf)
        print("Circuit Depth: ", cd)

        if autocfg:
            show_qc = Config['mode3']['show_qc']
        else:
            show_qc = input("\n  ===> Show Decomposed Quantum Circuit? [Y/N] (def.: N): ") or 'N'
        if show_qc == 'Y':
            print(qc)     

        if autocfg:
            save_qc = Config['mode3']['save_qc']
        else:
            save_qc = input("\n  ===> Save Decomposed Quantum Circuit? [Y/N] (def.: Y): ") or 'Y'
        if save_qc == 'Y':
            if autocfg:
                qc_fname = Config['mode3']['qc_fname']
            else:
                qc_fname = input("\n  ===> Enter filename (def.: <enter>): ") or 'qc_'+str(date.today())+'_'+U.label+'_'+gs_gates.replace(',','-')
            with open('results/data/'+qc_fname+'.txt', 'w') as f:
                for i in qc:
                    f.write(i.operation.label+' '+str(i.qubits[0].index)+'\n')

        return
    
    # ------------------------------------------------------------------------------------------------ #
    
    def compare_gs(self, ds, autocfg, Config = None):

        # Define Gate Sets here
        if autocfg:
            yaqq_cf_gs1 = Config['mode2']['yaqq_cf_gs1'].split(',')
            yaqq_cf_gs2 = Config['mode2']['yaqq_cf_gs2'].split(',')
            print(yaqq_cf_gs1, yaqq_cf_gs2)
            gs1, gs1_gates = self.def_gs(yaqq_cf_gs1)
            gs2, gs2_gates = self.def_gs(yaqq_cf_gs2)
        else:
            gs1, gs1_gates = self.def_gs(['H1','T1']) 
            gs2, gs2_gates = self.def_gs(['H1','T1'])
            # gs1, gs1_gates = self.def_gs(['H1','T1','TD1'])
            # gs2, gs2_gates = self.def_gs(['R1','R1','R1'])
            # gs1, gs1_gates = self.def_gs(['H1','T1','CX2'])
            # gs2, gs2_gates = self.def_gs(['R1','R1','R2']) 
            # gs1, gs1_gates = self.def_gs(['H1','T1','CX2'])         # TBD: Take from user input
            # gs2, gs2_gates = self.def_gs(['R1','R1','CX2'])         # TBD: Take from user input

        # Decompose Unitaries from Data Set into both Gate Set
        samples = len(ds)
        pf01_db, cd01_db = [], []
        pf02_db, cd02_db = [], []
        print("\n  Decomposing Data Set into Gate Set 1:["+gs1_gates+"] and Gate Set 2:["+gs2_gates+"] \n")
        for i in tqdm(range(samples)):   
            pfi, dep, _ = self.dcmp_U_gs(ds[i], gs1, gsid = 0)
            pf01_db.append(pfi)
            cd01_db.append(dep)
            pfi, dep, _ = self.dcmp_U_gs(ds[i], gs2, gsid = 1)
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
        dist_pf_avg = - np.mean(pf02_db) # -ve sign means it is maximizing
        dist_cd_avg = np.mean(cd02_db)

        cfn = self.w_apf*dist_pf_avg + self.w_npf*dist_pf_novelty + self.w_acd*dist_cd_avg + self.w_ncd*dist_cd_novelty + self.w_agf*self.cost_agf

        return cfn

    # ------------------------------------------------------------------------------------------------ #
    
    def nusa(self, ds, ngs_cfg, optimize, autocfg, Config = None):

        # Define Gate Set 1 here
        if autocfg:
            yaqq_cf_gs1 = Config['mode1']['yaqq_cf_gs1'].split(',')
            gs1, gs1_gates = self.def_gs(yaqq_cf_gs1)                   # TBD: Check that it should not have parametric gates
        else:
            gs1, gs1_gates = self.def_gs(['H1','T1','CX2'])

        # Decompose Unitaries from Data Set into Gate Set 1
        samples = len(ds)
        pf01_db, cd01_db = [], []
        print("\n  Decomposing Data Set into Gate Set 1:["+gs1_gates+"] \n")
        with tqdm(range(samples)) as t:
            for i in t:   
                pf, cd, _ = self.dcmp_U_gs(ds[i], gs1, gsid = 0)
                pf01_db.append(pf)
                cd01_db.append(cd)
            t_1gs = t.format_dict['elapsed']

        # Cost function for optimization
        if autocfg:
            maxiter = int(Config['mode1']['opt_max_iter'])
        else:
            maxiter = 500 # 500 for final experiment
        method = 'COBYLA'
        trials = 0
        db_gs2_NL2 = []
        def cost_to_optimize(gs2_params):
            gs2, _ = self.def_gs(ngs_cfg, gs2_params) 
            pf02_db, cd02_db = [], []
            for i in range(samples):   
                pf, cd, _ = self.dcmp_U_gs(ds[i], gs2, gsid = 1)
                pf02_db.append(pf)
                cd02_db.append(cd)
            cfn = self.cfn_calc(pf01_db, cd01_db, pf02_db, cd02_db)
            db_gs2_NL2.append([trials,gs2,cfn])
            return cfn

        # Optimize Gate Set 2
        param_ctr = self.gs_param_ctr(ngs_cfg)
        cfn_best, cfn_best_db = np.inf, []
        if autocfg:
            max_time = int(Config['mode1']['nus_max_time'])
        else:
            max_time = 500       # in seconds
        start = time.time()
        end = time.time()

        while (end - start) < max_time:     # Constraints: time/cost/trials
            trials += 1
            print("\n  Decomposing Data Set into Gate Set 2, trials = "+str(trials)+"\n") 
            params = np.random.rand(param_ctr)
            if optimize == 'Y':     # SciPy optimize
                print("  Estimated max. run time for 1 optimization trial = "+str(t_1gs*maxiter)+"\n") 
                res = minimize(cost_to_optimize, params, method = method, options={'maxiter': maxiter})
                if res['fun'] <= cfn_best:
                    cfn_best = res['fun']
                    gs2, gs2_gates = self.def_gs(ngs_cfg, res['x'])
                    pf02_db, cd02_db = [], []
                    for i in range(samples):   
                        pf, cd, _ = self.dcmp_U_gs(ds[i], gs2, gsid = 1)
                        pf02_db.append(pf)
                        cd02_db.append(cd)
                    cfn_best_db = [gs2, gs2_gates, pf02_db, cd02_db, res['x']] 
                    break
            else:                   # Random search
                gs2, gs2_gates = self.def_gs(ngs_cfg, params) 
                pf02_db, cd02_db = [], []
                for i in range(samples):   
                    pf, cd, _ = self.dcmp_U_gs(ds[i], gs2, gsid = 1)
                    pf02_db.append(pf)
                    cd02_db.append(cd)
                cfn = self.cfn_calc(pf01_db, cd01_db, pf02_db, cd02_db)
                if cfn <= cfn_best:
                    cfn_best = cfn
                    cfn_best_db = [gs2, gs2_gates, pf02_db, cd02_db, params] 
            end = time.time()

        return gs1, gs1_gates, pf01_db, cd01_db, cfn_best_db[0], cfn_best_db[1], cfn_best_db[2], cfn_best_db[3], cfn_best_db[4]

    # ------------------------------------------------------------------------------------------------ #