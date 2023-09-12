# For generation
    # take a set of Haar random 2 qubit unitary
    # do Cartan decomposition
    # save distribution of 15 parameters
    # sample from this distribution to generate/optimize gate set
# For decomposition
    # take a set of Haar random 2 qubit unitary
    # take random 2 single qubit and 1 Weyl non-local 2 qubit unitary
    # do random decomposition



from qiskit.circuit.library import XGate,ZGate,YGate
from qiskit.quantum_info import Pauli, Choi, process_fidelity, Operator
from scipy.optimize import minimize  
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import random
# from math import pi
import numpy as np

from tqdm import tqdm
import scipy as sp
import weylchamber
from itertools import product

from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import random_unitary

def qiskit_U3(theta, lamda, phi):
    mat = np.asarray(   [[np.cos(theta/2), -np.exp(1j*lamda)*np.sin(theta/2)],
                        [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(lamda+phi))*np.cos(theta/2)]])
    return mat

# Ref: https://threeplusone.com/pubs/on_gates.pdf
def Can(tx, ty, tz):
    # SWAP
    # tx = 0.5  # [0,1]
    # ty = 0.5  # [0,0.5]
    # tz = 0.5  # [0]

    
    # tx = 0.5  # [0,1]
    # ty = 0  # [0,0.5]
    # tz = 0  # [0]

    # print(XGate().to_matrix())
    S = tx*Pauli('XX').to_matrix() + ty*Pauli('YY').to_matrix() + tz*Pauli('ZZ').to_matrix()
    can = sp.linalg.expm(-1j*S*np.pi/2)
    # gp = can * np.exp(1j*np.pi/4)       # * np.sqrt(2)
    # gp = can * np.exp(1j*np.pi/2)       # * np.sqrt(2)
    # print(S)
    # print(np.matrix(can).getH() * np.matrix(can))
    # print()
    # print(np.round(gp,3))
    # print(xx.to_matrix())
    return can



def rand_2q_CAN_single_unitary_decomp(U, random_trail):

    choi0 = Choi(UnitaryGate(U))
    pfi_best = 0
    for i in range(random_trail):
        list_ang = np.random.random(12)*2*np.pi
        t = np.random.random(3)
        tx, ty, tz = t[0], t[1], t[2]
        # print(list_ang)
        U31, U32 = qiskit_U3(list_ang[0], list_ang[1], list_ang[2]), qiskit_U3(list_ang[3], list_ang[4], list_ang[5]),
        U33, U34 = qiskit_U3(list_ang[6], list_ang[7], list_ang[8]),qiskit_U3(list_ang[9], list_ang[10], list_ang[11])

        majhe_eso = Can(tx,ty,tz)
        peeche_hat = np.kron(U31, U32)
        egia_cholo = np.kron(U33, U34)
        prodhan_gate = np.matmul(egia_cholo, np.matmul(majhe_eso, peeche_hat))

        choi01 = Choi(UnitaryGate(prodhan_gate))
        pfi = process_fidelity(choi0,choi01)

        
        if pfi > pfi_best:
            pfi_best = pfi
            qcirc_best_list = [U31, U32, majhe_eso, U33, U34]
            print(i, pfi_best)
    return pfi_best, qcirc_best_list

def rand_2q_CAN_multi_unitary_decomp(U, random_trail):

    """
    KAJ HOINI
    """

    choi0 = Choi(UnitaryGate(U))
    pfi_best = 0
    for i in range(random_trail):
        list_ang = np.random.random(12)*2*np.pi
        t = np.random.random(3)
        tx, ty, tz = t[0], t[1], t[2]
        # print(list_ang)
        U31, U32 = qiskit_U3(list_ang[0], list_ang[1], list_ang[2]), qiskit_U3(list_ang[3], list_ang[4], list_ang[5]),
        U33, U34 = qiskit_U3(list_ang[6], list_ang[7], list_ang[8]),qiskit_U3(list_ang[9], list_ang[10], list_ang[11])

        majhe_eso = Can(tx,ty,tz)
        peeche_hat = np.kron(U31, U32)
        egia_cholo = np.kron(U33, U34)
        prodhan_gate = np.matmul(egia_cholo, np.matmul(majhe_eso, peeche_hat))

        choi01 = Choi(UnitaryGate(prodhan_gate))
        pfi = process_fidelity(choi0,choi01)

        
        if pfi > pfi_best:
            pfi_best = pfi
            qcirc_best_list = [U31, U32, majhe_eso, U33, U34]
            print(i, pfi_best)
    return pfi_best, qcirc_best_list



def scipy_2q_decomp_single_rand_unitary(U, random_trail):

    choi0 = Choi(UnitaryGate(U))

    cost_list, ang_list, t_list = [], [], []

    for _ in range(random_trail):
        
        def cost_to_optimize(list_ang):
            # t = np.random.random(3)
            tx, ty, tz = list_ang[12], list_ang[13], list_ang[14]
            U31, U32 = qiskit_U3(list_ang[0], list_ang[1], list_ang[2]), qiskit_U3(list_ang[3], list_ang[4], list_ang[5])
            U33, U34 = qiskit_U3(list_ang[6], list_ang[7], list_ang[8]),qiskit_U3(list_ang[9], list_ang[10], list_ang[11])

            majhe_eso = Can(tx,ty,tz)
            peeche_hat = np.kron(U31, U32)
            egia_cholo = np.kron(U33, U34)
            prodhan_gate = np.matmul(egia_cholo, np.matmul(majhe_eso, peeche_hat))
            choi01 = Choi(UnitaryGate(prodhan_gate))

            return -process_fidelity(choi0,choi01)
        list_ang = np.random.random(12)*2*np.pi
        t_ang = np.random.random(3)
        main_list = np.concatenate((list_ang, t_ang))
        res = minimize(cost_to_optimize, main_list, method = 'COBYLA', options={'maxiter': 100})
        cost_list.append(res['fun'])
        ang_list.append(res['x'])
        t_list.append(t_ang)
        print(res['fun'])
    best_GS = cost_list.index(np.min(cost_list))
    sort_opt_ang = ang_list[best_GS][:12]
    opt_t = ang_list[best_GS][12:]
        
    return sort_opt_ang, opt_t
        

def rand_decomp_2q(choi0, gs, trials, max_depth):

    
    pfi_best = 0
    dep_best = 0
    qcirc_best = []
    for i in range(trials):
        dep = random.randrange(1,max_depth)
        seq = random.choices(list(gs.keys()), k=dep)
        qc0 = QuantumCircuit(2)
        for seqid in seq:
            choice_U2 = random.choices([[0,1],[1,0]], k=1)
            # print(choice_U2)
            if seqid == 'U2':
                qc0.append(gs[seqid], choice_U2[0])
            else:
                qc0.append(gs[seqid], [choice_U2[0][0]])
        choi01 = Choi(qc0)
        pfi = process_fidelity(choi0,choi01)
        if pfi > pfi_best:
            pfi_best = pfi
            qcirc_best = qc0
            dep_best = dep

    return pfi_best, dep_best, qcirc_best
    

def scipy_2q_decomp_multi_rand_unitary(U, random_trail):

    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)

    gs1 = {}    
    gs1['H'] = UnitaryGate(h_U_mat,label='H')
    gs1['T'] = UnitaryGate(t_U_mat,label='T')

    choi0_list = []
    for uel in U:
        choi0_list.append(Choi(UnitaryGate(uel)))

    def cost_to_optimize(list_ang):
        tx, ty, tz = list_ang[12], list_ang[13], list_ang[14]
        U31, U32 = qiskit_U3(list_ang[0], list_ang[1], list_ang[2]), qiskit_U3(list_ang[3], list_ang[4], list_ang[5])
        U33, U34 = qiskit_U3(list_ang[6], list_ang[7], list_ang[8]),qiskit_U3(list_ang[9], list_ang[10], list_ang[11])

        majhe_eso = Can(tx,ty,tz)
        peeche_hat = np.kron(U31, U32)
        egia_cholo = np.kron(U33, U34)
        prodhan_gate = np.matmul(egia_cholo, np.matmul(majhe_eso, peeche_hat))
        gs1['U2'] = UnitaryGate(prodhan_gate, label = 'U2')

        fid_list = []
        for c0 in choi0_list:
            pfi_best, dep_best, qcirc_best = rand_decomp_2q(c0, gs1, trials=10, max_depth=100)
            fid_list.append(pfi_best)
        ave_fid_fin = np.mean(fid_list)
        return -ave_fid_fin

    cost_list, ang_list, t_list = [], [], []

    for _ in range(random_trail):
        
        list_ang = np.random.random(12)*2*np.pi
        t_ang = np.random.random(3)
        main_list = np.concatenate((list_ang, t_ang))
        res = minimize(cost_to_optimize, main_list, method = 'COBYLA', options={'maxiter': 100})
        cost_list.append(res['fun'])
        ang_list.append(res['x'])
        t_list.append(t_ang)
        print(res['fun'])
    best_GS = cost_list.index(np.min(cost_list))
    sort_opt_ang = ang_list[best_GS][:12]
    opt_t = ang_list[best_GS][12:]
        
    return sort_opt_ang, opt_t


## Notes:

# https://qiskit.org/documentation/stubs/qiskit.quantum_info.TwoQubitBasisDecomposer.html
# Decomposes a two-qubit unitary into a minimal number of KAK (0,1,2,3) gates, based on how many rotation axis needed
# ------------------------------------------------------------------------------------------------ #

from qiskit.quantum_info.synthesis import qsd
# https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/synthesis/qsd.py

def dcmp_QSD_rand(randU, gs, trials = 100, max_depth= 100):

    # Choi matrix of the unitary
    # U_dim = randU.num_qubits
    # qc0 = QuantumCircuit(U_dim)
    # qc0.append(Operator(randU),range(U_dim)) # The unitary is always applied in qubits in ascending order
    # choi0 = Choi(qc0)
    # print(randU)
    # print(qc0)

    # Decompose randU using Qiskit QSD
    qsd_circ = qsd.qs_decomposition(randU.to_matrix())
    # choi01 = Choi(qsd_circ)
    # print('\nQSD Fidelity:',process_fidelity(choi0,choi01))
    # print(qsd_circ)

    # Create a list of rotation gates to be decomposed
    ds_qsd_1q = []
    for g in qsd_circ:
        if g.operation.num_qubits == 1:
            ds_qsd_1q.append(UnitaryGate(Operator(g.operation),label='QSD_U'))
    
    # Decompose the rotation gates using 1-qb gs
    U_qsd_1q_gs_qcirc_best_db = []
    cum_pf = 1
    for U_qsd_1q in ds_qsd_1q:
        # print('Decomposing:')
        # print(U_qsd_1q.definition)
        pfi_best, _, qcirc_best = dcmp_rand(U_qsd_1q, gs, trials = trials, max_depth= max_depth)
        U_qsd_1q_gs_qcirc_best_db.append(qcirc_best)
        cum_pf *= pfi_best
        # print('Process Fidelity of decomposed gate:',pfi_best,'Cumulative Fidelity:', cum_pf)

    # Decompose the CX gate using KAK-CAN

    # Replace the rotation gates with decomposed circuits
    qc01 = QuantumCircuit(randU.num_qubits)
    for g in qsd_circ:
        if g.operation.num_qubits == 1:
            U_gs = U_qsd_1q_gs_qcirc_best_db.pop(0)
            tgt = [qsd_circ.find_bit(x).index for x in g.qubits]
            for g_gs in U_gs:
                qc01.append(g_gs.operation, tgt)
        else:
            qc01.append(g.operation, [qsd_circ.find_bit(x).index for x in g.qubits])
            # if  g.operation.num_qubits == 2:
            #     replace with the KAK-CAN decomposition
            
    # choi01 = Choi(qc01)
    # pfi01 = process_fidelity(choi0,choi01)
    # print('\nFinal Fidelity:',pfi01)
    # print(qc01)

    return qc01

# ------------------------------------------------------------------------------------------------ #

from qiskit.quantum_info.synthesis import qsd
# https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/synthesis/qsd.py

def dcmp_rand_N(randU, gs, trials = 100, max_depth= 100):

    # Choi matrix of the unitary
    U_dim = randU.num_qubits
    qc0 = QuantumCircuit(U_dim)
    qc0.append(Operator(randU),range(U_dim)) # The unitary is always applied in qubits in ascending order
    choi0 = Choi(qc0)
    print(randU)
    print(qc0)

    # Decompose randU using Qiskit QSD
    qsd_circ = qsd.qs_decomposition(randU.to_matrix())
    choi01 = Choi(qsd_circ)
    pfi = process_fidelity(choi0,choi01)
    print('\nQSD Fidelity:',pfi)
    print(qsd_circ)

    # Create a list of rotation gates to be decomposed
    ds_qsd_1q = []
    for g in qsd_circ:
        if g.operation.num_qubits == 1:
            ds_qsd_1q.append(UnitaryGate(Operator(g.operation),label='QSD_U'))
    
    # Decompose the rotation gates using 1-qb gs
    U_qsd_1q_gs_qcirc_best_db = []
    cum_pf = 1
    for U_qsd_1q in ds_qsd_1q:
        print('Decomposing:')
        print(U_qsd_1q.definition)
        pfi_best, _, qcirc_best = dcmp_rand(U_qsd_1q, gs, trials = trials, max_depth= max_depth)
        U_qsd_1q_gs_qcirc_best_db.append(qcirc_best)
        cum_pf *= pfi_best
        print('Process Fidelity of decomposed gate:',pfi_best,'Cumulative Fidelity:', cum_pf)

    # Replace the rotation gates with decomposed circuits
    qc01 = QuantumCircuit(randU.num_qubits)
    for g in qsd_circ:
        if g.operation.num_qubits == 1:
            U_gs = U_qsd_1q_gs_qcirc_best_db.pop(0)
            tgt = [qsd_circ.find_bit(x).index for x in g.qubits]
            for g_gs in U_gs:
                qc01.append(g_gs.operation, tgt)
        else:
            qc01.append(g.operation, [qsd_circ.find_bit(x).index for x in g.qubits])
    choi01 = Choi(qc01)
    pfi = process_fidelity(choi0,choi01)
    print('\nFinal Fidelity:',pfi)
    print(qc01)

    return

########################################################################################################################################################################################################

# ------------------------------------------------------------------------------------------------ #

"""
Data Set Generation: Haar random unitaries
Ref: https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_unitary.html
"""

def gen_ds_randU(samples = 100, max_dim = 1, rand_dim = False):
    ds = []
    for i in range(samples):
        dim = max_dim
        if rand_dim:
            dim =  random.randrange(1,max_dim+1)    # TBD: Samples should be proportional to dimension instead of uniform, i.e. exponentially more higher dimension than lower dimensions
        ds.append(UnitaryGate(random_unitary(2**dim),label='RndU'+str(i)))
    return ds

# ------------------------------------------------------------------------------------------------ #

"""
Data Set Generation: Equispaced non-local unitaries
Ref: https://weylchamber.readthedocs.io/en/latest/API/weylchamber.coordinates.html#weylchamber.coordinates.point_in_weyl_chamber
"""

def gen_ds_equiNL(px = 10, max_dim = 2, rand_dim = False):
    ds = []
    if max_dim != 2:
        print("gen_ds_equiNL works only for 2 qubits")
        return ds
    cx = np.linspace(0, 1, px)
    cy = np.linspace(0, 0.5, int(px/2))
    cz = np.linspace(0, 0.5, int(px/2))
    gs = product(cx, cy, cz)
    valid_points = 0
    for can in gs:
        # Enumerate points in the Weyl chamber
        c = list(can)
        c1,c2,c3 = c[0],c[1],c[2]
        if weylchamber.point_in_weyl_chamber(c1,c2,c3):
            valid_points+= 1
            ds.append(UnitaryGate(weylchamber.canonical_gate(c1,c2,c3),label='RndU'+str(valid_points)))   
    return ds

# ------------------------------------------------------------------------------------------------ #

"""
Data Set Generation: Random non-local unitaries
Ref: https://weylchamber.readthedocs.io/en/latest/API/weylchamber.coordinates.html#weylchamber.coordinates.random_gate
"""

def gen_ds_randNL(samples = 100, max_dim = 2, rand_dim = False):
    ds = []
    if max_dim != 2:
        print("gen_ds_randNL works only for 2 qubits")
        return ds
    for i in range(samples):
        ds.append(UnitaryGate(weylchamber.random_gate(),label='RndU'+str(i)))    
    return ds


# ------------------------------------------------------------------------------------------------ #

"""
Given an unitary and a gate set, use trials to find a circuit using the gate set that is close to the unitary
Generalized n-qubit version
"""

def dcmp_rand(randU, gs, trials = 100, max_depth= 100):

    # Choi matrix of the unitary
    U_dim = randU.num_qubits
    qc0 = QuantumCircuit(U_dim)
    qc0.append(Operator(randU),range(U_dim)) # The unitary is always applied in qubits in ascending order
    choi0 = Choi(qc0)
    
    # Considers only gates in gs with num_qubits <= U_dim for the decomposition
    U_gs = {}
    for i in gs:
        if gs[i].num_qubits <= U_dim:
            U_gs[i] = gs[i]

    pfi_best = 0
    dep_best = 0
    qcirc_best = []
    # for _ in tqdm(range(trials)):   # for TEST
    for _ in range(trials):
        dep = random.randrange(1,max_depth)
        seq = random.choices(list(U_gs.keys()), k=dep)
        qc0 = QuantumCircuit(U_dim)
        for i in seq:
            shf_qb = list(range(U_dim))
            random.shuffle(shf_qb)
            sel_shf_qb = shf_qb[0:U_gs[i].num_qubits]
            qc0.append(U_gs[i], sel_shf_qb)
        choi01 = Choi(qc0)
        pfi = process_fidelity(choi0,choi01)
        if pfi > pfi_best:                      # TBD: Should we consider the circuit with the highest fidelity or the one with the lowest depth?
            pfi_best = pfi
            qcirc_best = qc0
            dep_best = dep
    
    return pfi_best, dep_best, qcirc_best

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


        plt.show()

# ------------------------------------------------------------------------------------------------ #

def TEST_dcmp_rand():

    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    cx_U_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    gs1 = {}    
    gs1['H'] = UnitaryGate(h_U_mat,label='H')
    gs1['T'] = UnitaryGate(t_U_mat,label='T')
    gs1['CX'] = UnitaryGate(cx_U_mat,label='CX')
    gs1_gates = ','.join(list(gs1.keys()))
    
    samples = 5
    ds = gen_ds_randU(samples = samples, max_dim = 2)

    # Choi matrix of the unitary
    choi0_db = []
    for randU in ds:
        U_dim = randU.num_qubits
        qc0 = QuantumCircuit(U_dim)
        qc0.append(Operator(randU),range(U_dim)) # The unitary is always applied in qubits in ascending order
        choi0_db.append(Choi(qc0))

    pf01_db, cd01_db = [], []
    for i in range(samples):  
        qc01 = dcmp_QSD_rand(ds[i], gs1, trials = 100, max_depth= 50)
        choi01 = Choi(qc01)
        pf01_db.append(process_fidelity(choi0_db[i],choi01))
        cd01_db.append(qc01.size())

    print(pf01_db)
    print(cd01_db)

    cfn_best, cfn_best_db = 100000, []
    trials = 10
    for t in range(trials):
        # ===> Define gateset GS2 (random gates)    
        U1_mat = random_unitary(2).data
        U2_mat = random_unitary(2).data
        gs2 = {}   
        gs2['U1'] = UnitaryGate(U1_mat,label='U1') 
        gs2['U2'] = UnitaryGate(U2_mat,label='U2')
        gs2['CX'] = UnitaryGate(cx_U_mat,label='CX')
        gs2_gates = ','.join(list(gs2.keys()))

        pf02_db, cd02_db = [], []
        for i in range(samples):  
            qc02 = dcmp_QSD_rand(ds[i], gs2, trials = 100, max_depth= 50)
            choi02 = Choi(qc02)
            pf02_db.append(process_fidelity(choi0_db[i],choi02))
            cd02_db.append(qc02.size())      
        
        print("Trial:",t, pf02_db, cd02_db)
        # ===> Evaluate GS2 based on cost function
        cfn = cfn_calc(pf01_db, cd01_db, pf02_db, cd02_db)
        if cfn <= cfn_best:
            cfn_best = cfn
            cfn_best_db = [gs2, pf02_db, cd02_db]   

    print("\nBest settings found:",cfn_best_db[0])
    print("\nBest metrics found:",cfn_best_db[1:])

     
    # ===> Plot results
    plot_compare_gs(gs1_gates,gs2_gates,pf01_db,pf02_db,cd01_db,cd02_db,pfivt=True)   

    return

# ------------------------------------------------------------------------------------------------ #

from qiskit.quantum_info import TwoQubitBasisDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition

def TEST_KAK():

    # The basis gate is supercontrolled for an exact decomposition (i.e., has Weyl coordinates (π/4,β,0))
    bg = UnitaryGate(weylchamber.canonical_gate(0.5,0.25,0), label='B')  # Berkeley gate
    dcmp_B_KAK = TwoQubitBasisDecomposer(bg)

    # Define a random unitary to decompose
    U = Operator(random_unitary(4).data)
    qc = QuantumCircuit(2)
    qc.append(U, [0,1])
    qc_gate = qc.to_gate() 
    U_tgt = TwoQubitWeylDecomposition(Operator(qc_gate).data)  

    # print(B_KAK.num_basis_gates(U))
    U3r, U3l, U2r, U2l, U1r, U1l, U0r, U0l = dcmp_B_KAK.decomp3_supercontrolled(U_tgt)

    u0 = np.kron(U0l,U0r)
    u1 = np.kron(U1l,U1r)
    u2 = np.kron(U2l,U2r)
    u3 = np.kron(U3l,U3r)
    UD = Operator(u0 @ bg.to_matrix() @ u1 @ bg.to_matrix() @ u2 @ bg.to_matrix() @ u3)

    Uc = Choi(U)  
    UDc = Choi(UD)
    print(process_fidelity(Uc,UDc))

    return

########################################################################################################################################################################################################

if __name__ == "__main__":
    
    # prconda activate C:\Users\aritr\anaconda3\envs\yaqq
    
    TEST_KAK()
    exit()

    # TEST_dcmp_rand()
    # exit()


    samples, dim = 10,2

    rand_U = gen_ds_randU(samples = samples, dimensions = dim)
    # choi0 = []
    # for rand_el in rand_U:
    #     choi0.append(Choi(UnitaryGate(rand_el)))
    
    rand_trial = 10
    # rand_2q_decomp(rand_U0, rand_trial)
    best_ang, best_t = scipy_2q_decomp_multi_rand_unitary(rand_U, rand_trial)

    # list_ang = best_ang
    # t = best_t
    # tx, ty, tz = t[0], t[1], t[2]
    # # print(list_ang)
    # U31, U32 = qiskit_U3(list_ang[0], list_ang[1], list_ang[2]), qiskit_U3(list_ang[3], list_ang[4], list_ang[5]),
    # U33, U34 = qiskit_U3(list_ang[6], list_ang[7], list_ang[8]),qiskit_U3(list_ang[9], list_ang[10], list_ang[11])

    # majhe_eso = Can(tx,ty,tz)
    # peeche_hat = np.kron(U31, U32)
    # egia_cholo = np.kron(U33, U34)
    # prodhan_gate = np.matmul(egia_cholo, np.matmul(majhe_eso, peeche_hat))
    # # print('----------------------------')
    # # print(rand_U0)
    # # print()
    # # print(prodhan_gate)
    # # print('----------------------------')
    # print('-----FIDELTY ASCHE------')
    # choi01 = Choi(UnitaryGate(prodhan_gate))
    # x = []
    # for ch0 in choi0:
    #     x.append(process_fidelity(ch0,choi01))
    #     print(process_fidelity(ch0,choi01))
    # plt.plot(x)
    # plt.show()


