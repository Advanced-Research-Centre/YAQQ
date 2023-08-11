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



def rand_2q_decomp(U, random_trail):

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
            # qc01.append(U_qsd_1q_gs_qcirc_best_db.pop(0), [qsd_circ.find_bit(x).index for x in g.qubits])
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
    for _ in tqdm(range(trials)):   # for TEST
    # for _ in range(trials):
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
        if pfi > pfi_best:
            pfi_best = pfi
            qcirc_best = qc0
            dep_best = dep

    # print('TEST: Best process fidelity: ', pfi_best)    # for TEST
    # print('TEST: Best depth: ', dep_best)               # for TEST
    # print('TEST: Best circuit:')                        # for TEST
    # print(qcirc_best)                                   # for TEST
    
    return pfi_best, dep_best, qcirc_best
    

def TEST_dcmp_rand():

    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    cx_U_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    gs1 = {}    
    gs1['H'] = UnitaryGate(h_U_mat,label='H')
    gs1['T'] = UnitaryGate(t_U_mat,label='T')
    gs1['CX'] = UnitaryGate(cx_U_mat,label='CX')

    ds = gen_ds_randU(samples = 1, max_dim = 2)

    for randU in ds:
        dcmp_rand_N(randU, gs1, trials = 500, max_depth= 500)
        # dcmp_rand(randU = randU, gs = gs1, trials = 3000, max_depth= 500)

    return

# ------------------------------------------------------------------------------------------------ #

########################################################################################################################################################################################################

if __name__ == "__main__":
    
    # prconda activate C:\Users\aritr\anaconda3\envs\yaqq
    
    TEST_dcmp_rand()
    exit()


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


