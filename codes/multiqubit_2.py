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
from qiskit.quantum_info import Pauli, Choi, process_fidelity
from scipy.optimize import minimize  
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import random
# from math import pi
import numpy as np

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

def gen_ds_randU(samples = 1, dimensions = 2):
    ds = []
    for i in range(samples):
        ds.append(UnitaryGate(random_unitary(2**dimensions),label='RndU'+str(i)))
    return ds

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
    


    

if __name__ == "__main__":
    samples, dim = 10,2

    rand_U = gen_ds_randU(samples = samples, dimensions = dim)
    choi0 = []
    for rand_el in rand_U:
        choi0.append(Choi(UnitaryGate(rand_el)))
    
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


