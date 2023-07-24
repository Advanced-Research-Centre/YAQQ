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
from qiskit.extensions import UnitaryGate, U3Gate, UGate
from qiskit.circuit import Gate
from qiskit.quantum_info import random_unitary
import scipy.linalg as la
from qiskit.circuit.gate import Gate
from tqdm import tqdm
from scipy.optimize import minimize  
import random
from qiskit.quantum_info.operators.random import random_unitary
from qiskit.quantum_info import random_statevector, Statevector, DensityMatrix, Operator
from qiskit.visualization import plot_bloch_multivector
import qutip as qt
from qutip.measurement import measurement_statistics
    
####################################################################################################
    

    
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

"""
Calculate cost function based on distribution of process fidelity differences and gate depths 
of two gate sets approximating points on the Hilbert space
"""

def cfn_calc(fid_gs01,dep_gs01,fid_gs02,dep_gs02):
    ivt_fid_gs01 = np.subtract(1,fid_gs01)
    dist_fid = sum(abs(np.subtract(ivt_fid_gs01,fid_gs02)))
    avg_fid_gs01 = np.mean(fid_gs01) 
    avg_dep_gs01 = np.mean(dep_gs01)
    avg_fid_gs02 = np.mean(fid_gs02)
    avg_dep_gs02 = np.mean(dep_gs02)
    dist_fid_avg = - avg_fid_gs02   # avg_fid_gs01 - avg_fid_gs02
    dist_dep_avg = avg_dep_gs02     # avg_dep_gs02 - avg_dep_gs01
    w_trend, w_favg, w_davg = 200, 1000, 1
    cfn = w_trend*dist_fid + w_favg*dist_fid_avg + w_davg*dist_dep_avg
    return cfn

####################################################################################################
 
def novel_gs_rand_randS():
    
    # ===> Make trial states on Bloch sphere
    
    points = 50
    rz_ang_list, rx_ang_list = fibo_bloch(points)[0], fibo_bloch(points)[1]   

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

    # ===> Decompose each state on Bloch sphere with GS1 and store fid and depth
    qc0_db, choi0_db = [], []
    pf01_db, dep01_db = [], []
    for p in range(points):
        qc0 = QuantumCircuit(1)
        qc0.rz(rz_ang_list[p],0)
        qc0.rx(rx_ang_list[p],0)
        qc0_db.append(qc0)
        choi0 = Choi(qc0)
        choi0_db.append(choi0)
        qc01 = skd1(qc0)
        choi01 = Choi(qc01)
        pf01_db.append(process_fidelity(choi0,choi01))
        dep01_db.append(qc01.depth())

    # ===> Find complementary gate set via random search

    trials = 20
    cfn_best, cfn_best_db = 100000, []
    gateset_list, total_fid_list, total_depth_list = [], [], []
    for t in tqdm(range(trials)):

        # ===> Generate random unitaries, sequences and declare SKT object for GS2
        
        unitary1, unitary2, unitary3 = random_unitary(2).data, random_unitary(2).data, random_unitary(2).data
        U1 = UGate("U1", unitary1)
        U2 = UGate("U2", unitary2)
        U3 = UGate("U3", unitary3)
        agent2_gateset = [U1, U2, U3]
        agent2_gateseq = gbs.generate_basic_approximations(agent2_gateset, max_depth) 
        skd2 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent2_gateseq)    

        # ===> Decompose each state on Bloch sphere with GS2
       
        pf02_db, dep02_db = [], []
        for p in range(points):
            qc02 = skd2(qc0_db[p])
            choi02 = Choi(qc02)
            pf02_db.append(process_fidelity(choi0_db[p],choi02))
            dep02_db.append(qc02.depth())
            # fid_list.append([pf01,pf02])
            # depth_list.append([qc01.depth(),qc02.depth()])

        # ===> Evaluate GS2 based on cost function
        
        cfn = cfn_calc(pf01_db, dep01_db, pf02_db, dep02_db)
        if cfn <= cfn_best:
            cfn_best = cfn
            cfn_best_db = [[unitary1, unitary2, unitary3],pf02_db,dep02_db]
    
    ivt_fid_gs01 = np.subtract(1,pf01_db)
    avg_fid_gs01 = np.mean(pf01_db)         # Not same for ivt_fid_gs01 and pf01_db
    avg_fid_best = np.mean(cfn_best_db[1])
    avg_dep_gs01 = np.mean(dep01_db)
    avg_dep_best = np.mean(cfn_best_db[2])

        # ===> Save all data GS2 (required?)
    #     gateset_list.append([unitary1, unitary2, unitary3])
    #     total_fid_list.append(pf02_db)
    #     total_depth_list.append(dep02_db)

    # save_data = input("Save data? [y/n]: ")
    # if save_data == 'y':
    #     np.save( 'data/total_depth_list', total_depth_list )
    #     np.save( 'data/total_fid_list', total_fid_list )
    #     np.save( 'data/gateset_list', gateset_list )

    plot_data = input("Plot data? [y/n]: ")
    if plot_data == 'y':
        _, ax = plt.subplots(1, 2)
        ax[0].plot(pf01_db, '-x', color = 'r', label = 'PF [uh, ut, utdg]')
        ax[0].plot(ivt_fid_gs01, '-x', color = 'g', label = 'target PF trend')
        ax[0].plot(cfn_best_db[1], '-o', color = 'b', label = 'PF [u1, u2, u3]')

        ax[0].axhline(y=avg_fid_gs01, linestyle='-.', color = 'r' , label = 'avg.PF [uh, ut, utdg]')
        ax[0].axhline(y=avg_fid_best, linestyle='-.', color = 'b' , label = 'avg.PF [u1, u2, u3]')

        ax[1].plot(dep01_db, '-x', color = 'r', label = 'CD [uh, ut, utdg]')
        ax[1].plot(cfn_best_db[2], '-o', color = 'b', label = 'CD [u1, u2, u3]')

        ax[1].axhline(y=avg_dep_gs01, linestyle='-.', color = 'r', label = 'avg.CD [uh, ut, utdg]')
        ax[1].axhline(y=avg_dep_best, linestyle='-.', color = 'b', label = 'avg.CD [u1, u2, u3]')

        ax[0].set_ylabel("Process Fidelity")
        ax[1].set_ylabel("Circuit Depth")
        ax[0].set_ylim(bottom=0,top=1)
        ax[1].set_ylim(bottom=0,top=None)
        ax[0].legend()
        ax[1].legend()
        plt.show()

    return

####################################################################################################

def cost_func(rzrx, agent1_gateset, thetas, max_depth, recursion_degree):
    ## ===> Given set
    gbs = gen_basis_seq()
    agent1_gateseq = gbs.generate_basic_approximations(agent1_gateset, max_depth) 
    skd1 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent1_gateseq)
    ## ===> New set
    U3a = UGate("U3a", qiskit_U3(thetas[0], thetas[1], thetas[2]))
    U3b = UGate("U3b", qiskit_U3(thetas[3], thetas[4], thetas[5]))
    agent2_gateset = [U3a, U3b]
    agent2_gateseq = gbs.generate_basic_approximations(agent2_gateset, max_depth) 
    skd2 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent2_gateseq) 
    # ===> Decompose each state on Bloch sphere with GS1 and store fid and depth
    pf01_db, dep01_db, pf02_db, dep02_db = fidelity_per_point(rzrx, skd1, skd2)
    # ===> Evaluate GS2 based on cost function
    cfn = cfn_calc(pf01_db, dep01_db, pf02_db, dep02_db)
    return cfn

def fidelity_per_point(rzrx, skd1, skd2):
    rz_ang_list, rx_ang_list = rzrx[0], rzrx[1]
    qc0_db, choi0_db = [], []
    pf01_db, dep01_db = [], []
    for p in range(len(rz_ang_list)):
        qc0 = QuantumCircuit(1)
        qc0.rz(rz_ang_list[p],0)
        qc0.rx(rx_ang_list[p],0)
        qc0_db.append(qc0)
        choi0 = Choi(qc0)
        choi0_db.append(choi0)
        qc01 = skd1(qc0)
        choi01 = Choi(qc01)
        pf01_db.append(process_fidelity(choi0,choi01))
        dep01_db.append(qc01.depth())
    # ===> Decompose each state on Bloch sphere with GS2
    pf02_db, dep02_db = [], []
    for p in range(len(rz_ang_list)):
        qc02 = skd2(qc0_db[p])
        choi02 = Choi(qc02)
        pf02_db.append(process_fidelity(choi0_db[p],choi02))
        dep02_db.append(qc02.depth())
    return pf01_db, dep01_db, pf02_db, dep02_db


def rand_decompose(qb,randU,gs,trials,depth):
    # Given an unitary and a gate set, use trials to find a circuit using the gate set that is close to the unitary
    qc0 = QuantumCircuit(qb)
    qc0.append(randU, [0])
    choi0 = Choi(qc0)
    
    pfi_best = 0
    # qcirc_best = []
    for i in range(trials):
        seq = random.choices(list(gs.keys()),k=depth)
        qc0 = QuantumCircuit(qb)
        for i in seq:
            qc0.append(gs[i], [0])

            # need to handle for 2 qubit

            # if i == 'h':
            #     qc0.h(0)
            # elif i == 't':
            #     qc0.t(0)
            # elif i == 'h0':
            #     qc0.h(0)
            # elif i == 'h1':
            #     qc0.h(1)
            # elif i == 't0':
            #     qc0.t(0)
            # elif i == 't1':
            #     qc0.t(1)
            # elif i == 'cx01':
            #     qc0.cx(0,1)
            # elif i == 'cx10':
            #     qc0.cx(1,0)
            # else:   # for custom gates
            #     qc0.append(gs[i], [0])
        choi01 = Choi(qc0)
        pfi = process_fidelity(choi0,choi01)
        if pfi > pfi_best:
            pfi_best = pfi
            # qcirc_best = qc0

    return pfi_best

####################################################################################################
 
def novel_gs_rand_randU():
    
    # ===> Make test set
    points = 25
    testset = gen_testset(points, 1)

    # ===> Define gateset GS1 (standard gates via U-gate)
    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    gs1 = {}
    gs1['H'] = UnitaryGate(h_U_mat,label='Uh')
    gs1['T'] = UnitaryGate(t_U_mat,label='Ut')
    gs1_gates = ','.join(list(gs1.keys()))
    # gs1_U1, gs1_U2, gs1_U3 = random_unitary(2).data, random_unitary(2).data, random_unitary(2).data
    # gs1['U1'] = UnitaryGate(gs1_U1,label='U1')
    # gs1['U2'] = UnitaryGate(gs1_U2,label='U2')
    # gs1['U3'] = UnitaryGate(gs1_U3,label='U3')

    # ===> Decompose each randU with GS1 using random decomposition and store fid and depth
    depth = 20
    pf01_db, dep01_db = [], []
    for randU in testset:
        pf01_db.append(rand_decompose(1,randU,gs1,trials=50,depth=depth))
        dep01_db.append(depth)

    # ===> Find complementary gate set via random search

    trials = 50
    cfn_best, cfn_best_db = 100000, []
    # gateset_list, total_fid_list, total_depth_list = [], [], []
    for t in tqdm(range(trials)):

        # ===> Generate random unitaries for GS2
        
        gs2_U1, gs2_U2, gs2_U3 = random_unitary(2).data, random_unitary(2).data, random_unitary(2).data
        gs2 = {}
        gs2['U1'] = UnitaryGate(gs2_U1,label='U1')
        gs2['U2'] = UnitaryGate(gs2_U2,label='U2')
        gs2_gates = ','.join(list(gs2.keys()))
        # gs2['U3'] = UnitaryGate(gs2_U3,label='U3')

        # ===> Decompose each randU with GS2
       
        pf02_db, dep02_db = [], []
        for randU in testset:
            pf02_db.append(rand_decompose(1,randU,gs2,trials=50,depth=depth))
            dep02_db.append(depth)

        # ===> Evaluate GS2 based on cost function
        
        cfn = cfn_calc(pf01_db, dep01_db, pf02_db, dep02_db)
        if cfn <= cfn_best:
            cfn_best = cfn
            cfn_best_db = [[gs2_U1, gs2_U2, gs2_U3],pf02_db,dep02_db]
    
    print("Best settings found:",cfn_best_db[0])

    ivt_fid_gs01 = np.subtract(1,pf01_db)
    avg_fid_gs01 = np.mean(pf01_db)         # Not same for ivt_fid_gs01 and pf01_db
    avg_fid_best = np.mean(cfn_best_db[1])
    avg_dep_gs01 = np.mean(dep01_db)
    avg_dep_best = np.mean(cfn_best_db[2])    
        
    plot_data = input("Plot data? [y/n]: ")
    if plot_data == 'y':
        _, ax = plt.subplots(1, 2)
        ax[0].plot(pf01_db, '-x', color = 'r', label = 'PF ['+gs1_gates+']')
        ax[0].plot(ivt_fid_gs01, '-x', color = 'g', label = 'target PF trend')
        ax[0].plot(cfn_best_db[1], '-o', color = 'b', label = 'PF ['+gs2_gates+']')

        ax[0].axhline(y=avg_fid_gs01, linestyle='-.', color = 'r' , label = 'avg.PF ['+gs1_gates+']')
        ax[0].axhline(y=avg_fid_best, linestyle='-.', color = 'b' , label = 'avg.PF ['+gs2_gates+']')

        ax[1].plot(dep01_db, '-x', color = 'r', label = 'CD ['+gs1_gates+']')
        ax[1].plot(cfn_best_db[2], '-o', color = 'b', label = 'CD ['+gs2_gates+']')

        ax[1].axhline(y=avg_dep_gs01, linestyle='-.', color = 'r', label = 'avg.CD ['+gs1_gates+']')
        ax[1].axhline(y=avg_dep_best, linestyle='-.', color = 'b', label = 'avg.CD ['+gs2_gates+']')

        ax[0].set_ylabel("Process Fidelity")
        ax[1].set_ylabel("Circuit Depth")
        ax[0].set_ylim(bottom=0,top=1)
        ax[1].set_ylim(bottom=0,top=None)
        ax[0].legend()
        ax[1].legend()
        plt.show()

####################################################################################################

def cost_func_randU(testset, gs1, thetas, depth=10):
 
    ## ===> New set

    U3a_mat = qiskit_U3(thetas[0], thetas[1], thetas[2])
    U3b_mat = qiskit_U3(thetas[3], thetas[4], thetas[5])
    gs2 = {}    
    gs2['u3a'] = UnitaryGate(U3a_mat,label='U3a')
    gs2['u3b'] = UnitaryGate(U3b_mat,label='U3b')

    pf01_db, dep01_db = [], [10]*5
    pf02_db, dep02_db = [], [10]*5

    for randU in testset:
        pf01_db.append(rand_decompose(1,randU,gs1,trials=10,depth=depth))
        pf02_db.append(rand_decompose(1,randU,gs2,trials=30,depth=depth))

    # ===> Evaluate GS2 based on cost function
    cfn = cfn_calc(pf01_db, dep01_db, pf02_db, dep02_db)
    return cfn, pf01_db, pf02_db

  

####################################################################################################

def novel_gs_scipy_randU():
    """
    ADGHRFRGFDRETHNJGDEGHBBGHXNZ
    """

    # ===> Make test set
    points = 5
    testset = gen_testset(points, 1)
    cfn_best, cfn_best_db = 100000, []

    # ===> Define gateset GS1 (standard gates via U-gate)
    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    gs1 = {}
    gs1['h'] = UnitaryGate(h_U_mat,label='Uh')
    gs1['t'] = UnitaryGate(t_U_mat,label='Ut')

    # ===> Decompose each randU with GS1 using random decomposition and store fid and depth
    depth = 5
    pf01_db, pf02_db = [], []

    trials  = 3

    def cost_to_optimize(thetas):
        cost, pf01, pf02 = cost_func_randU(testset, gs1, thetas, depth=depth)
        return cost
    
    cost_list, ang_list = [], []
    for _ in range(trials):
        initial_guess = np.random.random(2*3)
        initial_cost = cost_to_optimize(initial_guess)
        print('Cost of initial random guess of GS:', initial_cost)
        res = minimize(cost_to_optimize, initial_guess, method = 'COBYLA', options={'maxiter': 50})
        print('Cost of optimized GS:', res.fun)
        cost, ang = 0, 0
        if res['fun'] < initial_cost:
            cost = res['fun']
            ang = res['x']
        else:
            cost = initial_cost
            ang = initial_guess
        
        cost_list.append(cost)
        ang_list.append(ang)   
    
    best_GS = cost_list.index(np.min(cost_list))
    sort_opt_ang = ang_list[best_GS]
    np.save('opt_ang_rand', sort_opt_ang)

    # bana tera gs2 
    U3a_mat = qiskit_U3(sort_opt_ang[0], sort_opt_ang[1], sort_opt_ang[2])
    U3b_mat = qiskit_U3(sort_opt_ang[3], sort_opt_ang[4], sort_opt_ang[5])
    gs2 = {}    
    gs2['u3a'] = UnitaryGate(U3a_mat,label='U3a')
    gs2['u3b'] = UnitaryGate(U3b_mat,label='U3b')

    pf02_db, pf01_db = [], []
    for randU in testset:
        pf02_db.append(rand_decompose(1,randU,gs2,trials=50,depth=depth))
        pf01_db.append(rand_decompose(1,randU,gs1,trials=50,depth=depth))
    

    print("\nAngles of best GS found:",sort_opt_ang, "with cost",cost_list[best_GS])

    ivt_fid_gs01 = np.subtract(1,pf01_db)
    avg_fid_gs01 = np.mean(pf01_db)
    avg_fid_best = np.mean(pf02_db)   
        
    plot_data = input("Plot data? [y/n]: ")
    if plot_data == 'y':


        _, ax = plt.subplots(1, 2)
        ax[0].plot(pf01_db, '-x', color = 'r', label = 'PF [gs1_gates]')
        ax[0].plot(ivt_fid_gs01, '-x', color = 'g', label = 'target PF trend')
        ax[0].plot(pf02_db, '-o', color = 'b', label = 'PF [+gs2_gates+]')

        ax[0].axhline(y=avg_fid_gs01, linestyle='-.', color = 'r' , label = 'avg.PF [+gs1_gates+]')
        ax[0].axhline(y=avg_fid_best, linestyle='-.', color = 'b' , label = 'avg.PF [+gs2_gates+]')

        ax[0].set_ylabel("Process Fidelity")
        ax[0].set_ylim(bottom=0,top=1)
        ax[0].legend()
        plt.show()

    return

####################################################################################################
 
def novel_gs_scipy_randS():
    
    # ===> Make trial states on Bloch sphere
    points = 10
    rzrx = fibo_bloch(points)

    # ===> Define gateset GS1 (standard gates via U-gate)
    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    tdg_U_mat = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
    Uh = UGate("Uh", h_U_mat)
    Ut = UGate("Ut", t_U_mat)
    Utdg = UGate("Utdg", tdg_U_mat)
    agent1_gateset = [Uh, Ut, Utdg]

    trials, max_depth, recursion_degree = 10, 3, 2

    def cost_to_optimize(thetas):
        return cost_func(rzrx, agent1_gateset, thetas, max_depth=max_depth, recursion_degree=recursion_degree)
    
    cost_list, ang_list = [], []
    for _ in range(trials):
        initial_guess = np.random.random(2*3)
        print('Cost of initial random guess of GS:', cost_to_optimize(initial_guess))
        res = minimize(cost_to_optimize, initial_guess, method = 'COBYLA', options={'maxiter': 200})
        print('Cost of optimized GS:', res.fun)

        cost_list.append(res['fun'])
        ang_list.append(res['x'])
    
    sort_opt_ang = ang_list[ cost_list.index(np.min(cost_list)) ]
    np.save('opt_ang', sort_opt_ang)


    return

####################################################################################################





####################################################################################################

"""
Data Set Generation
"""

# ------------------------------------------------------------------------------------------------ #

"""
Evenly distributed states (using golden mean)
Ref: stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
"""

def gen_ds_fiboS(samples = 100):
    ds = []
    phi = math.pi * (3. - math.sqrt(5.))        # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2    # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)           # radius at y
        theta = phi * i                         # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        ds.append([x,y,z])
    return ds

# ------------------------------------------------------------------------------------------------ #

"""
Haar random states
"""

def gen_ds_randS(samples = 100, dimensions = 1):
    ds = []
    for i in range(samples):
        ds.append(random_statevector(2**dimensions).data)
    return ds

# ------------------------------------------------------------------------------------------------ #

"""
Haar random unitaries
# https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_unitary.html
"""
def gen_ds_randU(samples = 100, dimensions = 1):
    ds = []
    for i in range(samples):
        ds.append(UnitaryGate(random_unitary(2**dimensions),label='RndU'+str(i)))
    return ds

####################################################################################################

"""
Data Set Visualization
"""

# ------------------------------------------------------------------------------------------------ #

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# ------------------------------------------------------------------------------------------------ #

def vis_ds_fiboS(ds):
    b = qt.Bloch()
    b.point_marker = ['o']
    b.point_size = [20]
    samples = len(ds)
    color = []
    for i in range(samples):
        b.add_points(ds[i])
        color.append(rgb_to_hex(int((ds[i][0]+1)*127),int((ds[i][1]+1)*127),int((ds[i][2]+1)*127)))
        
    b.point_color = color
    b.render()
    plt.show()

# ------------------------------------------------------------------------------------------------ #

def vis_ds_randS(ds):
    b = qt.Bloch()
    b.point_marker = ['o']
    b.point_size = [20]
    samples = len(ds)
    color = []
    for i in range(samples):
        b.add_states(qt.Qobj(ds[i]), kind='point')
        _, _, pX = measurement_statistics(qt.Qobj(ds[i]), qt.sigmax())
        _, _, pY = measurement_statistics(qt.Qobj(ds[i]), qt.sigmay())
        _, _, pZ = measurement_statistics(qt.Qobj(ds[i]), qt.sigmaz())
        color.append(rgb_to_hex(int(pX[0]*255),int(pY[0]*255),int(pZ[0]*255)))
        
    b.point_color = color
    b.render()
    plt.show()

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

####################################################################################################

"""
Given an unitary and a gate set, use trials to find a circuit using the gate set that is close to the unitary
"""

def dcmp_rand(randU, gs, trials = 100, max_depth= 100):

    qc0 = QuantumCircuit(1)
    qc0.append(randU,[0])
    choi0 = Choi(qc0)
    
    pfi_best = 0
    dep_best = 0
    qcirc_best = []
    for i in range(trials):
        dep = random.randrange(1,max_depth)
        seq = random.choices(list(gs.keys()), k=dep)
        qc0 = QuantumCircuit(1)
        for i in seq:
            qc0.append(gs[i], [0]) # TBD: 2 qubit gates
        choi01 = Choi(qc0)
        pfi = process_fidelity(choi0,choi01)
        if pfi > pfi_best:
            pfi_best = pfi
            qcirc_best = qc0
            dep_best = dep

    return pfi_best, dep_best, qcirc_best

####################################################################################################

"""
Mode 1: Compare GS2 (in code) w.r.t. GS1 (in code)
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

def compare_gs(yaqq_ds_type, ds, yaqq_dcmp):

    # ===> Define gateset GS1 (standard gates)    
    h_U_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t_U_mat = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
    tdg_U_mat = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
    gs1 = {}    
    gs1['H'] = UnitaryGate(h_U_mat,label='H')
    gs1['T'] = UnitaryGate(t_U_mat,label='T')
    gs1['Tdg'] = UnitaryGate(tdg_U_mat,label='Tdg')
    gs1_gates = ','.join(list(gs1.keys()))

    # ===> Load U3 angles from novel_gs_scipy and Define gateset GS2

    opt_angle_novel_gs = np.load('../opt_ang.npy')
    gs2 = {}    
    gs2['U3a'] = UnitaryGate(qiskit_U3(opt_angle_novel_gs[0], opt_angle_novel_gs[1], opt_angle_novel_gs[2]),label='U3a')
    gs2['U3b'] = UnitaryGate(qiskit_U3( opt_angle_novel_gs[3], opt_angle_novel_gs[4], opt_angle_novel_gs[5]),label='U3b')
    gs2_gates = ','.join(list(gs2.keys()))
    
    samples = len(ds)
    pf01_db, pf02_db = [], []
    dep01_db, dep02_db = [], []

    if yaqq_ds_type == 1: # fibo
        rz_ang_list, rx_ang_list = Cartesian_to_BlochRzRx(ds)

    for i in range(samples):        
        qc0 = QuantumCircuit(1)
        qc0.rz(rz_ang_list[i],0)
        qc0.rx(rx_ang_list[i],0)

        if yaqq_dcmp == 1: # skt
            # OUTSIDE LOOP
            # ===> Define gateset GS1 (standard gates)
            # t = TGate(label="t")
            # tdg = TdgGate(label="tdg")
            # h = HGate(label="h")
            # ===> Define gateset GS1 (standard gates via U-gate)
            # Uh = UGate("Uh", h_U_mat)
            # Ut = UGate("Ut", t_U_mat)
            # Utdg = UGate("Utdg", tdg_U_mat)
            # agent1_gateset = [Uh, Ut, Utdg]
            # U3a = UGate("U3a", qiskit_U3(opt_angle_novel_gs[0], opt_angle_novel_gs[1], opt_angle_novel_gs[2]))
            # U3b = UGate("U3b", qiskit_U3(opt_angle_novel_gs[3], opt_angle_novel_gs[4], opt_angle_novel_gs[5]))
            # agent2_gateset = [U3a, U3b]
            # gbs = gen_basis_seq()
            # max_depth = 3
            # agent1_gateseq = gbs.generate_basic_approximations(agent1_gateset, max_depth)
            # agent2_gateseq = gbs.generate_basic_approximations(agent2_gateset, max_depth)
            # # ===> SK Decompose trail states for both gate sets and calculate fidelity/length
            # # ===> Declare SKT object
            # recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition
            # skd1 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent1_gateseq)    
            # skd2 = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent2_gateseq) 
            # INSIDE LOOP
            #     qc01 = skd1(qc0)
            #     qc02 = skd2(qc0)
            print("Currently not supported")

        elif yaqq_dcmp == 2: # rand
            dsU = Operator(qc0)
            _, _, qc01 = dcmp_rand(dsU,gs1)    # default 100 trials and 100 max_depth
            _, _, qc02 = dcmp_rand(dsU,gs2)    # default 100 trials and 100 max_depth

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
        avg_fid_gs01 = np.mean(pf01_db)         # Not same for ivt_fid_gs01 and pf01_db
        avg_fid_gs02 = np.mean(pf02_db)
        avg_dep_gs01 = np.mean(dep01_db)
        avg_dep_gs02 = np.mean(dep02_db) 
        _, ax = plt.subplots(1, 2)
        ax[0].plot(pf01_db, '-x', color = 'r', label = 'PF ['+gs1_gates+']')
        ax[0].plot(pf02_db, '-o', color = 'b', label = 'PF ['+gs2_gates+']')

        ax[0].axhline(y=avg_fid_gs01, linestyle='-.', color = 'r' , label = 'avg.PF ['+gs1_gates+']')
        ax[0].axhline(y=avg_fid_gs02, linestyle='-.', color = 'b' , label = 'avg.PF ['+gs2_gates+']')

        ax[1].plot(dep01_db, '-x', color = 'r', label = 'CD ['+gs1_gates+']')
        ax[1].plot(dep02_db, '-o', color = 'b', label = 'CD ['+gs2_gates+']')

        ax[1].axhline(y=avg_dep_gs01, linestyle='-.', color = 'r', label = 'avg.CD ['+gs1_gates+']')
        ax[1].axhline(y=avg_dep_gs02, linestyle='-.', color = 'b', label = 'avg.CD ['+gs2_gates+']')

        ax[0].set_ylabel("Process Fidelity")
        ax[1].set_ylabel("Circuit Depth")
        ax[0].set_ylim(bottom=0,top=1)
        ax[1].set_ylim(bottom=0,top=None)
        ax[0].legend()
        ax[1].legend()
        plt.show()


    # _, ax = plt.subplots(1, 2, figsize = (7,3.5), sharex=True, layout="constrained")

    # ax[0].plot(fid_gs01, '-x', label = "[h, t, tdg]")
    # ax[0].plot(fid_gs02, '-o', label = "[U3a, U3b]")
    # ax[1].plot(len_gs01, '-x', label = "[h, t, tdg]")
    # ax[1].plot(len_gs02, '-o', label = "[U3a, U3b]")
    
    # ax[0].set_ylabel("Process Fidelity")
    # ax[1].set_ylabel("Decomposed Circuit Length")
    # ax[0].set_xlabel("Equidistant Points")
    # ax[1].set_xlabel("Equidistant Points")
    # ax[0].set_ylim(bottom=0,top=1)
    # ax[1].set_ylim(bottom=0,top=None)
    # plt.legend(ncol = 2, bbox_to_anchor = (1, 1.13))
    # plt.savefig('figures/lok_dekhano_plot.pdf')
    # plt.savefig('figures/lok_dekhano_plot.png')
    # plt.show()

####################################################################################################

"""
Mode 2: Generative novel GS2 w.r.t. GS1 (in code)
"""

def generate_gs(ds, yaqq_dcmp,yaqq_search):

    # print("Option 2: Generate complementary gate set of GS1 using random search for randS") # SKT
    # print("Option 3: Generate complementary gate set of GS1 using random search for randU")
    # print("Option 4: Generate complementary gate set of GS1 using scipy for randS") # RandDecomposition
    # print("Option 5: Generate complementary gate set of GS1 using scipy for randU") # RandDecomposition
    #     case '2': novel_gs_rand_randS()
    #     case '3': novel_gs_rand_randU()
    #     case '4': novel_gs_scipy_randS()
    #     case '5': novel_gs_scipy_randU()

    return

####################################################################################################        

if __name__ == "__main__":

    
    devmode = input("\n  ===> Run Default Configuration? [Y/N] (def.: Y): ") or 'Y'
    if devmode == 'Y':
        # generate_gs(gen_ds_fiboS(yaqq_ds_size),4)
        compare_gs(yaqq_ds_type=1,ds=gen_ds_fiboS(samples=20),yaqq_dcmp=2)
        exit(1)

    # Note: Currently YAQQ is configured only for 1 qubit
    # TBD: make general 2 qubit gate with 15 free parameters

    yaqq_ds_size = int(input("\n  ===> Enter Data Set Size (def.: 500): ")) or 500
        
    print("\n  ===> Choose Data Set:")
    print("   Data Set 1 - Evenly distributed states (using golden mean)")
    print("   Data Set 2 - Haar random states")
    print("   Data Set 3 - Haar random unitaries")
    yaqq_ds_type = int(input("   Option (def.: 1): ") or 1)
    match yaqq_ds_type:
        case 1: 
            ds = gen_ds_fiboS(samples=yaqq_ds_size)  # Returns list of [x,y,z] coordinates
        case 2: 
            ds = gen_ds_randS(samples=yaqq_ds_size)  # Returns list of state vectors
        case 3: 
            ds = gen_ds_randU(samples=yaqq_ds_size)  # Returns list of unitary gate objects
        case _ : 
            print("Invalid option")
            exit(1)   

    yaqq_ds_show = input("  ===> Plot Data Set? [Y/N] (def.: N): ") or 'N'
    if yaqq_ds_show == 'Y':
        match yaqq_ds_type:
            case 1: 
                vis_ds_fiboS(ds)
            case 2: 
                vis_ds_randS(ds)
            case 3: 
                vis_ds_randU(ds)    # Plots the states when the unitaries are applied to |0> state    

    print("\n  ===> Choose Gate Decomposition Method:")
    print("   Method 1 - Solovay-Kitaev Decomposition")
    print("   Method 2 - Random Decomposition")
    yaqq_dcmp = int(input("   Option (def.: 2): ") or 2)

    print("  ===> Choose YAQQ Mode:")
    print("   Mode 1 - Compare GS2 (in code) w.r.t. GS1 (in code)")
    print("   Mode 2 - Generative novel GS2 w.r.t. GS1 (in code)")
    yaqq_mode = int(input("   Option (def.: 2): ")) or 2
    match yaqq_mode:
        case 1: 
            ds = compare_gs(yaqq_ds_type, ds, yaqq_dcmp)  # Returns list of [x,y,z] coordinates
        case 2: 
            print("\n  ===> Choose Search Method:")
            print("   Method 1 - Random Gate Set Search")
            print("   Method 2 - U3 Angles SciPy L-BFGS-B/COBYLA with Multiple Random Initialization")
            yaqq_search = int(input("   Option (def.: 1): ")) or 1
            ds = generate_gs(yaqq_ds_type, ds, yaqq_dcmp, yaqq_search)  # Returns list of state vectors
        case _ : 
            print("Invalid option")
            exit(1)   

    print("\nThank you for using YAQQ.")