from qiskit.transpiler.passes.synthesis import SolovayKitaev
# from qiskit.transpiler.passes import SolovayKitaevSynthesis, UnitarySynthesis
# from qiskit.quantum_info import Operator
from qiskit import *
import numpy as np
from astropy.coordinates import cartesian_to_spherical
# from qiskit import QuantumCircuit, Aer, transpile, assemble
# from qiskit.circuit.library import TGate, HGate, TdgGate
# import matplotlib.pyplot as plt
from qiskit.quantum_info import process_fidelity, Choi

from qiskit.synthesis.discrete_basis.generate_basis_approximations import generate_basic_approximations

# from qiskit.synthesis.discrete_basis.generate_basis_approximations import (
#     generate_basic_approximations,
# )

# basis_gates = ["t", "h","tdg"]
# max_depth = 3 # maximum number of same consequetive gate allowed
# agent_approximations = generate_basic_approximations(basis_gates, max_depth)    
# for i in agent_approximations:
#     print(i.name)

def equivalent_decomposition(qc,gs):
    """
    INPUT
    ------
    qc : QuantumCircuit
    
    OUTPUT
    ------
    tuple : (decomposed circuit, depth)
    """
    max_depth = 3 # maximum number of same consequetive gates allowed
    recursion_degree = 3 # larger recursion depth increases the accuracy and length of the decomposition
    agent_approximations = generate_basic_approximations(gs, max_depth) 
    skd = SolovayKitaev(recursion_degree=recursion_degree,basic_approximations=agent_approximations)
    return skd(qc)

qc0 = QuantumCircuit(1)
# equiv_qc.x(0)
# equiv_qc.z(0)
qc0.rz(0.2,0)
# equiv_qc.rx(0.4,0)

agent01_gateset = ["t", "h","tdg"]
qc01 = equivalent_decomposition(qc0,agent01_gateset)
# print(qc1)
print(qc01.depth())

agent02_gateset = ["s", "h","tdg"]
qc02 = equivalent_decomposition(qc0,agent02_gateset)
print(qc02.depth())

choi0 = Choi(qc0)
choi01 = Choi(qc01)
choi02 = Choi(qc02)

pf01 = process_fidelity(choi0,choi01)
pf02 = process_fidelity(choi0,choi02)

print(pf01,pf02)

exit(1)

# TODO

# side quest: Check how fibonacci points depths change with different gateset
# ⍰ find how to set arbitrary unitary as gate set
# ☑ find how well it is approximating (process distance)
# find a gate set "new_gs" such that, on atleast 1 fibo point, it is better than "stand_gs" [h,t,tdag]
# find how the new_gs compares with stand_gs for all fibonacci points

def solovay_kiteav_decomposition(qc):
    """
    INPUT
    ------
    qc : QuantumCircuit
    
    OUTPUT
    ------
    tuple : (decomposed circuit, depth)
    """
    skd = SolovayKitaev(recursion_degree=3)
    dc = skd(qc)
    return dc, dc.depth()

import math
def fibonacci_sphere(samples=1000):

    rz_angle,rx_angle, x_points, y_points, z_points  = [],[],[],[],[]
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        x_points.append(x)
        y_points.append(y)
        z_points.append(z)
        sphe_coor = cartesian_to_spherical(x, y, z)
        rz_angle.append(sphe_coor[1].radian+np.pi/2)
        rx_angle.append(sphe_coor[2].radian)
    return rz_angle, rx_angle, x_points, y_points, z_points

points = 10
rz_ang_list, rx_ang_list = fibonacci_sphere(points)[0], fibonacci_sphere(points)[1]

depth_list = []
for p in range(points):
    equiv_qc = QuantumCircuit(1)
    equiv_qc.rz(rz_ang_list[p],0)
    equiv_qc.rx(rx_ang_list[p],0)
    qc, depth = solovay_kiteav_decomposition(equiv_qc)
#     print(depth)
    # print(qc)
    print(equiv_qc)
    depth_list.append(depth)

print(depth_list)