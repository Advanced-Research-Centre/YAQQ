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
from qiskit.quantum_info import Pauli
# from math import pi
import numpy as np

import scipy as sp

from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import random_unitary

# Ref: https://threeplusone.com/pubs/on_gates.pdf
def Can():
    # SWAP
    tx = 0.5  # [0,1]
    ty = 0.5  # [0,0.5]
    tz = 0.5  # [0]

    
    tx = 0.5  # [0,1]
    ty = 0  # [0,0.5]
    tz = 0  # [0]

    # print(XGate().to_matrix())
    S = tx*Pauli('XX').to_matrix() + ty*Pauli('YY').to_matrix() + tz*Pauli('ZZ').to_matrix()
    can = sp.linalg.expm(-1j*S*np.pi/2)
    gp = can * np.exp(1j*np.pi/4)       # * np.sqrt(2)
    # gp = can * np.exp(1j*np.pi/2)       # * np.sqrt(2)
    # print(S)
    print(np.matrix(can).getH() * np.matrix(can))
    print()
    print(np.round(gp,3))
    # print(xx.to_matrix())

def gen_ds_randU(samples = 1, dimensions = 2):
    ds = []
    for i in range(samples):
        ds.append(UnitaryGate(random_unitary(2**dimensions),label='RndU'+str(i)))
    return ds

print(gen_ds_randU().)
Can()