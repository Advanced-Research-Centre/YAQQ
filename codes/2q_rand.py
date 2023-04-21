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
from qiskit.extensions import UnitaryGate, U3Gate
from qiskit.circuit import Gate
from qiskit.quantum_info import random_unitary
import scipy.linalg as la
from qiskit.circuit.gate import Gate
from tqdm import tqdm
from scipy.optimize import minimize 
from typing import Optional, Union
from qiskit.circuit.controlledgate import ControlledGate


import random
from qiskit.quantum_info.operators.random import random_unitary

qb = 1

tgt = UnitaryGate(random_unitary(2**qb, seed=2),label='RndU4')
print(tgt)
qc0 = QuantumCircuit(qb)
qc0.append(tgt, [0])
# qc0.append(tgt, [0,1])
choi0 = Choi(qc0)
print(qc0.draw())

# a1_gs = ['h0', 'h1', 't0', 't1','cx01','cx10']
a1_gs = ['h0', 't0']


pfi_best = 10000
seq_best = []
qcirc_best = []
trials = 50
for i in range(trials):
    # print(i)
    seq = random.choices(a1_gs,k=10)
    # print(seq)
    qc0 = QuantumCircuit(qb)
    for i in seq:
        if i == 'h0':
            qc0.h(0)
        elif i == 'h1':
            qc0.h(1)
        elif i == 't0':
            qc0.t(0)
        elif i == 't1':
            qc0.t(1)
        elif i == 'cx01':
            qc0.cx(0,1)
        elif i == 'cx10':
            qc0.cx(1,0)
    choi01 = Choi(qc0)
    pfi = process_fidelity(choi0,choi01)
    if pfi < pfi_best:
        pfi_best = pfi
        seq_best = seq
        qcirc_best = qc0

print(pfi_best)
print(qcirc_best.draw())

