"""
Example code for Generalized Quantum Unitary Decomposition API
"""

import sys
sys.path.insert(0, '..')
from yaqq import yaqq

import numpy as np
from qiskit.quantum_info import random_unitary

g1 = random_unitary(2)
g2 = random_unitary(2)

u_1q = random_unitary(2)
u_2q = random_unitary(4)
u_3q = random_unitary(8)

qq = yaqq('../configs/')

'Test 1: Load gate from file'

# Save gates
# f_gs1 = '../results/data/GQUD_eid-0001_2024-11-15-13-00gs1_1F'
# np.save(f_gs1, g1)
# qq.yaqq_cfg('GQUD_eid-0001')
# decomposed_qc = qq.gqud(u_1q)

'Test 2: Load gate from arguments'

qq.yaqq_cfg('GQUD_eid-0002', gs_arg = [g1, g2])
decomposed_qc = qq.gqud(u_3q)
print(decomposed_qc)
