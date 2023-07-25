from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector

ds = []
for i in range(1):
    ds.append(random_statevector(2**1).data)

print(ds[0])
qc = QuantumCircuit(1)
initial_state = ds[0]
qc.initialize(initial_state, 0)

print(qc)

from qiskit import execute, Aer
usimulator=Aer.get_backend('unitary_simulator')
job = execute(qc, usimulator)

umatrix = job.result().get_unitary(qc)
print(umatrix) 


qc1 = QuantumCircuit(1)
qc1.unitary(umatrix.data,[0])
simulator=Aer.get_backend('statevector_simulator')
job = execute(qc1, simulator)
qc_state = job.result().get_statevector(qc1)
print(qc_state)