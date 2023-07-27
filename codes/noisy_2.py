from qiskit import QuantumCircuit
from math import pi
from qiskit.quantum_info import process_fidelity, Choi
import numpy as np

qc0 = QuantumCircuit(1)
qc0.rz(pi/5,0)
qc0.rx(pi/9,0)
choi0 = Choi(qc0)


from qiskit.providers.aer import noise
from qiskit.providers.aer.utils import insert_noise
import qiskit.quantum_info as qi

noise_model = noise.NoiseModel()
# Noise parameters
delta = 1e-3
kappa = 1e-3
gate_time = np.pi / 2.0
param_amplitude = 1 - np.exp(-kappa * gate_time)
param_phase = 1 - np.exp(-4 * delta * gate_time)
error = noise.phase_amplitude_damping_error(param_amplitude, param_phase, 1)
# print("Noise superop:\n\n", qi.SuperOp(error.to_instruction()).data)

noise_model.add_all_qubit_quantum_error(error, ["rz"])
noisy_circuit = insert_noise(qc0, noise_model, transpile=True)
superop = qi.SuperOp(noisy_circuit.to_instruction())

# print("\nNoisy X gate superop:\n\n", superop.data)

choi01 = Choi(noisy_circuit.to_instruction())
pfi = process_fidelity(choi0,choi01)

print(pfi)