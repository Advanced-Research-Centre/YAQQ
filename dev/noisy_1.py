from qiskit import transpile, QuantumCircuit
import qiskit.quantum_info as qi

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

circ = QuantumCircuit(1)
ref = QuantumCircuit(1)
ref.x(0)
circ.unitary(qi.Operator(ref), [0], label='h')
print('NON NOISY UNITARY')
circ.measure_all()
ideal_result = AerSimulator().run(circ).result()
ideal_counts = ideal_result.get_counts()

print(ideal_counts)

# Error parameters
param_q0 =  0.5 # damping parameter for qubit-0
param_q1 = 1.0   # damping parameter for qubit-1

# Construct the error
qerror_q0 = amplitude_damping_error(param_q0)

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(qerror_q0, 'h')

cx_circ_noise = transpile(circ, AerSimulator(noise_model=noise_model))

# unitary = qi.Operator(cx_circ_noise)

noise_result = AerSimulator().run(cx_circ_noise).result()
noise_counts = noise_result.get_counts(circ)
print('NOISY UNITARY')
print(noise_counts)