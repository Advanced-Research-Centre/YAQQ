# Breaking in new Qiskit v1.2.4's rust port. We can reuse Qiskit's inbuild version in next Qiskit release

import math
import cmath
import warnings
import numpy as np

from qiskit.circuit import Gate
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis import TwoQubitWeylDecomposition, OneQubitEulerDecomposer

_ipx = np.array([[0, 1j], [1j, 0]], dtype=complex)
_ipy = np.array([[0, 1], [-1, 0]], dtype=complex)
_ipz = np.array([[1j, 0], [0, -1j]], dtype=complex)
_id = np.array([[1, 0], [0, 1]], dtype=complex)

def rz_array(theta):
    """Return numpy array for Rz(theta).

    Rz(theta) = diag(exp(-i*theta/2),exp(i*theta/2))
    """
    return np.array(
        [[cmath.exp(-1j * theta / 2.0), 0], [0, cmath.exp(1j * theta / 2.0)]], dtype=complex
    )

class kak:

    def __init__(
            self,
            gate: Gate,
            euler_basis: str = "U"
        ):
            self.gate = gate

            basis = self.basis = TwoQubitWeylDecomposition(Operator(gate).data)
            self._decomposer1q = OneQubitEulerDecomposer(euler_basis)

            # FIXME: find good tolerances
            self.is_supercontrolled = math.isclose(basis.a, np.pi / 4) and math.isclose(basis.c, 0.0)

            # Create some useful matrices U1, U2, U3 are equivalent to the basis,
            # expand as Ui = Ki1.Ubasis.Ki2
            b = basis.b
            K11l = (
                1
                / (1 + 1j)
                * np.array(
                    [
                        [-1j * cmath.exp(-1j * b), cmath.exp(-1j * b)],
                        [-1j * cmath.exp(1j * b), -cmath.exp(1j * b)],
                    ],
                    dtype=complex,
                )
            )
            K11r = (
                1
                / math.sqrt(2)
                * np.array(
                    [
                        [1j * cmath.exp(-1j * b), -cmath.exp(-1j * b)],
                        [cmath.exp(1j * b), -1j * cmath.exp(1j * b)],
                    ],
                    dtype=complex,
                )
            )
            K12l = 1 / (1 + 1j) * np.array([[1j, 1j], [-1, 1]], dtype=complex)
            K12r = 1 / math.sqrt(2) * np.array([[1j, 1], [-1, -1j]], dtype=complex)
            K32lK21l = (
                1
                / math.sqrt(2)
                * np.array(
                    [
                        [1 + 1j * np.cos(2 * b), 1j * np.sin(2 * b)],
                        [1j * np.sin(2 * b), 1 - 1j * np.cos(2 * b)],
                    ],
                    dtype=complex,
                )
            )
            K21r = (
                1
                / (1 - 1j)
                * np.array(
                    [
                        [-1j * cmath.exp(-2j * b), cmath.exp(-2j * b)],
                        [1j * cmath.exp(2j * b), cmath.exp(2j * b)],
                    ],
                    dtype=complex,
                )
            )
            K22l = 1 / math.sqrt(2) * np.array([[1, -1], [1, 1]], dtype=complex)
            K22r = np.array([[0, 1], [-1, 0]], dtype=complex)
            K31l = (
                1
                / math.sqrt(2)
                * np.array(
                    [[cmath.exp(-1j * b), cmath.exp(-1j * b)], [-cmath.exp(1j * b), cmath.exp(1j * b)]],
                    dtype=complex,
                )
            )
            K31r = 1j * np.array([[cmath.exp(1j * b), 0], [0, -cmath.exp(-1j * b)]], dtype=complex)
            K32r = (
                1
                / (1 - 1j)
                * np.array(
                    [
                        [cmath.exp(1j * b), -cmath.exp(-1j * b)],
                        [-1j * cmath.exp(1j * b), -1j * cmath.exp(-1j * b)],
                    ],
                    dtype=complex,
                )
            )
            k1ld = basis.K1l.T.conj()
            k1rd = basis.K1r.T.conj()
            k2ld = basis.K2l.T.conj()
            k2rd = basis.K2r.T.conj()

            # Pre-build the fixed parts of the matrices used in 3-part decomposition
            self.u0l = K31l.dot(k1ld)
            self.u0r = K31r.dot(k1rd)
            self.u1l = k2ld.dot(K32lK21l).dot(k1ld)
            self.u1ra = k2rd.dot(K32r)
            self.u1rb = K21r.dot(k1rd)
            self.u2la = k2ld.dot(K22l)
            self.u2lb = K11l.dot(k1ld)
            self.u2ra = k2rd.dot(K22r)
            self.u2rb = K11r.dot(k1rd)
            self.u3l = k2ld.dot(K12l)
            self.u3r = k2rd.dot(K12r)

            # Pre-build the fixed parts of the matrices used in the 2-part decomposition
            self.q0l = K12l.T.conj().dot(k1ld)
            self.q0r = K12r.T.conj().dot(_ipz).dot(k1rd)
            self.q1la = k2ld.dot(K11l.T.conj())
            self.q1lb = K11l.dot(k1ld)
            self.q1ra = k2rd.dot(_ipz).dot(K11r.T.conj())
            self.q1rb = K11r.dot(k1rd)
            self.q2l = k2ld.dot(K12l)
            self.q2r = k2rd.dot(K12r)

            # Decomposition into different number of gates
            # In the future could use different decomposition functions for different basis classes, etc
            # if not self.is_supercontrolled:
            #     warnings.warn(
            #         "Only know how to decompose properly for supercontrolled basis gate. "
            #         "This gate is ~Ud({}, {}, {})".format(basis.a, basis.b, basis.c),
            #         stacklevel=2,
            #     )

            # self.decomposition_fns = [
            #     self.decomp0,
            #     self.decomp1,
            #     self.decomp2_supercontrolled,
            #     self.decomp3_supercontrolled,
            # ]
            self._rqc = None

    def decomp3_supercontrolled(self, target):
        """Decompose target with 3 uses of the basis.
        This is an exact decomposition for supercontrolled basis ~Ud(pi/4, b, 0), all b,
        and any target. No guarantees for non-supercontrolled basis."""

        U0l = target.K1l.dot(self.u0l)
        U0r = target.K1r.dot(self.u0r)
        U1l = self.u1l
        U1r = self.u1ra.dot(rz_array(-2 * target.c)).dot(self.u1rb)
        U2l = self.u2la.dot(rz_array(-2 * target.a)).dot(self.u2lb)
        U2r = self.u2ra.dot(rz_array(2 * target.b)).dot(self.u2rb)
        U3l = self.u3l.dot(target.K2l)
        U3r = self.u3r.dot(target.K2r)

        return U3r, U3l, U2r, U2l, U1r, U1l, U0r, U0l






from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary

def unit_test():

    U = random_unitary(4)
    bg = random_unitary(4)
    kak_obj = kak(bg)

    ds_kak_1q = kak_obj.decomp3_supercontrolled(TwoQubitWeylDecomposition(U)) 

    # Decompose 1 qubit gates in ds_kak_1q
    ds_kak_1q_gs = []
    ctr = 0
    for U_kak_1q in ds_kak_1q:
        # _, _, qcirc_KAK_1q = self.dcmp_U_gs(UnitaryGate(Operator(U_kak_1q),label='KAK_1q'), gs, gsid)
        ds_kak_1q_gs.append([UnitaryGate(Operator(U_kak_1q),label='KAK_1q_'+str(ctr))])
        ctr += 1

    # Replace the gates in KAK with decomposed circuits
    qc = QuantumCircuit(2)
    for g_gs in ds_kak_1q_gs[0]:
        qc.append(g_gs, [0])
    for g_gs in ds_kak_1q_gs[1]:
        qc.append(g_gs, [1])
    qc.append(bg,[0,1])
    for g_gs in ds_kak_1q_gs[2]:
        qc.append(g_gs, [0])
    for g_gs in ds_kak_1q_gs[3]:
        qc.append(g_gs, [1])
    qc.append(bg,[0,1])
    for g_gs in ds_kak_1q_gs[4]:
        qc.append(g_gs, [0])
    for g_gs in ds_kak_1q_gs[5]:
        qc.append(g_gs, [1])
    qc.append(bg,[0,1])
    for g_gs in ds_kak_1q_gs[6]:
        qc.append(g_gs, [0])
    for g_gs in ds_kak_1q_gs[7]:
        qc.append(g_gs, [1])

    print(qc)