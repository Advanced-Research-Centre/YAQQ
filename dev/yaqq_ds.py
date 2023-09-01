from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, random_unitary, Operator
from qiskit.extensions import UnitaryGate
import random
import math
from astropy.coordinates import cartesian_to_spherical
import weylchamber
import numpy as np
from itertools import product

class GenerateDataSet:

    # ------------------------------------------------------------------------------------------------ #

    def yaqq_gen_ds(self, ds_dim, ds_type, ds_size, ds_reso):
        if ds_dim == 2 and ds_type == 4:
            ds = self.gen_ds_equiNL(ds_reso)
            ds_size = len(ds)
            print("\n  ===> YAQQ Data Set Generated for Dimension =", ds_dim, "Type =", ds_type, "Spacing =", ds_reso, "Size =", ds_size)
        else:
            if ds_type == 1:
                ds = self.gen_ds_randS(ds_dim,  ds_size)
            elif ds_type == 2:
                ds = self.gen_ds_randU(max_dim = ds_dim,  ds_size = ds_size, rand_dim = False)
            elif ds_dim == 1 and ds_type == 3:
                ds = self.gen_ds_fiboS(ds_size)
            elif ds_dim == 2 and ds_type == 3:
                ds = self.gen_ds_randNL(ds_size)
            print("\n  ===> YAQQ Data Set Generated for Dimension =", ds_dim, "Type =", ds_type, "Size =", ds_size)
        return ds

    # ------------------------------------------------------------------------------------------------ #

    """
    Data Set Generation: Haar Random n-qubit pure States
    Ref: https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_statevector.html
    """

    def gen_ds_randS(self, ds_dim = 1,  ds_size = 100):
        ds = []
        for i in range(ds_size):
            qc = QuantumCircuit(ds_dim)
            randS = random_statevector(2**ds_dim).data
            qc.prepare_state(randS, list(range(0, ds_dim)))   
            randU_0 = Operator.from_circuit(qc)
            ds.append(UnitaryGate(randU_0,label='RndU'+str(i)))
        return ds
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Data Set Generation: Haar Random 2^nx2^n Unitaries
    Ref: https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_unitary.html
    """

    def gen_ds_randU(self, max_dim = 1, ds_size = 100, rand_dim = False):
        ds = []
        for i in range(ds_size):
            dim = max_dim
            if rand_dim:
                dim =  random.randrange(1,max_dim+1)    # TBD: Samples should be proportional to dimension instead of uniform, i.e. exponentially more higher dimension than lower dimensions
            ds.append(UnitaryGate(random_unitary(2**dim),label='RndU'+str(i)))
        return ds
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Data Set Generation: Equispaced States on Bloch Sphere using Golden mean
    Ref: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """

    def gen_ds_fiboS(self, ds_size = 100):
        ds = []
        phi = math.pi * (3. - math.sqrt(5.))        # golden angle in radians
        for i in range(ds_size):
            y = 1 - (i / float(ds_size - 1)) * 2    # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)           # radius at y
            theta = phi * i                         # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            sphe_coor = cartesian_to_spherical(x, y, z)
            z_ang = sphe_coor[1].radian+math.pi/2
            x_ang = sphe_coor[2].radian
            qc = QuantumCircuit(1)       
            qc.ry(z_ang,0)  # To rotate state z_ang from |0>, rotate about Y
            qc.rz(x_ang,0)  # To rotate state x_ang from |+>, rotate about Z
            fiboU_0 = Operator.from_circuit(qc)      
            ds.append(UnitaryGate(fiboU_0,label='FiboU'+str(i)))
        return ds
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Data Set Generation: Random Non-local Unitaries on Weyl chamber
    Ref: https://weylchamber.readthedocs.io/en/latest/API/weylchamber.coordinates.html#weylchamber.coordinates.random_gate
    """

    def gen_ds_randNL(self, ds_size = 100):
        ds = []
        for i in range(ds_size):
            ds.append(UnitaryGate(weylchamber.random_gate(),label='RndU'+str(i)))    
        return ds
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Data Set Generation: Equispaced non-local unitaries
    Ref: https://weylchamber.readthedocs.io/en/latest/API/weylchamber.coordinates.html#weylchamber.coordinates.point_in_weyl_chamber
    """

    def gen_ds_equiNL(self, px = 23):
        ds = []
        cx = np.linspace(0, 1, px)
        cy = np.linspace(0, 0.5, int(px/2))
        cz = np.linspace(0, 0.5, int(px/2))
        gs = product(cx, cy, cz)
        valid_points = 0
        for can in gs:
            # Enumerate points in the Weyl chamber
            c = list(can)
            c1,c2,c3 = c[0],c[1],c[2]
            if weylchamber.point_in_weyl_chamber(c1,c2,c3):
                valid_points+= 1
                ds.append(UnitaryGate(weylchamber.canonical_gate(c1,c2,c3),label='RndU'+str(valid_points)))   
        return ds
    
    # ------------------------------------------------------------------------------------------------ #