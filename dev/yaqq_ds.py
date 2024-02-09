from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, random_unitary, Operator, Statevector
from qiskit.extensions import UnitaryGate
import random
import math
from astropy.coordinates import cartesian_to_spherical
import weylchamber
import numpy as np
from itertools import product
import qutip as qt
from qutip.measurement import measurement_statistics
import matplotlib.pyplot as plt
from datetime import datetime

class GenerateDataSet:

    # ------------------------------------------------------------------------------------------------ #

    def load_ds(self):
             
        ds_fname = input("\n  ===> Enter Dataset Filename (e.g.: NUSA_eid-0010_2024-02-09-11-26ds): " or 'NUSA_eid-0010_2024-02-09-11-26ds')
        ds = np.load('results/data/'+ds_fname+'.npy', allow_pickle=True)
        ds_dim = len(ds[0].to_matrix())
        ds_size = len(ds)
        print("\n  ===> YAQQ Data Set Loaded = "+str(ds_dim)+", Size = "+str(ds_size))

        return ds_dim, ds
    
    # ------------------------------------------------------------------------------------------------ #

    def yaqq_gen_ds(self, ds_dim, ds_type, ds_size, ds_reso):

        if ds_dim == 1 and ds_type == 4:
            ds = self.gen_ds_equiA(ds_reso)
            ds_size = len(ds)
            print("\n  ===> YAQQ Data Set Generated for Dimension =", ds_dim, "Type =", ds_type, "Spacing =", ds_reso, "Size =", ds_size)
        elif ds_dim == 2 and ds_type == 4:
            ds = self.gen_ds_equiNL(ds_reso)
            ds_size = len(ds)
            print("\n  ===> YAQQ Data Set Generated for Dimension =", ds_dim, "Type =", ds_type, "Spacing =", ds_reso, "Size =", ds_size)
        elif ds_dim == 1 and ds_type == 5:
            ds = self.gen_ds_quantumness()
            ds_size = len(ds)
            print("\n  ===> YAQQ Data Set Generated for Dimension = "+str(ds_dim)+", Type = "+str(ds_type)+", Size = "+str(ds_size))
        else:
            if ds_type == 1:
                ds = self.gen_ds_randS(ds_dim,  ds_size)
            elif ds_type == 2:
                ds = self.gen_ds_randU(max_dim = ds_dim,  ds_size = ds_size, rand_dim = False)
            elif ds_dim == 1 and ds_type == 3:
                ds = self.gen_ds_fiboS(ds_size)
            elif ds_dim == 2 and ds_type == 3:
                ds = self.gen_ds_randNL(ds_size)
            print("\n  ===> YAQQ Data Set Generated for Dimension = "+str(ds_dim)+", Type = "+str(ds_type)+", Size = "+str(ds_size))

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
            fiboU = Operator.from_circuit(qc)      
            ds.append(UnitaryGate(fiboU,label='FiboU'+str(i)))

        return ds
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Data Set Generation: Equispaced angles on Bloch Sphere
    parameters a_rz and a_rx are re-interpreted in spherical coordinates as respectively the colatitude with respect to the z-axis and the longitude with respect to the x-axis
    """

    def gen_ds_equiA(self, px = 20):

        ds = []
        a_rz = np.linspace(0, math.pi, px, endpoint=False)
        a_rx = np.linspace(0, 2*math.pi, 2*px, endpoint=False)
        ang_ds = product(a_rz, a_rx)
        points = 0
        for ang in ang_ds:
            # Enumerate points in the Bloch sphere
            qc = QuantumCircuit(1)       
            qc.ry(ang[0],0)  # To rotate state z_ang from |0>, rotate about Y
            qc.rz(ang[1],0)  # To rotate state x_ang from |+>, rotate about Z
            points+= 1
            equiU = Operator.from_circuit(qc)      
            ds.append(UnitaryGate(equiU,label='EquiU'+str(points)))

        return ds

    # ------------------------------------------------------------------------------------------------ #

    """
    Data Set Generation: Equispaced Haar random on Bloch Sphere
    TBD: need to debug
    """

    def gen_ds_equiU(self, px = 5):

        ds = []
        points = 0
        param_phi = 2 * np.pi * np.linspace(0, 1, px, endpoint=False)
        param_others = np.arcsin(np.sqrt(np.linspace(0, 1, px, endpoint=False)))
        param = product(param_others,param_others,param_others,param_phi)   # alpha, psi, chi
        print(len(list(param)))
        for p in param:
            U = np.exp(1j*p[0])*np.array([[np.exp(1j*p[1])*np.cos(p[3]), np.exp(1j*p[2])*np.sin(p[3])],[-np.exp(-1j*p[2])*np.sin(p[3]), np.exp(-1j*p[1])*np.cos(p[3])]])
            points+= 1    
            ds.append(UnitaryGate(U,label='EquiU'+str(points)))

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

    """
    Data Set Generation: Magic and Stabilizer States
    Ref: https://earltcampbell.com/research/magic-states/
    """

    def gen_ds_quantumness(self):

        ds = []

        # Stabilizer states
        SS = [[0,1],                            # +Z
              [1,0],                            # -Z
              [1/np.sqrt(2),1/np.sqrt(2)],      # +X
              [1/np.sqrt(2),-1/np.sqrt(2)],     # -X
              [1/np.sqrt(2),1j/np.sqrt(2)],     # +Y
              [1/np.sqrt(2),-1j/np.sqrt(2)]]    # -Y
        for i in range(len(SS)):
            qc = QuantumCircuit(1)
            qc.prepare_state(SS[i], [0])
            ds.append(UnitaryGate(Operator.from_circuit(qc),label='SS'+str(i)))
          
        # Magic states
        qc = QuantumCircuit(1)                  # mid(+X,+Y,+Z)
        qc.ry(np.pi/(2*np.sqrt(3)),0)
        qc.p(np.pi/4,0)
        ds.append(UnitaryGate(Operator.from_circuit(qc),label='MS0'))
        qc = QuantumCircuit(1)                  # mid(+X,+Y,-Z)
        qc.ry(np.pi - np.pi/(2*np.sqrt(3)),0)
        qc.p(np.pi/4,0)
        ds.append(UnitaryGate(Operator.from_circuit(qc),label='MS1'))
        qc = QuantumCircuit(1)                  # mid(+X,-Y,+Z)
        qc.ry(np.pi/(2*np.sqrt(3)),0)
        qc.p(-np.pi/4,0)
        ds.append(UnitaryGate(Operator.from_circuit(qc),label='MS2'))
        qc = QuantumCircuit(1)                  # mid(+X,-Y,-Z)
        qc.ry(np.pi - np.pi/(2*np.sqrt(3)),0)
        qc.p(-np.pi/4,0)
        ds.append(UnitaryGate(Operator.from_circuit(qc),label='MS3'))
        
        qc = QuantumCircuit(1)                  # mid(-X,+Y,+Z)
        qc.ry(np.pi/(2*np.sqrt(3)),0)
        qc.p(np.pi - np.pi/4,0)
        ds.append(UnitaryGate(Operator.from_circuit(qc),label='MS4'))
        qc = QuantumCircuit(1)                  # mid(-X,+Y,-Z)
        qc.ry(np.pi - np.pi/(2*np.sqrt(3)),0)
        qc.p(np.pi - np.pi/4,0)
        ds.append(UnitaryGate(Operator.from_circuit(qc),label='MS5'))
        qc = QuantumCircuit(1)                  # mid(-X,-Y,+Z)
        qc.ry(np.pi/(2*np.sqrt(3)),0)
        qc.p(np.pi + np.pi/4,0)
        ds.append(UnitaryGate(Operator.from_circuit(qc),label='MS6'))
        qc = QuantumCircuit(1)                  # mid(-X,-Y,-Z)
        qc.ry(np.pi - np.pi/(2*np.sqrt(3)),0)
        qc.p(np.pi + np.pi/4,0)
        ds.append(UnitaryGate(Operator.from_circuit(qc),label='MS7'))

        return ds
    
    # ------------------------------------------------------------------------------------------------ #

    def save_ds(self, yaqq_ds, Config = None):
             
        now = datetime.now()
        exp_id = Config['general']['exp_name']+'_eid-'+Config['general']['exp_id']+'_'+now.strftime("%Y-%m-%d-%H-%M")
        np.save('results/data/'+exp_id+'ds', yaqq_ds)
        return

    # ------------------------------------------------------------------------------------------------ #

class VisualizeDataSet:

    # ------------------------------------------------------------------------------------------------ #

    def rgb_to_hex(self, r, g, b):

        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    # ------------------------------------------------------------------------------------------------ #

    def vis_ds_Bloch(self, ds):

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
            color.append(self.rgb_to_hex(int(pX[0]*255),int(pY[0]*255),int(pZ[0]*255)))
            
        b.point_color = color
        b.render()
        plt.show()

        return

    # ------------------------------------------------------------------------------------------------ #

    def vis_ds_Weyl(self, ds):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in ds:
            c1, c2, c3 = weylchamber.c1c2c3(i.to_matrix())
            if weylchamber.point_in_weyl_chamber(c1,c2,c3):
                ax.scatter(c1, c2, c3, s=2, c=self.rgb_to_hex(255-int(255*(math.cos(c1*2*math.pi)+1)/2),255-int(255*(math.cos(c2*2*math.pi)+1)/2),255-int(255*(math.cos(c3*2*math.pi)+1)/2)))
            
        ax.plot3D([0,0.5],[0,0.5],[0,0.5],linestyle='--',color='black')       # Isotropic exchange Swap-alpha gates
        ax.plot3D([1,0.5],[0,0.5],[0,0.5],linestyle='--',color='black')       # Isotropic exchange Swap-alpha gates 
        ax.plot3D([0.5,0.5],[0.5,0.5],[0,0.5],linestyle='--',color='black')   # Parameterized Swap gates
        ax.plot3D([0,1],[0,0],[0,0],linestyle='--',color='black')             # Ising gates
        ax.plot3D([0,0.5],[0,0.5],[0,0],linestyle='--',color='black')         # XY gates
        ax.plot3D([1,0.5],[0,0.5],[0,0],linestyle='--',color='black')         # XY gates
        # Perfect Entangler boundary
        ax.plot3D([0.25,0.75],[0.25,0.25],[0.25,0.25],linestyle=':',color='grey')
        ax.plot3D([0.25,0.5],[0.25,0],[0.25,0],linestyle=':',color='grey')
        ax.plot3D([0.5,0.75],[0,0.25],[0,0.25],linestyle=':',color='grey')
        ax.plot3D([0.25,0.25],[0.25,0.25],[0.25,0],linestyle=':',color='grey')
        ax.plot3D([0.75,0.75],[0.25,0.25],[0.25,0],linestyle=':',color='grey')
        ax.plot3D([0.5,0.25],[0,0.25],[0,0],linestyle=':',color='grey')
        ax.plot3D([0.5,0.75],[0,0.25],[0,0],linestyle=':',color='grey')
        ax.plot3D([0.25,0.5],[0.25,0.5],[0.25,0],linestyle=':',color='grey')
        ax.plot3D([0.75,0.5],[0.25,0.5],[0.25,0],linestyle=':',color='grey')

        ax.view_init(elev=15, azim=-55)
        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                            tmp_planes[0], tmp_planes[1], 
                            tmp_planes[4], tmp_planes[5])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
        ax.zaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel('$c_1/\pi$')
        ax.set_ylabel('$c_2/\pi$')
        ax.zaxis.set_rotate_label(False) 
        ax.set_zlabel('$c_3/\pi$', rotation=90)
        # ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        plt.show()

        return

    # ------------------------------------------------------------------------------------------------ #

class ResultsPlotSave:

    # ------------------------------------------------------------------------------------------------ #

    def rgb_to_hex(self, r, g, b):

        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
    # ------------------------------------------------------------------------------------------------ #

    def plot_compare_gs(self, gs1, gs1_gates, pf1, cd1, gs2, gs2_gates, pf2, cd2, pfivt, autocfg, Config = None):
        
        avg_fid_gs01 = np.mean(pf1)
        avg_fid_gs02 = np.mean(pf2)
        avg_dep_gs01 = np.mean(cd1)
        avg_dep_gs02 = np.mean(cd2) 
        
        ivt_fid_gs01 = np.subtract(1,pf1)

        _, ax = plt.subplots(1, 2)
        ax[0].plot(pf1, '-x', color = 'r', label = 'PF ['+gs1_gates+']')
        ax[0].plot(pf2, '-o', color = 'b', label = 'PF ['+gs2_gates+']')
        if pfivt:
            ax[0].plot(ivt_fid_gs01, ':', color = 'g', label = 'target PF trend')

        ax[0].axhline(y=avg_fid_gs01, linestyle='-.', color = 'r' , label = 'avg.PF ['+gs1_gates+']')
        ax[0].axhline(y=avg_fid_gs02, linestyle='-.', color = 'b' , label = 'avg.PF ['+gs2_gates+']')

        ax[1].plot(cd1, '-x', color = 'r', label = 'CD ['+gs1_gates+']')
        ax[1].plot(cd2, '-o', color = 'b', label = 'CD ['+gs2_gates+']')

        ax[1].axhline(y=avg_dep_gs01, linestyle='-.', color = 'r', label = 'avg.CD ['+gs1_gates+']')
        ax[1].axhline(y=avg_dep_gs02, linestyle='-.', color = 'b', label = 'avg.CD ['+gs2_gates+']')

        ax[0].set_ylabel("Process Fidelity")
        ax[1].set_ylabel("Circuit Depth")
        ax[0].set_ylim(bottom=0,top=1)
        ax[1].set_ylim(bottom=0,top=None)
        ax[0].legend()
        ax[1].legend()
        # _, ax = plt.subplots(1, 2, figsize = (7,3.5), sharex=True, layout="constrained")
        # ax[0].set_xlabel("Equidistant Points")
        # ax[1].set_xlabel("Equidistant Points")
        # plt.legend(ncol = 2, bbox_to_anchor = (1, 1.13))

        if autocfg:
            save_res = Config['result']['yaqq_plt_save']
            now = datetime.now()
            exp_id = Config['general']['exp_name']+'_eid-'+Config['general']['exp_id']+'_'+now.strftime("%Y-%m-%d-%H-%M")
        else:
            save_res = input("Save plots and data? [Y/N] (def.: N): ") or 'N'
            if save_res == 'Y':
                exp_id = input("Enter experiment ID: ") or 'exp_1'
        
        if save_res == 'Y':
            plt.savefig('results/figures/'+exp_id+'.pdf')
            plt.savefig('results/figures/'+exp_id+'.png')      
            np.save('results/data/'+exp_id+'gs1', gs1)
            np.save('results/data/'+exp_id+'pf1', pf1)
            np.save('results/data/'+exp_id+'cd1', cd1)
            np.save('results/data/'+exp_id+'gs2', gs2)
            np.save('results/data/'+exp_id+'pf2', pf2)
            np.save('results/data/'+exp_id+'cd2', cd2)

        plt.show()

        # ------------------------------------------------------------------------------------------------ #

    def vis_pf_Bloch(self, ds, pf):

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
            # color.append(self.rgb_to_hex(int((pf[i]-min(pf))*255/(max(pf)-min(pf))),int(pf[i]*255),int(pf[i]*255)))
            color.append(self.rgb_to_hex(int(pf[i]*255),int(pf[i]*255),int(pf[i]*255))) # TBD: https://matplotlib.org/stable/users/explain/colors/colorbar_only.html#sphx-glr-users-explain-colors-colorbar-only-py
            
        b.point_color = color
        b.render()
        plt.show()

        return
