# This is to analyse the results for novel gate set.

import matplotlib.pyplot as plt
import weylchamber
from qiskit.quantum_info import random_unitary
from qiskit.extensions import UnitaryGate
import math
import numpy as np

def rgb_to_hex(r, g, b):
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def vis_Weyl(ds):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for t in range(1,3):
        print(t,min([z for [x,y,z] in ds if x == t]),max([z for [x,y,z] in ds if x == t]))

    trial = 1
    ds_trial = [y['3NL2'].params[0] for [x,y,z] in ds if x == trial]
    ds_sz = len(ds_trial)
    print(ds_sz)

    for i in range(ds_sz-1):
        c1, c2, c3 = weylchamber.c1c2c3(ds_trial[i])
        c1n, c2n, c3n = weylchamber.c1c2c3(ds_trial[i+1])
        if weylchamber.point_in_weyl_chamber(c1,c2,c3) and weylchamber.point_in_weyl_chamber(c1n,c2n,c3n):
            ax.plot3D([c1,c1n],[c2,c2n],[c3,c3n],linestyle='dotted',linewidth=1,color = rgb_to_hex(255-int(255*i/ds_sz),int(255*i/ds_sz),0))
        if i == 0:
            ax.scatter(c1, c2, c3, s=10, marker = '.', color=rgb_to_hex(0, 255, 0))
        if i == ds_sz-2:
            ax.scatter(c1n, c2n, c3n, s=10, marker = '>', color=rgb_to_hex(255, 0, 0))
        
    ax.plot3D([0,0.5],[0,0.5],[0,0.5],linestyle='--',color='black')     # Isotropic exchange Swap-alpha gates
    ax.plot3D([1,0.5],[0,0.5],[0,0.5],linestyle='--',color='black')     # Isotropic exchange Swap-alpha gates 
    ax.plot3D([0.5,0.5],[0.5,0.5],[0,0.5],linestyle='--',color='black') # Parameterized Swap gates
    ax.plot3D([0,1],[0,0],[0,0],linestyle='--',color='black')           # Ising gates
    ax.plot3D([0,0.5],[0,0.5],[0,0],linestyle='--',color='black')       # XY gates
    ax.plot3D([1,0.5],[0,0.5],[0,0],linestyle='--',color='black')       # XY gates
    # # Perfect Entangler boundary
    # ax.plot3D([0.25,0.75],[0.25,0.25],[0.25,0.25],linestyle=':',color='grey')
    # ax.plot3D([0.25,0.5],[0.25,0],[0.25,0],linestyle=':',color='grey')
    # ax.plot3D([0.5,0.75],[0,0.25],[0,0.25],linestyle=':',color='grey')
    # ax.plot3D([0.25,0.25],[0.25,0.25],[0.25,0],linestyle=':',color='grey')
    # ax.plot3D([0.75,0.75],[0.25,0.25],[0.25,0],linestyle=':',color='grey')
    # ax.plot3D([0.5,0.25],[0,0.25],[0,0],linestyle=':',color='grey')
    # ax.plot3D([0.5,0.75],[0,0.25],[0,0],linestyle=':',color='grey')
    # ax.plot3D([0.25,0.5],[0.25,0.5],[0.25,0],linestyle=':',color='grey')
    # ax.plot3D([0.75,0.5],[0.25,0.5],[0.25,0],linestyle=':',color='grey')

    ax.plot3D([0.5,0.5],[0,0.5],[0,0],linestyle='-',color='red')        # Special Perfect Entangler line

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



# gs2 = np.load("data/convergence_check_2_NL2_SPE2_gs2.npy", allow_pickle=True)
gs2 = np.load("data/convergence_check_3_NL2_SPE2_gs2.npy", allow_pickle=True)
print(gs2.item()['3NL2'].params[0])

# db_gs2_NL2 = np.load("data/db_gs2_NL2_iter100_time5000.npy", allow_pickle=True)
db_gs2_NL2 = np.load("data/db_gs2_H1T1NL2_iter1000_time500.npy", allow_pickle=True)
vis_Weyl(db_gs2_NL2)