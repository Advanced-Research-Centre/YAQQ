# pip install weylchamber
import numpy as np
import qutip
import matplotlib
import matplotlib.pylab as plt
import weylchamber
from weylchamber.visualize import WeylChamber
from itertools import product
from qiskit.extensions import UnitaryGate

# WeylChamber().plot()

w = WeylChamber()
SWAP = qutip.Qobj([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
CX = qutip.Qobj([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

max_samples = 100
s_yz = int(np.cbrt(2*max_samples))

cx = np.linspace(0, 1, 4*s_yz)
cy = np.linspace(0, 0.5, 2*s_yz)
cz = np.linspace(0, 0.5, 2*s_yz)

gs = product(cx, cy, cz)

valid_points = 0 # 624

# CAN = weylchamber.canonical_gate(0.5, 0.5, 0.5)

for can in gs:
    # Enumerate points in the Weyl chamber
    c = list(can)
    c1,c2,c3 = c[0],c[1],c[2]
    if weylchamber.point_in_weyl_chamber(c1,c2,c3):
        # Add points to the Weyl chamber
        w.add_point(c1,c2,c3)
        print(weylchamber.canonical_gate(c1,c2,c3))
        print(UnitaryGate(weylchamber.canonical_gate(c1,c2,c3),label='RndU'+str(valid_points)))
        exit()
        valid_points+= 1

# for i in range(valid_points):
#     # Haar Random unitary
#     # RndU = qutip.rand_unitary_haar(4)             
#     # c1, c2, c3 = weylchamber.c1c2c3(RndU)
    
#     # Random non-local gate as a point in the Weyl chamber
#     # c1,c2,c3 = weylchamber.random_weyl_point()    
    
#     # Another way of doing the same thing
#     RndU = weylchamber.random_gate()              # Random non-local gate
#     c1, c2, c3 = weylchamber.c1c2c3(RndU)
#     if i == 3:
#         print(RndU)

#     w.add_point(c1,c2,c3)
    

# w.full_cube=True
# w.grid = True
# w.fig_width = 17
# w.fig_height = 12
# w.z_axis_left=False

w.plot()
print(valid_points)


# ax.set_xlim(0, 1)
# ax.set_ylim(0, 0.5)
# ax.set_zlim(0, 0.5)

plt.show()