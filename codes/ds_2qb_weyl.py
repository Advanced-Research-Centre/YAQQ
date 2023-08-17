# pip install weylchamber
import numpy as np
import qutip
import matplotlib
import matplotlib.pylab as plt
import weylchamber
from weylchamber.visualize import WeylChamber
from itertools import product

# WeylChamber().plot()

w = WeylChamber()
SWAP = qutip.Qobj([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
CX = qutip.Qobj([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

cx = np.linspace(0, 1, 8)
cy = np.linspace(0, 1, 8)
cz = np.linspace(0, 1, 8)

gs = product(cx, cy, cz)

# CAN = weylchamber.canonical_gate(0.5, 0.5, 0.5)
for can in gs:
    # Option 1
    # RndU = qutip.rand_unitary_haar(4)             # Haar Random unitary
    # c1, c2, c3 = weylchamber.c1c2c3(RndU)
    # Option 2
    # c1,c2,c3 = weylchamber.random_weyl_point()    # Random non-local gate as a point in the Weyl chamber
    # Option 3
    # RndU = weylchamber.random_gate()              # Random non-local gate
    # c1, c2, c3 = weylchamber.c1c2c3(RndU)
    # Option 4                                      # Enumerate points in the Weyl chamber
    c = list(can)
    c1,c2,c3 = c[0],c[1],c[2]
    # Add points to the Weyl chamber
    w.add_point(c1,c2,c3)
    
# w.full_cube=True
# w.grid = True
# w.fig_width = 17
# w.fig_height = 12
# w.z_axis_left=False
w.plot()

plt.show()