import weylchamber
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

import math

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

px = 23
cx = np.linspace(0, 1, px)
cy = np.linspace(0, 1, px)
cz = np.linspace(0, 1, px)
gs = product(cx, cy, cz)

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

for i in gs:
    g = list(i)    
    if weylchamber.point_in_weyl_chamber(g[0], g[1], g[2]):
        ax.scatter(g[0], g[1], g[2], s=2,c=rgb_to_hex(255-int(255*(math.cos(g[0]*2*math.pi)+1)/2),255-int(255*(math.cos(g[1]*2*math.pi)+1)/2),255-int(255*(math.cos(g[2]*2*math.pi)+1)/2)))

# ax.scatter(g[0], g[1], g[2], c=rgb_to_hex(int(g[0]*255),int(g[1]*255),int(g[2]*255)))
# ax.scatter(g[0], g[1], g[2], s=5,c=rgb_to_hex(int(255*(math.sin(g[0]*math.pi)+1)/2),int(255*(math.sin(g[1]*math.pi)+1)/2),int(255*(math.sin(g[2]*math.pi)+1)/2)))
    
ax.plot3D([0,0.5],[0,0.5],[0,0.5],linestyle='--',color='black')       # Isotropic exchange Swap-alpha gates
ax.plot3D([1,0.5],[0,0.5],[0,0.5],linestyle='--',color='black')       # Isotropic exchange Swap-alpha gates 
ax.plot3D([0.5,0.5],[0.5,0.5],[0,0.5],linestyle='--',color='black')   # Parameterized Swap gates
ax.plot3D([0,1],[0,0],[0,0],linestyle='--',color='black')             # Ising gates
ax.plot3D([0,0.5],[0,0.5],[0,0],linestyle='--',color='black')         # XY gates
ax.plot3D([1,0.5],[0,0.5],[0,0],linestyle='--',color='black')         # XY gates

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

ax.grid(False)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# ax.title.set_text('Weyl Chamber')

plt.show()