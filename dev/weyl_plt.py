
# import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D  # required to activate 2D plotting
# from mpl_toolkits.mplot3d import proj3d
# from matplotlib.patches import FancyArrowPatch

# cx = 0.5
# cy = 0.5
# cz = 0.5

# plt.plot(cx,cy,cz,'o')
# plt.show()


from itertools import product
import matplotlib.pyplot as plt
import numpy as np

import math

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

px = 6
cx = np.linspace(0, 1, px)
cy = np.linspace(0, 1, px)
cz = np.linspace(0, 1, px)
gs = product(cx, cy, cz)

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

for i in gs:
    g = list(i)
    # ax.scatter(g[0], g[1], g[2], c=rgb_to_hex(int(g[0]*255),int(g[1]*255),int(g[2]*255)))
    ax.scatter(g[0], g[1], g[2], c=rgb_to_hex(int(255*(math.sin(g[0]*math.pi)+1)/2),int(255*(math.sin(g[1]*math.pi)+1)/2),int(255*(math.sin(g[2]*math.pi)+1)/2)))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()