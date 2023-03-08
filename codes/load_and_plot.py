import matplotlib.pyplot as plt
import numpy as np


gateset_load = np.load('data/gateset_list.npy')
depth_load = np.load('data/total_depth_list.npy')
fidelity_load = np.load('data/total_fid_list.npy')


# print(gateset_load)
# print('-----------')
# print(depth_load)
# print('-----------')
# print(fidelity_load)
points = len(fidelity_load[0])
trials = len(fidelity_load)
x1_list, x2_list = [], []
for i in fidelity_load:
    x1, x2 = 0, 0
    for fid in i:
        x1 += fid[0]
        x2 += fid[1]
    x1_list.append(x1 / points)
    x2_list.append(x2 / points)

plt.plot(x1_list, '-x' , label = '[H, T, Tdag]')
plt.plot(x2_list, '-o', label = 'Random Unitary')
plt.legend()
plt.show()

# print(x1)

# indx = 


exit()

y1,y2 = [],[]
for i in result_db:
    y1.append(i[2])
    y2.append(i[3])

plt.plot(y1, '-x', label = "[t, h, tdg]")
plt.plot(y2, '-o', label = "[b, h, tdg]")
plt.ylim((0,1))
plt.legend()
plt.show()