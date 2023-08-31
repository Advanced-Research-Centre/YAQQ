import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

x = []
for _ in range(30000):
    statistics_generated = np.clip(np.random.negative_binomial(n=70,p=0.573, size=100),25,70)
    # statistics_generated = np.clip(np.random.negative_binomial(n=40,p=0.6, size=100),0,0)
    c = Counter(statistics_generated)
    halting_step = c.most_common(1)[0][0]
    x.append(halting_step)

# plt.hist(x)
plt.hist(x, bins=30, density=True, alpha=0.6, color='g', label='Tailed Distribution')
plt.show()
    # if statistics_generated <= 30:
    #     print(statistics_generated)

exit()

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