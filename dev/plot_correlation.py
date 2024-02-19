import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 12}
)
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(4,4), dpi = 100)

data = 'NUSA_eid-0013_2024-02-10-10-55'
y1, y2, y3 = [], [], []
for smaller_pojnts in range(2,50+1,1):
    # pf_ht = np.load(f'results/data/{data}pf1.npy')[smaller_pojnts:500]
    # pf_ht = np.load(f'results/data/{data}pf1.npy')[smaller_pojnts:500]
    pf_ht = np.load(f'results/data/{data}pf1.npy')[:smaller_pojnts]
    pf_p1p1 = np.load(f'results/data/{data}pf2.npy')[:smaller_pojnts]
    ## ------------------- depth data load --------------------
    # cd_ht = np.load(f'results/data/{data}cd1.npy')[:smaller_pojnts]
    # cd_p1p1 = np.load(f'results/data/{data}cd2.npy')[:smaller_pojnts]
    y3.append(np.mean(pf_ht))
    y2.append(pearsonr(pf_ht, pf_p1p1)[0])
y1 = np.load(f'results/data/{data}pf1.npy')
ax2.set_xlabel('Data set of target unitary', fontsize = 9)
ax1.set_xticks([])
ax2.set_ylabel('Pearson correlation', fontsize = 9)
ax1.set_ylabel('Process fidelity (PF)', fontsize = 9)
ax1.plot(y1, 'ob', markersize = 3, alpha = 0.5, label = 'PF')
ax1.plot(y3, '-r', label = '$$\\textrm{Cumulative}\;\\langle PF\\rangle$$')
ax2.plot(y2)
ax1.legend(fontsize = 8)
fig.tight_layout()
# plt.savefig('correlation_plot.pdf')
plt.show()