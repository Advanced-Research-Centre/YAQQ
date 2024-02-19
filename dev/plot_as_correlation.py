import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# import re
# import os

# dir_res = 'results/data/'

# fnames = []
# for (dirpath, dirnames, filenames) in os.walk(dir_res):
#     fnames.extend(filenames)

# y_pf_corr = []
# for file in fnames:
#     match = re.search("NUSA_eid-001([1-5])_.*pf1", file) 
#     if match:
#         pf_ht = np.load(dir_res+file)
#         pf_p1p1 = np.load(dir_res+file[:-5]+'2.npy')
#         y_pf_corr.append([int(match.group(1)),pearsonr(pf_ht, pf_p1p1)[0]])
#         print(file,pearsonr(pf_ht, pf_p1p1)[0])

# NUSA_eid-0013_2024-02-10-10-55pf1.npy -0.42662448519333446

fs = 16

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 14}
)

fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(6,6), dpi = 200)

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
ax2.set_xlabel('Data set of target unitary matrices', fontsize = fs)
ax1.set_xticks([])
ax2.set_ylabel('Pearson correlation', fontsize = fs)
ax1.set_ylabel('Process fidelity (PF)', fontsize = fs)
ax1.plot(y1, 'ob', markersize = 3, alpha = 0.5, label = 'PF')
ax1.plot(y3, '-r', label = '$$\\textrm{Cumulative}\;\\langle \\textrm{PF}\\rangle$$')
ax2.plot(y2, '-b')
ax2.set_ylim(-1.0,0)
ax1.set_ylim(0,1)
ax1.set_xlim(0,50)
ax2.set_xlim(0,50)
ax1.legend(fontsize = fs)
fig.tight_layout()
plt.savefig('results/figures_for_paper/correlation_plot.pdf')
plt.show()