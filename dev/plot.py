import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {"font.family": "serif", "font.size": 18}
)

# # expname = 'results/data/NUSA_eid-0008_2024-02-06-12-50'     # 1 qubit 200 pts (for paper)
# expname = 'results/data/NUSA_eid-0009_2024-02-06-13-14'     # 1 qubit 500 pts (for paper)
# ds = np.load(expname+'ds.npy', allow_pickle=True)
# from yaqq_ds import VisualizeDataSet
# vds = VisualizeDataSet()
# vds.vis_ds_Bloch(ds)


# expname = 'results/data/NUSA_eid-0008_2024-02-06-13-00'     # 1 qubit 200 pts (for paper)
# expname = 'results/data/NUSA_eid-0009_2024-02-06-13-40'     # 1 qubit 500 pts (for paper)
# expname = 'results/data/NUSA_eid-0013_2024-02-10-10-55' # 1 qubit 50 pts (for paper) novel
# gs2_gates = 'P1,P1'
expname = 'results/data/NUSA_eid-0017_2024-02-12-16-21'     # 2 qubit 10 pts (for paper) F1,F1
gs2_gates = 'P1$^{opt.200}_{\\vec{a}}$,P1$^{opt.200}_{\\vec{b}}$,NL2'
# expname = 'results/data/NUSA_eid-0003_2024-01-05-22-22'     # 2 qubit 10 pts (bad results)
# gs2_gates = 'P1,P1,NL2'
# expname = 'results/data/NUSA_eid-0004_2024-01-05-20-40'     # 2 qubit 10 pts (bad results)
# gs2_gates = 'R1,P1,SPE2'
# expname = 'results/data/NUSA_eid-0005_2024-01-05-18-35'     # 3 qubit 10 pts (bad results)
# gs2_gates = 'P1,P1,SPE2'
# expname = 'results/data/NUSA_eid-0007_2024-01-09-11-32'     # 3 qubit 10 pts
# gs2_gates = 'R1,P1,SPE2'

# expname = 'results/data/NUSA_eid-0007_2024-01-09-11-32'     # 3 qubit 10 pts
# gs2_gates = 'R1,P1,SPE2'

pf1 = np.load(expname+'pf1.npy', allow_pickle=True)
pf2 = np.load(expname+'pf2.npy', allow_pickle=True)
cd1 = np.load(expname+'cd1.npy', allow_pickle=True)
cd2 = np.load(expname+'cd2.npy', allow_pickle=True)
# gs1 = np.load(expname+'gs1.npy', allow_pickle=True)
gs2 = np.load(expname+'gs2.npy', allow_pickle=True)
print(gs2)

gs1_gates = 'H1,T1'
# gs1_gates = 'H1,T1,CX2'

# """HQEC Experiment"""

# gs1_gates = 'T1,S1,Z1,X1'
# gs2_gates = 'H1,S1,Z1,X1'
# expnameA = 'results/data/HQEC_eid-0002_2024-02-07-16-04'     # QEC 16 pts Rnd
# expnameB = 'results/data/HQEC_eid-0003_2024-02-07-16-14'     # QEC 16 pts Skt
# pf1 = np.load(expnameB+'pf1.npy', allow_pickle=True)
# pf2 = np.load(expnameA+'pf2.npy', allow_pickle=True)
# cd1 = np.load(expnameB+'cd1.npy', allow_pickle=True)
# cd2 = np.load(expnameA+'cd2.npy', allow_pickle=True)

avg_fid_gs01 = np.mean(pf1)
avg_fid_gs02 = np.mean(pf2)
avg_dep_gs01 = np.mean(cd1)
avg_dep_gs02 = np.mean(cd2) 
std_fid_gs01 = np.std(pf1)
std_fid_gs02 = np.std(pf2)
std_dep_gs01 = np.std(cd1)
std_dep_gs02 = np.std(cd2) 

ivt_fid_gs01 = np.subtract(1,pf1)

fs = 24
ls = 1
ms = 5

figg, ax = plt.subplots(1, 2)
ax[0].plot(pf1, '-o', markersize=ms, linewidth=ls, color = 'b', label = 'PF ['+gs1_gates+']')
ax[0].plot(pf2, '-x', markersize=ms, linewidth=ls, color = 'r', label = 'PF ['+gs2_gates+']')

# ax[0].plot(ivt_fid_gs01, '-s', markersize=ms, linewidth=ls, color = 'g', label = 'target PF trend')

ax[0].axhline(y=avg_fid_gs01, linestyle='-.', linewidth=2, color = 'b' , label = 'avg.PF ['+gs1_gates+']')
ax[0].axhline(y=avg_fid_gs02, linestyle='-', linewidth=2, color = 'r' , label = 'avg.PF ['+gs2_gates+']')

# ax[0].fill_between(range(0,len(pf1)), np.repeat(avg_fid_gs01+std_fid_gs01,len(pf1)), np.repeat(avg_fid_gs01-std_fid_gs01,len(pf1)), color='b', alpha=0.2)
# ax[0].fill_between(range(0,len(pf2)), np.repeat(avg_fid_gs02+std_fid_gs02,len(pf2)), np.repeat(avg_fid_gs02-std_fid_gs02,len(pf2)), color='r', alpha=0.2)


ax[1].plot(cd1, '-o', markersize=ms, linewidth=ls, color = 'b', label = 'CD ['+gs1_gates+']')
ax[1].plot(cd2, '-x', markersize=ms, linewidth=ls, color = 'r', label = 'CD ['+gs2_gates+']')

ax[1].axhline(y=avg_dep_gs01, linestyle='-.', linewidth=2, color = 'b', label = 'avg.CD ['+gs1_gates+']')
ax[1].axhline(y=avg_dep_gs02, linestyle='-', linewidth=2, color = 'r', label = 'avg.CD ['+gs2_gates+']')

# ax[1].fill_between(range(0,len(cd1)), np.repeat(avg_dep_gs01+std_dep_gs01,len(cd1)), np.repeat(avg_dep_gs01-std_dep_gs01,len(cd1)), color='b', alpha=0.2)
# ax[1].fill_between(range(0,len(cd2)), np.repeat(avg_dep_gs02+std_dep_gs02,len(cd2)), np.repeat(avg_dep_gs02-std_dep_gs02,len(cd2)), color='r', alpha=0.2)

ax[0].set_ylabel("Process Fidelity (PF)", fontsize=fs, fontfamily='serif')
ax[1].set_ylabel("Circuit Depth (CD)", fontsize=fs, fontfamily='serif')

ax[0].set_xlabel("Data set of target unitary matrices", fontsize=fs, fontfamily='serif')
ax[1].set_xlabel("Data set of target unitary matrices", fontsize=fs, fontfamily='serif')

ax[0].set_xlim([0, len(pf1)-1])
ax[1].set_xlim([0, len(cd1)-1])

ax[0].set_ylim(bottom=-0.1,top=1.1)    # not set to show std. div.
# ax[1].set_ylim(bottom=0,top=None) # not set to show std. div.
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')

# figg.set_size_inches(18,8)
figg.set_size_inches(26,8)
figg.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.098, wspace=0.183, hspace=0.2)

plt.show()