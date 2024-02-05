import numpy as np
import matplotlib.pyplot as plt

# expname = 'results/data/NUSA_eid-0003_2024-01-05-22-22'     # 2 qubit
# gs2_gates = 'P1,P1,NL2'
# expname = 'results/data/NUSA_eid-0004_2024-01-05-20-40'     # 2 qubit 
# gs2_gates = 'R1,P1,SPE2'
expname = 'results/data/NUSA_eid-0005_2024-01-05-18-35'     # 3 qubit       (for paper)
gs2_gates = 'P1,P1,SPE2'
# expname = 'results/data/NUSA_eid-0006_2024-01-08-11-12'   # 1 qubit
# expname = 'results/data/NUSA_eid-0007_2024-01-09-11-32'   # 3 qubit
# gs2_gates = 'R1,P1,SPE2'

pf1 = np.load(expname+'pf1.npy', allow_pickle=True)
pf2 = np.load(expname+'pf2.npy', allow_pickle=True)
cd1 = np.load(expname+'cd1.npy', allow_pickle=True)
cd2 = np.load(expname+'cd2.npy', allow_pickle=True)
# gs1 = np.load(expname+'gs1.npy', allow_pickle=True)
# gs2 = np.load(expname+'gs2.npy', allow_pickle=True)

gs1_gates = 'H1,T1,CX2'

avg_fid_gs01 = np.mean(pf1)
avg_fid_gs02 = np.mean(pf2)
avg_dep_gs01 = np.mean(cd1)
avg_dep_gs02 = np.mean(cd2) 

ivt_fid_gs01 = np.subtract(1,pf1)

_, ax = plt.subplots(1, 2)
ax[0].plot(pf1, '-x', color = 'r', label = 'PF ['+gs1_gates+']')
ax[0].plot(pf2, '-o', color = 'b', label = 'PF ['+gs2_gates+']')

# ax[0].plot(ivt_fid_gs01, ':', color = 'g', label = 'target PF trend')

ax[0].axhline(y=avg_fid_gs01, linestyle='-.', color = 'r' , label = 'avg.PF ['+gs1_gates+']')
ax[0].axhline(y=avg_fid_gs02, linestyle='-.', color = 'b' , label = 'avg.PF ['+gs2_gates+']')

ax[1].plot(cd1, '-x', color = 'r', label = 'CD ['+gs1_gates+']')
ax[1].plot(cd2, '-o', color = 'b', label = 'CD ['+gs2_gates+']')

ax[1].axhline(y=avg_dep_gs01, linestyle='-.', color = 'r', label = 'avg.CD ['+gs1_gates+']')
ax[1].axhline(y=avg_dep_gs02, linestyle='-.', color = 'b', label = 'avg.CD ['+gs2_gates+']')

ax[0].set_ylabel("Process Fidelity")
ax[1].set_ylabel("Circuit Depth")
# ax[0].set_ylim(bottom=0,top=1)
# ax[1].set_ylim(bottom=0,top=None)
ax[0].legend()
ax[1].legend()

plt.show()