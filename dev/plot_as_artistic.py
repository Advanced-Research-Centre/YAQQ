import numpy as np
import matplotlib.pyplot as plt

gs1_gates = 'original gate set'
gs2_gates = 'novel gate set'

num_pts = 20

""" Code for fidelity data generation"""

# Set mean for gs1
avg_fid_gs01 = 0.85

# Generate data around the average with some noise
pf1 = np.random.normal(avg_fid_gs01, 0.1, num_pts)

# Clip fidelity to [0,1]
for i in range(len(pf1)):
    if pf1[i] > 1:
        pf1[i] = 1

std_fid_gs01 = np.std(pf1)

# Target trend
ivt_fid_gs01 = np.subtract(1,pf1)

# Reduce deviation
pf2 = ivt_fid_gs01 - np.mean(ivt_fid_gs01) 
for i in range(len(pf2)):
    pf2[i] = pf2[i] * 0.6

# Add some noise
pf2 = pf2 + np.random.normal(0, 0.02, num_pts)

# Move to the right average
pf2 = pf2 + (avg_fid_gs01 + 0.05)

# Clip fidelity to [0,1]
for i in range(len(pf2)):
    if pf2[i] > 1:
        pf2[i] = 1

avg_fid_gs02 = np.mean(pf2)
std_fid_gs02 = np.std(pf2)

""" Code for depth data generation"""

# Set mean for gs1
avg_dep_gs01 = 200

# Generate data around the average with some noise
cd1 = np.random.normal(avg_dep_gs01, 80, num_pts)

std_dep_gs01 = np.std(cd1)

# Target trend
ivt_dep_gs01 = np.subtract(1,cd1)

# Reduce deviation
cd2 = ivt_dep_gs01 - np.mean(ivt_dep_gs01) 
for i in range(len(cd2)):
    cd2[i] = cd2[i] * 0.4

# Add some noise
cd2 = cd2 + np.random.normal(0, 20, num_pts)

# Move to the right average
cd2 = cd2 + (avg_dep_gs01 + 20)

avg_dep_gs02 = np.mean(cd2)
std_dep_gs02 = np.std(cd2)

""" Code for plots"""

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 20}
)

fs = 24
ls = 2
ms = 5

figg, ax = plt.subplots(1, 2)
ax[0].plot(pf1, '-o', markersize=ms, linewidth=ls, color = 'b', label = 'PF ['+gs1_gates+']')
ax[0].plot(pf2, '-x', markersize=ms, linewidth=ls, color = 'r', label = 'PF ['+gs2_gates+']')

ax[0].plot(ivt_fid_gs01, '-s', markersize=ms, linewidth=ls, color = 'g', label = 'target PF trend')

ax[0].axhline(y=avg_fid_gs01, linestyle='-.', linewidth=2, color = 'b' , label = '$$\\langle \\textrm{PF} \\rangle \; [\\textrm{original gate set}]$$')
ax[0].axhline(y=avg_fid_gs02, linestyle='-', linewidth=2, color = 'r' , label = '$$\\langle \\textrm{PF} \\rangle \; [\\textrm{novel gate set}]$$')

ax[0].fill_between(range(0,len(pf1)), np.repeat(avg_fid_gs01+std_fid_gs01,len(pf1)), np.repeat(avg_fid_gs01-std_fid_gs01,len(pf1)), color='b', alpha=0.2)
ax[0].fill_between(range(0,len(pf2)), np.repeat(avg_fid_gs02+std_fid_gs02,len(pf2)), np.repeat(avg_fid_gs02-std_fid_gs02,len(pf2)), color='r', alpha=0.2)


ax[1].plot(cd1, '-o', markersize=ms, linewidth=ls, color = 'b', label = 'CD ['+gs1_gates+']')
ax[1].plot(cd2, '-x', markersize=ms, linewidth=ls, color = 'r', label = 'CD ['+gs2_gates+']')

ax[1].axhline(y=avg_dep_gs01, linestyle='-.', linewidth=2, color = 'b', label = '$$\\langle \\textrm{CD} \\rangle \; [\\textrm{original gate set}]$$')
ax[1].axhline(y=avg_dep_gs02, linestyle='-', linewidth=2, color = 'r', label = '$$\\langle \\textrm{CD} \\rangle \; [\\textrm{novel gate set}]$$')

ax[1].fill_between(range(0,len(cd1)), np.repeat(avg_dep_gs01+std_dep_gs01,len(cd1)), np.repeat(avg_dep_gs01-std_dep_gs01,len(cd1)), color='b', alpha=0.2)
ax[1].fill_between(range(0,len(cd2)), np.repeat(avg_dep_gs02+std_dep_gs02,len(cd2)), np.repeat(avg_dep_gs02-std_dep_gs02,len(cd2)), color='r', alpha=0.2)

ax[0].set_ylabel("Process fidelity (PF)", fontsize=fs, fontfamily='serif')
ax[1].set_ylabel("Circuit depth (CD)", fontsize=fs, fontfamily='serif')

ax[0].set_xlabel("Data set of target unitary matrices", fontsize=fs, fontfamily='serif')
ax[1].set_xlabel("Data set of target unitary matrices", fontsize=fs, fontfamily='serif')

ax[0].set_xlim([-1, len(pf1)])
ax[1].set_xlim([-1, len(cd1)])

ax[0].set_ylim(bottom=-0.1,top=1.1)    # not set to show std. div.
# ax[1].set_ylim(bottom=0,top=None) # not set to show std. div.
ax[0].legend(loc='center right')
ax[1].legend(loc='lower right')

# figg.set_size_inches(18,8)
figg.set_size_inches(26,8)
figg.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.098, wspace=0.183, hspace=0.2)

plt.savefig('results/figures_for_paper/artistic.pdf')
# plt.show()