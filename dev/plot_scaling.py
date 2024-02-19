import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 12}
)

exp_id = '0018_2024-02-12-16-21'
fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi = 100)
points = list(range(1, 20+1,1))

## ------------------- process fidelity data load --------------------
pf_ht = np.load(f'results/data/NUSA_eid-{exp_id}pf1.npy')
pf_p1p1 = np.load(f'results/data/NUSA_eid-{exp_id}pf2.npy')
pf_target = [1-x for x in pf_ht]
    ## ------------------- depth data load --------------------
cd_ht = np.load(f'results/data/NUSA_eid-{exp_id}cd1.npy')
cd_p1p1 = np.load(f'results/data/NUSA_eid-{exp_id}cd2.npy')

    ## correlation calculator
    # print(pearsonr(pf_ht, pf_p1p1)

ax[0].plot(points, pf_ht, 'b-o', markersize = 3, \
           label = '$$\\textrm{PF}\;[\\textrm{H}1,\;\\textrm{T}1]$$')
ax[0].plot(points,pf_p1p1, 'r-x', markersize = 3, \
           label = '$$\\textrm{PF}\;[\\textrm{P}1_{\\vec{a}}^{opt. 200},\;\\textrm{P}1_{\\vec{b}}^{opt. 200}]$$')
# ax[0].plot(pf_target, 'g-s', markersize = 3, \
            #    label = '$$\\textrm{Target PF}\;[\\textrm{H}1,\;\\textrm{T}1]\;\\textrm{trend}$$')
ax[0].set_ylim([0.7,1])
ax[0].axhline(np.mean(pf_ht), color = 'b', linestyle = '-.', \
              label = '$$\\langle \\textrm{PF}\\rangle\;[\\textrm{H}1,\;\\textrm{T}1]$$')
ax[0].axhline(np.mean(pf_p1p1), color = 'r', linestyle = '-', \
              label = '$$\\langle \\textrm{PF}\\rangle\;[\\textrm{P}1_{\\vec{a}}^{opt. 200},\;\\textrm{P}1_{\\vec{b}}^{opt. 200}]$$')
ax[0].fill_between(points, np.mean(pf_ht)-np.std(pf_ht), np.mean(pf_ht)+np.std(pf_ht), color = 'b', alpha=0.25)
ax[0].fill_between(points, np.mean(pf_p1p1)-np.std(pf_p1p1), np.mean(pf_p1p1)+np.std(pf_p1p1), color = 'r', alpha=0.25)
ax[0].set_xlabel('Data set of target unitary')
ax[0].set_ylabel('Process fidelity (PF)')
ax[0].legend(ncol=1, fontsize = 9, bbox_to_anchor = (0.55,0.32))

ax[1].plot(cd_ht, 'b-o', markersize = 3, label = '$$\\textrm{CD}\;[\\textrm{H}1,\;\\textrm{T}1]$$')
ax[1].plot(cd_p1p1, 'r-x', markersize = 3, label = '$$\\textrm{CD}\;[\\textrm{P}1_{\\vec{a}}^{opt. 200},\;\\textrm{P}1_{\\vec{b}}^{opt. 200}]$$')
ax[1].axhline(np.mean(cd_ht), color = 'b', linestyle = '-.', label = '$$\\langle \\textrm{CD}\\rangle\;[\\textrm{H}1,\;\\textrm{T}1]$$')
ax[1].axhline(np.mean(cd_p1p1), color = 'r', linestyle = '-', label = '$$\\langle \\textrm{CD}\\rangle\;[\\textrm{P}1_{\\vec{a}}^{opt. 200},\;\\textrm{P}1_{\\vec{b}}^{opt. 200}]$$')
ax[1].fill_between(points, np.mean(cd_ht)-np.std(cd_ht), np.mean(cd_ht)+np.std(cd_ht), color = 'b', alpha=0.25)
ax[1].fill_between(points, np.mean(cd_p1p1)-np.std(cd_p1p1), np.mean(cd_p1p1)+np.std(cd_p1p1), color = 'r', alpha=0.25)
ax[1].set_xlabel('Data set of target unitary')
ax[1].set_ylabel('Circuit depth (CD)')
ax[1].legend(ncol=1, fontsize = 9, bbox_to_anchor = (0.84,0.32))
fig.tight_layout()

# plt.savefig(f'2q_scaling.pdf')
# plt.savefig(f'2q_scaling.png')

plt.show()