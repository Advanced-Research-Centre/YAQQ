import numpy as np
import matplotlib.pyplot as plt

point = 5
max_dep = 100
trials = 100

fid_gs01 = np.random.rand(point,1)
dep_gs01 = (max_dep*np.random.rand(point,1)).astype(int)
ivt_fid_gs01 = 1-fid_gs01
avg_fid_gs01 = np.mean(ivt_fid_gs01)
avg_dep_gs01 = np.mean(dep_gs01)

cfn_best = 100000
fid_best = []
dep_best = []

for _ in range(trials):

    fid_gs02 = np.random.rand(point,1)
    dep_gs02 = (max_dep*np.random.rand(point,1)).astype(int)

    avg_fid_gs02 = np.mean(fid_gs02)
    avg_dep_gs02 = np.mean(dep_gs02)

    dist_fid = sum(abs(np.subtract(ivt_fid_gs01,fid_gs02)))
    dist_fid_avg = avg_fid_gs01 - avg_fid_gs02
    dist_dep_avg = avg_dep_gs02 - avg_dep_gs01
    
    cfn = dist_fid + dist_fid_avg + dist_dep_avg

    if cfn <= cfn_best:
        cfn_best = cfn
        fid_best = fid_gs02
        dep_best = dep_gs02

avg_fid_best = np.mean(fid_best)
avg_dep_best = np.mean(dep_best)

_, ax = plt.subplots(1, 2)

ax[0].plot(fid_gs01, '-x', label = "gs01", color = 'r' )
ax[0].plot(ivt_fid_gs01, '-x', label = "gs01_ivt", color = 'g')
ax[0].plot(fid_best, '-o', label = "gs02", color = 'b')

ax[0].axhline(y=avg_fid_gs01, linestyle='-.', color = 'g')
ax[0].axhline(y=avg_fid_best, linestyle='-.', color = 'b')

ax[1].plot(dep_gs01, '-x', label = "gs01", color = 'r')
ax[1].plot(dep_best, '-o', label = "gs02", color = 'b')

ax[1].axhline(y=avg_dep_gs01, linestyle='-.', color = 'r')
ax[1].axhline(y=avg_dep_best, linestyle='-.', color = 'b')

ax[0].set_ylabel("Process Fidelity")
ax[1].set_ylabel("Decomposed Circuit Depth")
ax[0].set_ylim(bottom=0,top=1)
ax[1].set_ylim(bottom=0,top=None)
ax[0].legend()
ax[1].legend()
plt.show()