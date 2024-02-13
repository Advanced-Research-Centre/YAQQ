import numpy as np
import matplotlib.pyplot as plt

expname = 'results/data/NUSA_eid-0008_2024-02-06-13-00gs1'
# expname = 'results/data/NUSA_eid-0008_2024-02-06-13-00gs2'

gs2 = (np.load(expname+'.npy', allow_pickle=True)).tolist()

for gid in gs2.keys():
    np.save(expname+'_'+gid, gs2[gid])

# fname = expname+'_1P1.npy'
# g = np.load(fname, allow_pickle=True)
# print(g)

# results/data/NUSA_eid-0008_2024-02-06-13-00gs1_1H1.npy
# results/data/NUSA_eid-0008_2024-02-06-13-00gs1_2T1.npy
# results/data/NUSA_eid-0008_2024-02-06-13-00gs2_1P1.npy
# results/data/NUSA_eid-0008_2024-02-06-13-00gs2_2P1.npy