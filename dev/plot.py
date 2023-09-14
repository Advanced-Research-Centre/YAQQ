import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(1, 2, figsize = (7,4))

rand_search_pf1 = np.load('results/data/data_typ1_gateset_p1h1t1_iter100_time100_randsearch_pf1.npy', allow_pickle=True)
rand_search_pf2 = np.load('results/data/data_typ1_gateset_p1h1t1_iter100_time100_randsearch_pf2.npy', allow_pickle=True)
rand_search_cd1 = np.load('results/data/data_typ1_gateset_p1h1t1_iter100_time100_randsearch_cd1.npy', allow_pickle=True)
rand_search_cd2 = np.load('results/data/data_typ1_gateset_p1h1t1_iter100_time100_randsearch_cd2.npy', allow_pickle=True)

ax[0].plot(rand_search_pf1, 'r-x', label = '[H,T]')
ax[0].plot(rand_search_pf2, 'b-o', label = '$[P(\\theta),H,T]$')
# ax[1].semilogy(rand_search_cd1, 'r-x')
# ax[1].plot(rand_search_cd2, 'b-o')

ax[1].axhline(y=np.mean(rand_search_cd1), linestyle='-.', color = 'r', label = '[H,T]')
ax[1].axhline(y=np.mean(rand_search_cd2), linestyle='-.', color = 'b', label = '$[P(\\theta),H,T]$')

ax[0].legend()

plt.show()