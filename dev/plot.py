import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
search = ['randsearch', 'scipy']
gateset = ['p1h1t1', 'p1p1']
# fig, ax = plt.subplots(1, 2, figsize = (7,4))
# gateset = gateset[0]
search, gateset = search[1], gateset[1]

# rand_search_pf1 = np.load(f'results/data/data_typ1_gateset_p1h1t1_iter100_time100_{search}_100_data_pf1.npy', allow_pickle=True)
# rand_search_pf2 = np.load(f'results/data/data_typ1_gateset_p1h1t1_iter100_time100_{search}_100_data_pf2.npy', allow_pickle=True)
# rand_search_cd1 = np.load(f'results/data/data_typ1_gateset_p1h1t1_iter100_time100_{search}_100_data_cd1.npy', allow_pickle=True)
# rand_search_cd2 = np.load(f'results/data/data_typ1_gateset_p1h1t1_iter100_time100_{search}_100_data_cd2.npy', allow_pickle=True)

x = []
for w1 in [50]:#range(0,200+1,50):
    print('weight', [w1,50,1,1,0])
    if w1 == 0:
        rand_search_pf1 = np.load(f'results/data/data_typ1_gateset_{gateset}_iter100_time100_{search}_pf1.npy', allow_pickle=True)
        rand_search_pf2 = np.load(f'results/data/data_typ1_gateset_{gateset}_iter100_time100_{search}_pf2.npy', allow_pickle=True)
        rand_search_cd1 = np.load(f'results/data/data_typ1_gateset_{gateset}_iter100_time100_{search}_cd1.npy', allow_pickle=True)
        rand_search_cd2 = np.load(f'results/data/data_typ1_gateset_{gateset}_iter100_time100_{search}_cd2.npy', allow_pickle=True)
    else:
        rand_search_pf1 = np.load(f'results/data/data_typ1_gateset_{gateset}_iter100_time100_{search}_{w1}_1_1_1_0_pf1.npy', allow_pickle=True)
        rand_search_pf2 = np.load(f'results/data/data_typ1_gateset_{gateset}_iter100_time100_{search}_{w1}_1_1_1_0_pf2.npy', allow_pickle=True)
        rand_search_cd1 = np.load(f'results/data/data_typ1_gateset_{gateset}_iter100_time100_{search}_{w1}_1_1_1_0_cd1.npy', allow_pickle=True)
        rand_search_cd2 = np.load(f'results/data/data_typ1_gateset_{gateset}_iter100_time100_{search}_{w1}_1_1_1_0_cd2.npy', allow_pickle=True)

# ax[0].plot(rand_search_pf1, 'r-', label = '[H,T]')
# # ax[0].plot(rand_search_pf2, 'b--', label = '$[P(\\theta),H,T]$')
# ax[0].axhline(y=np.mean(rand_search_pf1), linestyle='-.', color = 'r', label = '[H,T]')
# ax[0].axhline(y=np.mean(rand_search_pf2), linestyle='-.', color = 'b', label = '$[P(\\theta),H,T]$')
# # ax[1].semilogy(rand_search_cd1, 'r-x')
# # ax[1].plot(rand_search_cd2, 'b-o')

# ax[1].axhline(y=np.mean(rand_search_cd1), linestyle='-.', color = 'r', label = '[H,T]')
# ax[1].axhline(y=np.mean(rand_search_cd2), linestyle='-.', color = 'b', label = '$[P(\\theta),H,T]$')


    print(f'gateset [H,T] average process fidelity ({search}):', np.mean(rand_search_pf1))
    print(f'gateset {gateset} average process fidelity ({search}):', np.mean(rand_search_pf2))
    print(f'gateset [H,T] average process fidelity ({search}):', np.mean(rand_search_cd1))
    print(f'gateset {gateset} average process fidelity ({search}):', np.mean(rand_search_cd2))
    x.append(np.mean(rand_search_pf2) - np.mean(rand_search_pf1))
    print()
# plt.plot([0,50,100,150,200], x, '-o', label = '(pf2-pf1)')
# # plt.legend()
# plt.xticks([0,50,100,150,200])
# plt.xlabel('$w_{apf}$')
# plt.ylabel('$\\mathcal{P}_{f_n} - \\mathcal{P}_{f}$')
# plt.title('$G(\\theta),H,T$')
# # ax[0].legend()
# # ax[0].set_ylim([0.8,1])

# plt.show()