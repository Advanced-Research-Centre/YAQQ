import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

fig, (ax1,ax2) = plt.subplots(1,2, sharey=False, constrained_layout = True)

# for t in range(1, 18+1):
#     x1 = np.load(f'results/data/data_2q_gateset_p1p1nl2_dec_rand_rand_rand_iter500_time5000_scipy_gs2_per_time_trial_{t}.npy', allow_pickle=True)
#     # x2 = np.load('results/data/data_nl2_converge_spe2_pf1.npy')
#     # x2 = np.load('results/data/data_nl2_converge_spe2_pf2.npy')
#     print(len(x1))

# x1 = np.load(f'results/data/data_2q_gateset_p1p1nl2_dec_rand_rand_rand_iter500_time5000_scipy_gs2_per_time_trial_1.npy', allow_pickle=True)
# x2 = np.load(f'results/data/data_2q_gateset_p1p1nl2_dec_rand_rand_rand_iter500_time5000_scipy_gs2_per_time_trial_18.npy', allow_pickle=True)

x1 = np.load(f'results/data/SPE2_NL2_convergence_iter300_time5000_pf1.npy')
x2 = np.load(f'results/data/SPE2_NL2_convergence_iter300_time5000_pf2.npy')

xd1 = np.load(f'results/data/SPE2_NL2_convergence_iter300_time5000_cd1.npy')
xd2 = np.load(f'results/data/SPE2_NL2_convergence_iter300_time5000_cd2.npy')
xticks = [1,2,3,4,5]

ax1.plot(xticks, x1, 'or', label = '[H,T,CX]')
ax1.plot(xticks, x2, 'xb', label = '[P,P,NL]')

ax2.plot(xticks, xd1, 'or')
ax2.plot(xticks, xd2, 'xb')

ax1.plot(xticks, [np.mean(x1)]*5, 'r--', label = 'Average [H,T,CX]' )
ax1.plot(xticks, [np.mean(x2)]*5, 'b--', label = 'Average [P,P,NL]')

ax2.plot(xticks, [np.mean(xd1)]*5, 'r--' )
ax2.plot(xticks, [np.mean(xd2)]*5, 'b--' )

ax1.set_xticks([1,2,3,4,5])
print()
print('00000000')
print()
ax1.legend()
ax1.set_ylim(0,1)
ax1.set_xlabel('Points')
ax2.set_xlabel('Points')
ax1.set_ylabel('Fidelity')
ax2.set_ylabel('Depth')

plt.show()
# print(x2[:5])

