import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# plt.rcParams.update(
#     {"text.usetex": True, "font.family": "serif", "font.size": 14}
# )

plt.rcParams.update(
    {"font.family": "serif", "font.size": 18}
)


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
plt.xlabel('Data set of target unitary')
plt.ylabel('Pearson correlation')
plt.plot(y1, 'ob', markersize = 3, alpha = 0.5)
plt.plot(y3, '-r')
plt.plot(y2)
plt.show()
exit()

how_many_points = 200
print(how_many_points)

if how_many_points == 500:
    exp_id = '0009_2024-02-06-13-40'
elif how_many_points == 200:
    exp_id = '0008_2024-02-06-13-00'

from matplotlib.pyplot import figure
fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi = 100)
points = range(1, how_many_points+1)

## ------------------- process fidelity data load --------------------
x= []
for smaller_pojnts in range(2,200+1):
    pf_ht = np.load(f'results/data/NUSA_eid-{exp_id}pf1.npy')[:smaller_pojnts]
    pf_p1p1 = np.load(f'results/data/NUSA_eid-{exp_id}pf2.npy')[:smaller_pojnts]
    pf_target = [1-x for x in pf_ht]
    ## ------------------- depth data load --------------------
    cd_ht = np.load(f'results/data/NUSA_eid-{exp_id}cd1.npy')[:smaller_pojnts]
    cd_p1p1 = np.load(f'results/data/NUSA_eid-{exp_id}cd2.npy')[:smaller_pojnts]

    ## correlation calculator
    # print(pearsonr(pf_ht, pf_p1p1)[0])
    x.append(pearsonr(pf_ht, pf_p1p1)[0])
plt.semilogx(x)
plt.xlabel('Data set of target unitary')
plt.ylabel('Pearson correlation')
plt.savefig('correlation.png')
plt.savefig('correlation.pdf')
plt.show()
# print(pearsonr(pf_target, pf_p1p1))
# print(pearsonr(pf_target, pf_ht))
exit()

ax[0].plot(pf_ht, 'bo', markersize = 3, label = '$$P_f^{H1,T1}$$')
ax[0].plot(pf_p1p1, 'rx', markersize = 3, label = '$$P_f^{P1,P1}$$')
# ax[0].plot(pf_target, 'gs', markersize = 3, label = '$$P_f^\\textrm{target}$$')
ax[0].axhline(np.mean(pf_ht), color = 'b', linestyle = '-.', label = '$$\\textrm{mean}^{H1,T1}$$')
ax[0].axhline(np.mean(pf_p1p1), color = 'r', linestyle = '-', label = '$$\\textrm{mean}^{P1,P1}$$')
ax[0].fill_between(points, np.mean(pf_ht)-np.std(pf_ht), np.mean(pf_ht)+np.std(pf_ht), color = 'b', alpha=0.3)
ax[0].fill_between(points, np.mean(pf_p1p1)-np.std(pf_p1p1), np.mean(pf_p1p1)+np.std(pf_p1p1), color = 'r', alpha=0.3)
ax[0].set_xlabel('Data set of target unitary')
ax[0].set_ylabel('$$\\textrm{P}_f$$')

fig.legend(ncol=4, fontsize = 9, bbox_to_anchor = (0.75,1.01))

ax[1].semilogy(cd_ht, 'bo', markersize = 3, label = '$$\\textrm{CD}_f^{H1,T1}$$')
ax[1].semilogy(cd_p1p1, 'rx', markersize = 3, label = '$$\\textrm{CD}_f^{P1,P1}$$')
ax[1].axhline(np.mean(cd_ht), color = 'b', linestyle = '-.', label = '$$\\textrm{mean}\; \\textrm{CD}_f^{H1,T1}$$')
ax[1].axhline(np.mean(cd_p1p1), color = 'r', linestyle = '-', label = '$$\\textrm{mean}\; \\textrm{CD}_f^{P1,P1}$$')
ax[1].fill_between(points, np.mean(cd_ht)-np.std(cd_ht), np.mean(cd_ht)+np.std(cd_ht), color = 'b', alpha=0.3)
ax[1].fill_between(points, np.mean(cd_p1p1)-np.std(cd_p1p1), np.mean(cd_p1p1)+np.std(cd_p1p1), color = 'r', alpha=0.3)
ax[1].set_xlabel('Data set of target unitary')
ax[1].set_ylabel('$$\\textrm{Circuit depth (CD)}$$')

plt.tight_layout()

plt.savefig(f'1q_benchmark_{how_many_points}_points.pdf')
plt.savefig(f'1q_benchmark_{how_many_points}_points.png')

plt.show()



# figure(figsize=(6.5, 3.5), dpi=80)
# weights_list = [1, 10,20,30,40,50,60,70,80,90,100]
# weights_list_label = [0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# y_points = []
# for w in weights_list:
#     y = np.mean(np.load(f'results/data/data_{w}1110pf1.npy'))-np.mean(np.load(f'results/data/data_{w}1110pf2.npy'))
#     y_points.append(y)
#     if w != 50:
#         plt.plot(w/100, y, 'bo', markersize=7)
#     else:
#         plt.plot(w/100, y, 'r*', markersize=14)
# plt.xticks(weights_list_label)
# plt.ylabel('$$\Delta P_{af}$$')
# plt.xlabel('$$w_{apf}$$')
# plt.tight_layout()
# plt.savefig('weight_justification.pdf')
# plt.show()