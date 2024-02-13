import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import re
import os

dir_res = 'results/data/'

fnames = []
for (dirpath, dirnames, filenames) in os.walk(dir_res):
    fnames.extend(filenames)

y_pf_corr = []
for file in fnames:
    match = re.search("NUSA_eid-001([1-5])_.*pf1", file) 
    if match:
        pf_ht = np.load(dir_res+file)
        pf_p1p1 = np.load(dir_res+file[:-5]+'2.npy')
        y_pf_corr.append([int(match.group(1)),pearsonr(pf_ht, pf_p1p1)[0]])
        print(pearsonr(pf_ht, pf_p1p1)[0],file)

exit()

print(y_pf_corr)
x, y = zip(*y_pf_corr)
plt.plot(x, y, "_", markersize=4, label = 'correlation PF')
plt.xticks([1,2,3,4,5],[50,40,30,20,10])
plt.xlabel('w_npf in [50,w_npf,1,1,0]')
plt.ylabel('Pearson correlation')
plt.show()

exit()

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 17}
)

# from matplotlib.pyplot import figure
fig, ax = plt.subplots(1,1, figsize=(4, 4), dpi = 100)
exp_name_list = ['0011_2024-02-09-15-20', '0012_2024-02-09-15-31', '0013_2024-02-09-15-33',\
                  '0014_2024-02-09-15-34', '0015_2024-02-09-15-43', '0016_2024-02-09-15-43' ]
exp_name_list = ['0014_2024-02-09-15-34', '0014_2024-02-09-16-10', '0014_2024-02-09-16-24' ]
## ------------------- process fidelity data load --------------------
y_pf_corr = []
for exp_id in exp_name_list:
    pf_ht = np.load(f'results/data/NUSA_eid-{exp_id}pf1.npy')
    pf_p1p1 = np.load(f'results/data/NUSA_eid-{exp_id}pf2.npy')
    pf_target = [1-x for x in pf_ht]#

    ## correlation calculator
    y_pf_corr.append(pearsonr(pf_ht, pf_p1p1)[0])
    print(pearsonr(pf_ht, pf_p1p1)[0], np.mean(pf_p1p1)-np.mean(pf_ht), exp_id)
# plt.plot(list(reversed(y_pf_corr)), label = 'correlation PF')
# plt.xlabel('Data set of target unitary')
# plt.ylabel('Pearson correlation')
# plt.tight_layout()
# plt.legend()
# # plt.savefig('correlation.png')
# # plt.savefig('correlation.pdf')
# plt.show()



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