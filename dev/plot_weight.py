import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 14}
)
from matplotlib.pyplot import figure
figure(figsize=(6.5, 3.5), dpi=80)
weights_list = [1, 10,20,30,40,50,60,70,80,90,100]
weights_list_label = [0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
y_points = []
for w in weights_list:
    y = np.mean(np.load(f'results/data/data_{w}1110pf1.npy'))-np.mean(np.load(f'results/data/data_{w}1110pf2.npy'))
    y_points.append(y)
    if w != 50:
        plt.plot(w/100, y, 'bo', markersize=7)
    else:
        plt.plot(w/100, y, 'r*', markersize=14)
plt.xticks(weights_list_label)
plt.ylabel('$$\Delta P_{af}$$')
plt.xlabel('$$w_{apf}$$')
plt.tight_layout()
plt.savefig('weight_justification.pdf')
plt.show()