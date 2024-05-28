import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 12}
)

direc = 'yaqq_test.csv'
df = pd.read_csv(direc, delimiter=';')
df = df.sort_values('Size')

fig, axs = plt.subplots(2,1,figsize=(17,9),dpi=80)

axs[0].set_ylabel('Process fidelity (PF)', fontsize=20, labelpad=15)
axs[0].plot(df['benchmark'], df['fidelity yaqq'], 'r-v', markersize = 8, label = '$$\\textrm{PF}\;[\\textrm{P}1_{\\vec{a}}^{opt. 200},\;\\textrm{P}1_{\\vec{b}}^{opt. 200},\;\\textrm{P}1_{\\vec{a}}^{opt. 200\;\\dagger},\;\\textrm{P}1_{\\vec{b}}^{opt. 200\;\\dagger},\;\\textrm{CX}2]$$')
axs[0].plot(df['benchmark'], df['fidelity universal'], 'b-^', markersize = 8, label='$$\\textrm{PF}\;[\\textrm{H}1,\;\\textrm{T}1,\;\\textrm{T}1^{\\dagger},\;\\textrm{CX}2]$$') 
axs[0].tick_params(axis='x', labelbottom=False)
axs[0].yaxis.set_tick_params(labelsize=14)
axs[0].legend(loc='lower left', fontsize=14)

axs[1].set_ylabel('Circuit depth (CD)', fontsize=20, labelpad=15)
axs[1].plot(df['benchmark'], df['circ_depth yaqq'], 'r-s', markersize = 8, label = '$$\\textrm{CD}\;[\\textrm{P}1_{\\vec{a}}^{opt. 200},\;\\textrm{P}1_{\\vec{b}}^{opt. 200},\;\\textrm{P}1_{\\vec{a}}^{opt. 200\;\\dagger},\;\\textrm{P}1_{\\vec{b}}^{opt. 200\;\\dagger},\;\\textrm{CX}2]$$')
axs[1].plot(df['benchmark'], df['circ_depth universal'], 'b-o', markersize = 8, label='$$\\textrm{CD}\;[\\textrm{H}1,\;\\textrm{T}1,\;\\textrm{T}1^{\\dagger},\;\\textrm{CX}2]$$') 
axs[1].set_xticklabels(df['benchmark'], fontsize=16, rotation=45, ha = 'right')
axs[1].yaxis.set_tick_params(labelsize=14)
axs[1].set_yscale('log')
axs[1].legend(loc='upper left', fontsize=14)

axs[1].set_xlabel('Benchmark and no. of qubits', fontsize=20, labelpad=15)

plt.subplots_adjust(left=0.062, right=0.99, top=0.99, bottom=0.22, wspace=0.19, hspace=0.16)

plt.savefig('mqt_yaqq_p200.pdf', dpi=400)
plt.show()