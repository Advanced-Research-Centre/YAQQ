import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

weights = [0.0, 0.2, 0.5, 0.8]
# drift_hamiltonian = h_d_1_qubit
# control_hamiltonian = h_c_yaqq
# hamiltonian_label = h_l_yaqq
hamiltonian_label = ["$u_x$ control", "$u_y$ control"]
timespace = np.linspace(0,2*np.pi,500) # np.linspace(0, gate_duration, number_of_timesteps)
cH_nos = 2 # len(control_hamiltonian)

fig, ax = plt.subplots(cH_nos, 2, sharex = True)
xticks = [0, np.pi, 2 * np.pi]
# colors = ['#03080c','#214868', '#5b97ca', '#9fc2e0']
colors = ['blue','red', 'green', 'goldenrod']

cmap = mpl.colormaps['tab20b'].colors
colors = [cmap[12], cmap[13], cmap[14], cmap[15]]


control_pulses = np.load("p_1_opt_500_a.npy")

for i in range(cH_nos):
    for index_weight, value_weight in enumerate(weights):
        ax[i][0].plot(timespace, control_pulses[i, :, index_weight], label = f"$w_f$ = {round(1 - value_weight, 1)}, $w_e$ = {round(value_weight, 1)}", color = colors[index_weight], alpha = 0.9)
        ax[i][0].set_ylim(-0.4, 0.4)
        if i == 0:
            ax[i][0].set_title(r"Energy-Optimized-GRAPE Pulse for $P1_{\vec{a}}^{opt.500}$", fontsize = 15, pad = 20)
        else:
            ax[i][0].set_xlabel("Time", fontsize = 14)
        ax[i][0].set_ylabel(f"{hamiltonian_label[i]}", fontsize = 14)

control_pulses = np.load("p_1_opt_500_b.npy")
        
for i in range(cH_nos):
    for index_weight, value_weight in enumerate(weights):
        ax[i][1].plot(timespace, control_pulses[i, :, index_weight], label = f"$w_f$ = {round(1 - value_weight, 1)}, $w_e$ = {round(value_weight, 1)}", color = colors[index_weight], alpha = 0.9)
        ax[i][1].set_ylim(-0.4, 0.4)
        if i == 0:
            ax[i][1].set_title(r"Energy-Optimized-GRAPE Pulse for $P1_{\vec{b}}^{opt.500}$", fontsize = 15, pad = 20)
        else:
            ax[i][1].set_xlabel("Time", fontsize = 14)
        ax[i][1].set_ylabel(f"{hamiltonian_label[i]}", fontsize = 14)
        

plt.xticks(xticks, ['0', '$\pi$', '2 $\pi$'])
plt.legend(loc='upper right',bbox_to_anchor=(1.46, 1.35), fontsize=12)
# plt.subplot_tool()
fig.set_size_inches(16, 6)
plt.tight_layout()
plt.subplots_adjust(left=0.06, right=0.83, top=0.9, bottom=0.1, wspace=0.2, hspace=0.095)
plt.savefig('pulse_p1ab500.jpg', format='jpg', dpi=300)
plt.show()