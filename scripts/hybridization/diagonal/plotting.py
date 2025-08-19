import matplotlib.pyplot as plt
import numpy as np

import scienceplots  # noqa: F401

plt.style.use('science')
fig, ax = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True,
                       gridspec_kw={'height_ratios': [1, 2]})
fig2, ax2 = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True,
                         gridspec_kw={'height_ratios': [1, 2]})

for a, color, label in zip([3.184, 3.2515, 3.319],
                           ['C0', 'C1', 'C2'],
                           ['a=MoS2', 'a=Average', 'a=WSe2']):
    data = np.loadtxt(f'gap_shift_{a:.2f}.csv', delimiter=',', skiprows=1)

    shift_arr = data[:, 0]
    z_dist_arr = data[:, 1]
    gap_arr = data[:, 2]

    ax[0].plot(shift_arr, gap_arr - min(gap_arr), '-o',
               label=label, color=color)
    ax[1].plot(shift_arr, z_dist_arr, '-o',
               label=label, color=color)

    ax2[0].plot(shift_arr, gap_arr, '-o',
                label=label, color=color)
    ax2[1].plot(shift_arr, z_dist_arr, '-o',
                label=label, color=color)


ax[0].set_ylabel('Band Gap (eV) \n (Relative to min)')
ax[0].grid()
ax[1].set_xlabel('Shift')
ax[1].set_ylabel('Z distance (Å)')
ax[1].legend()
ax[1].grid()
fig.tight_layout()
fig.savefig('gap_shift_plot_norm.png', dpi=500)

ax2[0].set_ylabel('Band Gap (eV)')
ax2[0].grid()
ax2[1].set_xlabel('Shift')
ax2[1].set_ylabel('Z distance (Å)')
ax2[1].legend()
ax2[1].grid()
fig2.tight_layout()
fig2.savefig('gap_shift_plot.png', dpi=500)
