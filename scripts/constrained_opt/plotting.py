import matplotlib.pyplot as plt
import numpy as np

import scienceplots  # noqa: F401

plt.style.use('science')

data = np.loadtxt('gap_shift.csv', delimiter=',', skiprows=1)

shift_arr = data[:, 0]
post_z_dist_arr = data[:, 1]
pre_gap_arr = data[:, 2]
post_gap_arr = data[:, 3]


fig, ax = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True,
                       gridspec_kw={'height_ratios': [1, 2]})
# ax[0].plot(shift_arr, pre_gap_arr, '-o', label='Pre-relaxation')
ax[0].plot(shift_arr, post_gap_arr, '-o', label='Post-relaxation')
ax[0].set_ylabel('Band Gap (eV)')
ax[0].set_ylim(0.17, 0.30)
# ax[0].legend()
ax[0].grid()
ax[1].plot(shift_arr, post_z_dist_arr, '-o',
           label='Post-Z Distance', color='green')
ax[1].set_xlabel('Shift')
ax[1].set_ylabel('Z distance (Å)')
# ax[1].legend()
ax[1].grid()
plt.tight_layout()
plt.savefig('gap_shift_plot.png', dpi=500)
