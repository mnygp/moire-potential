import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('gap_shift.csv', delimiter=',', skiprows=1)

shift_arr = data[:, 0]
post_z_dist_arr = data[:, 1]
pre_gap_arr = data[:, 2]
post_gap_arr = data[:, 3]


fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(shift_arr, pre_gap_arr, '-o', label='Pre-relaxation')
ax[0].plot(shift_arr, post_gap_arr, '-o', label='Post-relaxation')
ax[0].set_ylabel('Band Gap (eV)')
ax[0].legend()
ax[0].grid()
ax[1].plot(shift_arr, post_z_dist_arr, '-o',
           label='Post-Z Distance', color='green')
ax[1].set_xlabel('Shift')
ax[1].set_ylabel('Z Distance (Å)')
ax[1].legend()
ax[1].grid()
plt.tight_layout()
plt.savefig('gap_shift_plot.png', dpi=500)
