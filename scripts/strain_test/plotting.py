import matplotlib.pyplot as plt
import numpy as np

shifts = ['0.00', '0.33', '0.67']

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

for shift in shifts:
    data = np.loadtxt(f"bandgap_shift_{shift}.csv", delimiter=',', skiprows=1)
    lattice_lengths = data[:, 0]
    distance = data[:, 4]
    bandgap = data[:, 3]

    # Left plot: z distance
    axs[0].plot(lattice_lengths, distance, label=f'Shift: {shift}', marker='o')

    # Right plot: band gap
    axs[1].plot(lattice_lengths, bandgap, label=f'Shift: {shift}', marker='o')

# Customize left plot
axs[0].set_title('Distance between TM layers')
axs[0].set_xlabel('Lattice Length (Å)')
axs[0].set_ylabel('z distance (Å)')
axs[0].legend()
axs[0].grid(True)

# Customize right plot
axs[1].set_title('Band Gap vs Lattice Length')
axs[1].set_xlabel('Lattice Length (Å)')
axs[1].set_ylabel('Band Gap (eV)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('z_distance_and_bandgap.png', dpi=500)
plt.close()
