from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from functions.geometry import interlayer_distance, strain


def scatter_plot_color(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                       filename: str, title: str, color_map: str,
                       label: str, strings=None):

    fig, ax = plt.subplots()

    scatter = ax.scatter(X, Y, c=Z, cmap=color_map, s=50, edgecolor='none')
    fig.colorbar(scatter, ax=ax, label=label)

    ax.set_xlabel("X Position [Å]")
    ax.set_ylabel("Y Position [Å]")
    ax.axis('equal')
    ax.set_title(title)

    if strings is not None:
        text = '\n'.join(strings)
        ax.text(0.03, 0.17, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top')

    fig.savefig(f"plots/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

mace_atoms = read('structure_MACE.json')
mattersim_atoms = read('structure_mattersim.json')
dft_atoms = read('structure_DFT.json')

structs = [dft_atoms, mace_atoms, mattersim_atoms]


x, y, z_dft = interlayer_distance(dft_atoms)
_, _, Mo_strain_dft = strain(dft_atoms, 'Mo')
_, _, W_strain_dft = strain(dft_atoms, 'W')

_, _, z_mace = interlayer_distance(mace_atoms)
_, _, Mo_strain_mace = strain(mace_atoms, 'Mo')
_, _, W_strain_mace = strain(mace_atoms, 'W')

_, _, z_mattersim = interlayer_distance(mattersim_atoms)
_, _, Mo_strain_mattersim = strain(mattersim_atoms, 'Mo')
_, _, W_strain_mattersim = strain(mattersim_atoms, 'W')

for s, name in zip(structs, ['DFT', 'MACE', 'MatterSim']):
    z_dist_x, z_dist_y, z_dist = interlayer_distance(s)
    Mo_strain_x, Mo_strain_y, Mo_strain = strain(s, 'Mo')
    W_strain_x, W_strain_y, W_strain = strain(s, 'W')

    if (name != 'DFT'):
        z_error = np.mean(z_dft - z_dist)
        z_error_std = np.std(z_dft - z_dist)
        print(f'Average error in z dist for {name}: {z_error:.3f} +/- {z_error_std:.3f}')
        print(f'Error range for {name}: {min(z_dft - z_dist) - max(z_dft - z_dist):.4f}')
        print()

        Mo_error = np.mean(Mo_strain_dft - Mo_strain)
        Mo_error_std = np.std(Mo_strain_dft - Mo_strain)
        print(f'Average error in Mo strain for {name}: {Mo_error:.5f} +/- {Mo_error_std:.5f}')

        W_error = np.mean(W_strain_dft - W_strain)
        W_error_std = np.std(W_strain_dft - W_strain)
        print(f'Average error in W strain for {name}: {W_error:.5f} +/- {W_error_std:.5f}')

mace_dif = z_mace - z_dft
mattersim_dif = z_mattersim - z_dft
print("-----------------------")
print("DFT z interval:")
print(max(z_dft), min(z_dft))
print(max(z_dft) - min(z_dft))

print("Mace z interval:")
print(max(z_mace), min(z_mace))
print(max(z_mace) - min(z_mace))


print("MatterSim z interval:")
print(max(z_mattersim), min(z_mattersim))
print(max(z_mattersim) - min(z_mattersim))
print("----------------------")

max_val = max(max(mace_dif), max(mattersim_dif))
min_val = min(min(mace_dif), min(mattersim_dif))

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(mace_dif, bins=100, range=(min_val, max_val),
        alpha=0.8, label='MACE')
ax.hist(mattersim_dif, bins=100, range=(min_val, max_val),
        alpha=0.8, label='MatterSim')

ax.vlines(0, 0, 25, colors='black', linestyles='dashed', label='DFT reference')

ax.set_xlabel('Z distance difference [Å]')
ax.set_ylabel('Count')
ax.legend()
ax.set_title('The difference in interlayer distance compared to DFT')
ax.grid(True)
fig.tight_layout()

fig.savefig('plots/z_distance_difference_histogram.png', dpi=500)
plt.close(fig)  # optional: closes figure if running in loops


DFT_data = np.genfromtxt('DFT_data.csv', delimiter=',', skip_header=1)
MACE_data = np.genfromtxt('MACE_data.csv', delimiter=',', skip_header=1)
MAtterSim_data = np.genfromtxt('MatterSim_data.csv', delimiter=',',
                               skip_header=1)

shift = DFT_data[:, 0]
# z distance [:, 1]
# vec1 [:, 2]
# vec2 [:, 3]
# gap [:, 4]

fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7, 6))

"""
for data, color, name in zip([DFT_data, MACE_data, MAtterSim_data],
                             ['C0', 'C1', 'C2'],
                             ['DFT', 'MACE', 'MatterSim']):
"""
for data, color, name in zip([DFT_data, MAtterSim_data],
                             ['C0','C2'],
                             ['DFT', 'MatterSim']):
    z_dist = data[:, 1]
    average_vec = (data[:, 2] + data[:, 3]) / 2
    gap = data[:, 4]

    ax1[0].plot(shift, z_dist, '-o', color=color)
    # ax1[1].plot(shift, (average_vec - 3.184)/3.184 * 100, '-o', color=color)
    ax1[1].plot(shift, gap, '-o', color=color, label=name)

ax1[1].set_xlabel('Shift [fraction of lattice vector]')
ax1[0].set_ylabel('Interlayer distance [Å]')
# ax1[1].set_ylabel('Lattice deviation from MoS2 [%]')
ax1[1].set_ylabel('Band gap [eV]')
ax1[1].legend()
ax1[0].set_title('Comparison of DFT and MatterSim')
ax1[0].grid(True)
ax1[1].grid(True)
# ax1[2].grid(True)
fig1.tight_layout()
fig1.savefig('plots/DFT_MACE_MatterSim_comparison.png', dpi=500)
