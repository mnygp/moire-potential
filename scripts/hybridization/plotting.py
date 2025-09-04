import numpy as np
import matplotlib.pyplot as plt

ref_data = np.load('ref_values.npy')
# TODO: Triple check the signs of these corrections
# Load strain data to calculate the hybridization for "0%" strain
strain_data = np.genfromtxt('../gap-changes/band_edges_wide.csv',
                            delimiter=',', skip_header=1)

# Correction to the WSe2 HOMO energy
WSe2_minus_4 = strain_data[0, 3]
WSe2_equilibrium = strain_data[15, 3]
WSe2_lattice_corr = WSe2_equilibrium - WSe2_minus_4

# Correction to both the HOMO and LUMO energies
WSe2_minus_2 = strain_data[8, 3]
HOMO_correction = WSe2_equilibrium - WSe2_minus_2
MoS2_plus_2 = strain_data[23, 2]
MoS2_equilibrium = strain_data[15, 2]
LUMO_correction = MoS2_equilibrium - MoS2_plus_2

# Correction to the MoS2 LUMO energy
MoS2_plus_4 = strain_data[30, 2]
MoS2_lattice_corr = MoS2_equilibrium - MoS2_plus_4


corrections = [-WSe2_lattice_corr,
               LUMO_correction-HOMO_correction,
               MoS2_lattice_corr]

print(corrections)

for i, lattice, name in zip([0, 1, 2],
                            [3.184, 3.2515, 3.319],
                            ['MoS2', 'Average', 'WSe2']):
    ref = ref_data[i]
    data = np.load(f'{name}_values.npy')

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    fig4, ax4 = plt.subplots(figsize=(6, 4))

    for j, shift in enumerate([0.0, 0.1, 0.2, 0.3]):
        # Get reference values
        ref_values = ref[j]
        ref_z = ref_values[0]
        ref_homo = ref_values[1]
        ref_lumo = ref_values[2]

        # Get data
        z = data[j, :, 0]
        homo = data[j, :, 1]
        lumo = data[j, :, 2]

        # Plot data
        ax1.plot(z, homo, '-o', color=f'C{2*j}', label=f'{shift}-HOMO')
        ax1.hlines([ref_homo], min(z), max(z), linestyles='dashed',
                   color=f'C{2*j}', alpha=0.6)
        ax1.plot(z, lumo, '-*', color=f'C{2*j+1}', label=f'{shift}-LUMO')
        ax1.hlines([ref_lumo], min(z), max(z), linestyles='dashed',
                   color=f'C{2*j+1}', alpha=0.6)

        ax2.plot(z, homo - ref_homo, '-o', color=f'C{2*j}',
                 label=f'{shift}-HOMO')
        ax2.plot(z, lumo - ref_lumo, '-*', color=f'C{2*j+1}',
                 label=f'{shift}-LUMO')

        ax3.plot(z, lumo - homo, '-o', color=f'C{j}', label=f'{shift}')
        ax3.hlines([ref_lumo-ref_homo], min(z), max(z), linestyles='dashed',
                   color=f'C{j}', alpha=0.6)

        ax4.plot(z, (lumo - homo) - (ref_lumo-ref_homo), '-o',
                 color=f'C{j}', label=f'{shift}')

    ax1.set(title=f'Energies with lattice constant equal to {name}',
            xlabel='z distance [Å]',
            ylabel='Energy - Vacuum [eV]')
    ax1.legend()
    fig1.tight_layout()
    ax1.grid()
    fig1.savefig(f'plots/{name}.png', dpi=500)
    plt.close(fig1)

    ax2.set(title=f'Energies normalised with lattice constant equal to {name}',
            xlabel='z distance [Å]',
            ylabel='Energy - Vacuum [eV]')
    ax2.legend()
    fig2.tight_layout()
    ax2.grid()
    fig2.savefig(f'plots/{name}_normed.png', dpi=500)
    plt.close(fig2)

    ax3.set(title=f'Band gap with lattice constant equal to {name}',
            xlabel='z distance [Å]',
            ylabel='Energy - Vacuum [eV]')
    ax3.legend()
    fig3.tight_layout()
    ax3.grid()
    fig3.savefig(f'plots/{name}_bandgap.png', dpi=500)
    plt.close(fig3)

    ax4.set(title=f'Band gap normalised with lattice constant equal to {name}',
            xlabel='z distance [Å]',
            ylabel='Energy - Vacuum [eV]')
    ax4.legend()
    fig4.tight_layout()
    ax4.grid()
    fig4.savefig(f'plots/{name}_bandgap_normed.png', dpi=500)
    plt.close(fig4)

print('First part done')

for i, shift in enumerate([0.0, 0.1, 0.2, 0.3]):

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    fig8, ax8 = plt.subplots(figsize=(6, 4))

    for j, lattice, name in zip([0, 1, 2],
                                [3.184, 3.2515, 3.319],
                                ['MoS2', 'Average', 'WSe2']):
        ref = ref_data[j]
        ref_values = ref[i]
        ref_z = ref_values[0]
        ref_homo = ref_values[1]
        ref_lumo = ref_values[2]

        data = np.load(f'{name}_values.npy')

        # Get data
        z = data[i, :, 0]
        homo = data[i, :, 1]
        lumo = data[i, :, 2]

        ax5.plot(z, lumo - homo, '-o', color=f'C{j}', label=f'{name}')
        ax5.hlines([ref_lumo-ref_homo], min(z), max(z), linestyles='dashed',
                   color=f'C{j}', alpha=0.6)

        ax6.plot(z, (lumo - homo) - (ref_lumo-ref_homo), '-o',
                 color=f'C{j}', label=f'{name}')

        ax7.plot(z, (lumo - homo) - max(lumo - homo), '-o',
                 color=f'C{j}', label=f'{name}')

        ax8.plot(z, (lumo - homo) + corrections[j], '-o',
                 color=f'C{j}', label=f'{name}')

    ax5.set(title=f'Band gap with shift {shift}',
            xlabel='z distance [Å]',
            ylabel='Energy - Vacuum [eV]')
    ax5.legend()
    fig5.tight_layout()
    ax5.grid()
    fig5.savefig(f'plots/{shift}_bandgap.png', dpi=500)
    plt.close(fig5)

    ax6.set(title=f'Band gap normalised with shift {shift}',
            xlabel='z distance [Å]',
            ylabel='Energy - Vacuum [eV]')
    ax6.legend()
    fig6.tight_layout()
    ax6.grid()
    fig6.savefig(f'plots/{shift}_bandgap_normed.png', dpi=500)
    plt.close(fig6)

    ax7.set(title=f'Band gap normalised to 6.9Å with shift {shift}',
            xlabel='z distance [Å]',
            ylabel='Energy - Vacuum [eV]')
    ax7.legend()
    fig7.tight_layout()
    ax7.grid()
    fig7.savefig(f'plots/{shift}_bandgap_normed_6.9.png', dpi=500)
    plt.close(fig7)

    ax8.set(title=f'Band gap with strain correction from {name} lattice',
            xlabel='z distance [Å]',
            ylabel='Energy - Vacuum [eV]')
    ax8.legend()
    fig8.tight_layout()
    ax8.grid()
    fig8.savefig(f'plots/{shift}_bandgap_strain_normed.png', dpi=500)
    plt.close(fig8)