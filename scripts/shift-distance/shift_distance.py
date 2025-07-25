from functions.structure import create_bilayer
from gpaw import GPAW, PW
import numpy as np
import matplotlib.pyplot as plt
from ase.parallel import parprint

shifts = np.linspace(0, 1, 20)
distances = np.linspace(6.25, 7.0, 20)

gap_matrix = np.zeros((len(shifts), len(distances)))
data = []


for i, shift in enumerate(shifts):
    for j, dist in enumerate(distances):
        bilayer = create_bilayer(z_dist=dist, lattice_length=3.2515,
                                 a_shift=shift, b_shift=shift)

        calc = GPAW(mode=PW(500), xc='PBE', kpts={'size': (20, 20, 1)},
                    txt='output.txt')

        bilayer.calc = calc

        bilayer.get_potential_energy()
        homo, lumo = calc.get_homo_lumo()
        gap = lumo - homo
        parprint(f"Gap calculated for shift={shift:.2f} and dist={dist:.2f}")

        gap_matrix[i, j] = gap
        data.append([dist, shift, gap])


# Save CSV file with flat data
np.savetxt("bandgap_data.csv", np.array(data), delimiter=",",
           header="distance,shift,gap", comments='')

# Plot heatmap from gap_matrix
plt.figure(figsize=(6, 5))
plt.imshow(gap_matrix, origin='lower',
           extent=[distances[0], distances[-1], shifts[0], shifts[-1]],
           aspect='auto', cmap='viridis')
plt.colorbar(label='Band Gap (eV)')
plt.xlabel("Interlayer Distance (Å)")
plt.ylabel("Shift along diagonal")
plt.title("Band Gap Heatmap for MoS2/WSe2")
plt.tight_layout()
plt.savefig("bandgap_heatmap.png", dpi=300)
plt.close()
