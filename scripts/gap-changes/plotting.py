import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('band_edges_medium.csv', skip_header=1,
                     dtype=float, delimiter=',')

strain = (data[:, 0] - 1)*100
MoS2_homo = data[:, 1]
MoS2_lumo = data[:, 2]
WSe2_homo = data[:, 3]
WSe2_lumo = data[:, 4]


plt.plot(strain, MoS2_homo, '-o', label='MoS2 Homo')
plt.plot(strain, MoS2_lumo, '-o', label='MoS2 Lumo')
plt.plot(strain, WSe2_homo, '-o', label='WSe2 Homo')
plt.plot(strain, WSe2_lumo, '-o', label='WSe2 Lumo')
plt.xlabel("Layer strain [%]")
plt.ylabel("Energy [eV]")
plt.title("HOMO and LUMO energy levels")
plt.legend()
plt.grid()
plt.savefig('Homo-lumo-strain-medium.png', dpi=500)
plt.close()

lumo_grid, homo_grid = np.meshgrid(MoS2_lumo, WSe2_homo)

band_gap_grid = lumo_grid - homo_grid

im = plt.imshow(band_gap_grid, extent=(strain[0], strain[-1],
                                       strain[0], strain[-1]),
                origin="lower")
plt.xlabel("MoS2 strain [%]")
plt.ylabel("WSe2 strain [%]")
plt.title("Band gap as a function of layer strain")
plt.colorbar(im, label="Band Gap (eV)")
plt.tight_layout()
plt.savefig("band-gap-grid-medium.png", dpi=500)
