import numpy as np
import matplotlib.pyplot as plt
from functions.geometry import strain
from functions.util import repeate_cells
from ase.io import read
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

# The medium data set goes from -2% to 2% along both axis
data = np.genfromtxt('band_edges_medium.csv', skip_header=1,
                     dtype=float, delimiter=',')

strain_data = (data[:, 0] - 1)*100
MoS2_homo = data[:, 1]
MoS2_lumo = data[:, 2]
WSe2_homo = data[:, 3]
WSe2_lumo = data[:, 4]

lumo_grid, homo_grid = np.meshgrid(MoS2_lumo, WSe2_homo)
band_gap_grid = lumo_grid - homo_grid

# ################# Plot HOMO and LUMO Levels as a function of strain #########
plt.plot(strain_data, MoS2_homo, '-o', label='MoS2 Homo')
plt.plot(strain_data, MoS2_lumo, '-o', label='MoS2 Lumo')
plt.plot(strain_data, WSe2_homo, '-o', label='WSe2 Homo')
plt.plot(strain_data, WSe2_lumo, '-o', label='WSe2 Lumo')
plt.xlabel("Layer strain [%]")
plt.ylabel("Energy [eV]")
plt.title("HOMO and LUMO energy levels")
plt.legend()
plt.grid()
plt.savefig('Homo-lumo-strain-medium.png', dpi=500)
plt.close()


# ############### Plot gap as a function of strain ############################
im = plt.imshow(band_gap_grid, extent=(strain_data[0], strain_data[-1],
                                       strain_data[0], strain_data[-1]),
                origin="lower")
plt.xlabel("MoS2 strain [%]")
plt.ylabel("WSe2 strain [%]")
plt.title("Band gap as a function of layer strain")
plt.colorbar(im, label="Band Gap (eV)")
plt.tight_layout()
plt.savefig("band-gap-grid-medium.png", dpi=500)
plt.close()

# ############### Plot actual strain values on strain map ####################
struct = read('../../structures/MoS2-WSe2-MatterSim/'
              '1.11_2940/structure_ml.json')

x_Mo, y_Mo, Mo_strain = strain(struct, 'Mo')
x_W, y_W, W_strain = strain(struct, 'W')

(x_W_large, y_W_large,
 W_strain_large) = repeate_cells(x_W, y_W, W_strain,
                                 range(-1, 2),
                                 struct.cell[0, :2],  # type: ignore
                                 struct.cell[1, :2])  # type: ignore

interp = LinearNDInterpolator(list(zip(x_W_large, y_W_large)), W_strain_large)

interp_W_strain = interp(x_Mo, y_Mo)

ref_gap = MoS2_lumo[-1] - WSe2_homo[0]

band_gap_correction = (lumo_grid - homo_grid) - ref_gap

im = plt.imshow(band_gap_correction,
                extent=(strain_data[0], strain_data[-1],
                        strain_data[0], strain_data[-1]),
                origin="lower", interpolation='spline16',
                vmin=-0.01, vmax=0.06)

plt.scatter(Mo_strain*100, interp_W_strain*100, marker='x', color='black')


plt.xlabel("MoS2 strain [%]")
plt.ylabel("WSe2 strain [%]")
plt.title("Band gap correction as a function of layer strain")
plt.xlim(-0.3, 0.6)
plt.ylim(-0.3, 0.8)
plt.colorbar(im, label="Band Gap [eV]")
plt.tight_layout()
plt.savefig("band-gap-correction.png", dpi=500)
plt.close()

# ################ Plot histogram of correction ##############################
strain_to_gap_interp = RegularGridInterpolator((strain_data, strain_data),
                                               band_gap_grid - ref_gap)
print(Mo_strain.shape, W_strain.shape)
print(np.max(Mo_strain), np.max(interp_W_strain))
print(np.min(Mo_strain), np.min(interp_W_strain))
corrections = strain_to_gap_interp(list(zip(Mo_strain, interp_W_strain)))

plt.hist(corrections, bins=50)
plt.title("Histogram of the strain correction at every Mo atom")
plt.xlabel("Strain correction [eV]")
plt.savefig("correction-histogram.png", dpi=500)


# ############### Plot correction as a function of position ################
print(min(corrections), max(corrections))
correction_interp = LinearNDInterpolator(list(zip(x_Mo, y_Mo)), corrections)


X, Y = np.meshgrid(np.linspace(-36, 72, 800),
                   np.linspace(0, 63, 800))

# interpolated_dist = dist_interp(X, Y)
interpolated_gap = correction_interp(X, Y)

mesh = plt.pcolormesh(X, Y, interpolated_gap, shading="auto")
plt.colorbar(mesh, label="Gap [eV]", cmap='viridis')
plt.title("Strain corrected local band gap")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("gap_correction.png", dpi=500)
plt.close()
