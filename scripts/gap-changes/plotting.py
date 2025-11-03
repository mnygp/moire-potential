import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functions.geometry import strain
from functions.util import repeate_cells
from ase.io import read
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

# The medium data set goes from -2% to 2% along both axis
data = np.genfromtxt("band_edges_medium_soc.csv",
                     skip_header=1, dtype=float, delimiter=",")

strain_data = (data[:, 0] - 1) * 100
MoS2_homo = data[:, 1]
MoS2_lumo = data[:, 2]
WSe2_homo = data[:, 3]
WSe2_lumo = data[:, 4]

lumo_grid, homo_grid = np.meshgrid(MoS2_lumo, WSe2_homo)
band_gap_grid = lumo_grid - homo_grid

# ################# Plot HOMO and LUMO Levels as a function of strain #########
plt.plot(strain_data, MoS2_homo, "-o", label="MoS2 Homo")
plt.plot(strain_data, MoS2_lumo, "-o", label="MoS2 Lumo")
plt.plot(strain_data, WSe2_homo, "-o", label="WSe2 Homo")
plt.plot(strain_data, WSe2_lumo, "-o", label="WSe2 Lumo")
plt.xlabel("Layer strain [%]")
plt.ylabel("Energy [eV]")
plt.title("HOMO and LUMO energy levels")
plt.legend()
plt.grid()
plt.savefig("Homo-lumo-strain-medium-soc.png", dpi=500)
plt.close()


# ############### Plot gap as a function of strain ############################
im = plt.imshow(
    band_gap_grid,
    extent=(strain_data[0], strain_data[-1], strain_data[0], strain_data[-1]),
    origin="lower",
    interpolation='spline16'
)
plt.xlabel("MoS2 strain [%]")
plt.ylabel("WSe2 strain [%]")
plt.title("Band gap as a function of layer strain")
plt.colorbar(im, label="Band Gap (eV)")
plt.tight_layout()
plt.savefig("band-gap-grid-medium-soc.png", dpi=500)
plt.close()

# ############### Plot actual strain values on strain map ####################
struct = read("../../structures/MoS2-WSe2-MatterSim/"
              "1.11_2946/structure_ml.json")

x_Mo, y_Mo, Mo_strain = strain(struct, "Mo")
x_W, y_W, W_strain = strain(struct, "W")

(x_W_large, y_W_large, W_strain_large) = repeate_cells(
    x_W,
    y_W,
    W_strain,
    range(-1, 2),
    struct.cell[0, :2],  # type: ignore
    struct.cell[1, :2],  # type: ignore
)

interp = LinearNDInterpolator(list(zip(x_W_large, y_W_large)), W_strain_large)

interp_W_strain = interp(x_Mo, y_Mo)

ref_gap = MoS2_lumo[-1] - WSe2_homo[0]

band_gap_correction = (lumo_grid - homo_grid) - ref_gap

vmin, vmax = 0, 200
masked = np.ma.masked_outside(band_gap_correction, vmin - 0.01, vmax + 0.01)

fig = plt.figure(figsize=(6, 8))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

im = ax1.imshow(
    masked*1000,
    extent=(strain_data[0], strain_data[-1], strain_data[0], strain_data[-1]),
    origin="lower",
    interpolation="spline16",
    vmin=vmin,
    vmax=vmax,
)

ax1.scatter(Mo_strain * 100, interp_W_strain * 100, marker="x",
            color="black", label="Values at ")


ax1.set_xlabel("MoS2 strain [%]")
ax1.set_ylabel("WSe2 strain [%]")
ax1.set_title("Band gap correction as a function of layer strain")
ax1.set_xlim(-0.3, 0.6)
ax1.set_ylim(-0.3, 0.8)
cbar = fig.colorbar(im, ax=ax1, label="Band Gap correction [meV]")
# ax1.tight_layout()
# ax1.savefig("band-gap-correction.png", dpi=500)
# ax1.close()

# ##### Plot histogram of correction ###
strain_to_gap_interp = RegularGridInterpolator(
    (strain_data, strain_data), band_gap_grid - ref_gap
)
corrections = strain_to_gap_interp(list(zip(Mo_strain, interp_W_strain)))

ax2.hist(corrections*1000, bins=50)
ax2.set_title("Histogram of the strain correction at every Mo atom")
ax2.set_xlabel("Strain correction [meV]")
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.25)
plt.savefig("correction-values_soc.png", dpi=500)
plt.close()

# ############### Plot correction as a function of position ################
correction_interp = LinearNDInterpolator(list(zip(x_Mo, y_Mo)), corrections)

X, Y = np.meshgrid(np.linspace(-36, 72, 800), np.linspace(0, 63, 800))

# interpolated_dist = dist_interp(X, Y)
interpolated_gap = correction_interp(X, Y)

mesh = plt.pcolormesh(X, Y, (interpolated_gap - min(corrections))*1000,
                      shading="auto")
plt.colorbar(mesh, label="Gap [meV]", cmap="viridis")
plt.title("Strain corrected local band gap")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("gap_correction_soc.png", dpi=500)
plt.close()
