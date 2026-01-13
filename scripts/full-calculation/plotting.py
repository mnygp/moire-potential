import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from ase.io import read
from functions.util import repeate_cells

path = '../../structures/MoS2-WSe2-MatterSim/1.11_2946/structure_ml.json'
atoms = read(path)
v1 = atoms.cell[0, :2]
v2 = atoms.cell[1, :2]
print(v1, v2)

data = np.genfromtxt('results_fixed_cell_fixed_TM_scissors.csv', dtype=float,
                     skip_header=1, delimiter=',')


# x,y,i,j,z_ml,z_dft,gap,correction,Mo_strain,W_strain

x = data[:, 0]
y = data[:, 1]
shift_a = data[:, 2]
shift_b = data[:, 3]
z_ml = data[:, 4]
z_relax = data[:, 5]
gap = data[:, 6]
corr = data[:, 7]
Mo_strain = data[:, 8]
W_strain = data[:, 9]

# x_L, y_L, gap = repeate_cells(x, y, gap, range(-1, 2),
#                           atoms.cell[0, :2], atoms.cell[1, :2])

print(np.mean(gap+corr), np.std(gap+corr))
print(min(gap+corr), max(gap+corr))

gap_interp = LinearNDInterpolator(list(zip(x, y)), gap)
corr_interp = LinearNDInterpolator(list(zip(x, y)), corr)

Mo_strain_interp = LinearNDInterpolator(list(zip(x, y)), Mo_strain)
W_strain_interp = LinearNDInterpolator(list(zip(x, y)), W_strain)

z_ml_interp = LinearNDInterpolator(list(zip(x, y)), z_ml)

X, Y = np.meshgrid(np.linspace(min(x), max(x), 800),
                   np.linspace(min(y), max(y), 800))

# Non corrected gap
interpolated_gap = gap_interp(X, Y)

plt.figure(figsize=(6, 5))

v_min = np.nanmin((interpolated_gap - np.nanmin(interpolated_gap))*1000)
v_max = np.nanmax((interpolated_gap - np.nanmin(interpolated_gap))*1000)

im = plt.imshow(
    (interpolated_gap - np.nanmin(interpolated_gap)) * 1000,
    origin='lower',
    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
    cmap='viridis',
    aspect='equal',
    vmin=v_min,
    vmax=v_max
)

plt.colorbar(im, label="Gap [meV]")
plt.title("Non-corrected local band gap")
plt.xlabel("x [Å]")
plt.ylabel("y [Å]")
plt.tight_layout()
plt.savefig("plots/non-corrected-gap.png", dpi=500)
plt.close()


# Gap correction
interpolated_correction = corr_interp(X, Y)

plt.figure(figsize=(6, 5))
im = plt.imshow(
    (interpolated_correction - np.nanmin(interpolated_correction))*1000,
    origin='lower',
    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
    cmap='viridis',
    aspect='equal'
)
plt.colorbar(im, label="Gap [eV]")
plt.title("Strain correction")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("plots/gap-correction.png", dpi=500)
plt.close()


# Corrected gap
corrected = interpolated_gap + interpolated_correction

plt.figure(figsize=(6, 5))

im = plt.imshow(
    (corrected - np.nanmin(corrected))*1000,
    origin='lower',
    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
    cmap='viridis',
    aspect='equal',
    vmin=v_min,
    vmax=v_max
)
plt.colorbar(im, label="Gap [meV]")
plt.title("Corrected Gap")
plt.xlabel("x [Å]")
plt.ylabel("y [Å]")
plt.tight_layout()
plt.savefig("plots/corrected-gap.png", dpi=500)
plt.close()


# Z distance
interpolated_z = z_ml_interp(X, Y)

plt.figure(figsize=(6, 5))
# mesh = plt.pcolormesh(X, Y, interpolated_z, shading="auto")
im = plt.imshow(
    interpolated_z,
    origin='lower',
    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
    cmap='viridis',
    aspect='equal'
)
plt.colorbar(im, label="Z distance [Å]")
plt.title("Interlayer Distance")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("plots/Mo-strain.png", dpi=500)
plt.close()

# Mo strain
interpolated_Mo_strain = Mo_strain_interp(X, Y)

plt.figure(figsize=(6, 5))
# mesh = plt.pcolormesh(X, Y, interpolated_Mo_strain*100, shading="auto")
im = plt.imshow(
    interpolated_Mo_strain*100,
    origin='lower',
    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
    cmap='viridis',
    aspect='equal'
)
plt.colorbar(im, label="Gap")
plt.title("Mo Strain")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("plots/Mo-strain.png", dpi=500)
plt.close()


# W strain
interpolated_W_strain = W_strain_interp(X, Y)

plt.figure(figsize=(6, 5))
# mesh = plt.pcolormesh(X, Y, interpolated_W_strain*100, shading="auto")
im = plt.imshow(
    interpolated_Mo_strain*100,
    origin='lower',
    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
    cmap='viridis',
    aspect='equal'
)
plt.colorbar(im, label="Gap")
plt.title("W Strain")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("plots/W-strain.png", dpi=500)
plt.close()

# Line plot
n_points = 100
t_line = np.linspace(0, 1, n_points)
# Generate points along the line
line_points = np.linspace(v1, v2, n_points)
x_line = line_points[:, 0]
y_line = line_points[:, 1]
gap_line = gap_interp(x_line, y_line)*1000
corr_line = corr_interp(x_line, y_line)*1000

# Corrected gap along the line
corrected_line = gap_line + corr_line

# Plot
plt.figure(figsize=(6, 4))
plt.plot(t_line, (gap_line - np.nanmin(gap_line))[::-1], label='Gap', marker='o', markersize=3)
plt.plot(t_line, (corr_line - np.nanmin(corr_line))[::-1], label='Strain Correction', marker='x', markersize=3)
plt.plot(t_line, (corrected_line - np.nanmin(corrected_line))[::-1], label='Corrected Gap', marker='.', markersize=3)
plt.xlabel("Shift along diagonal")
plt.ylabel("Energy [eV]")
plt.title("Gap and correction variation along the diagonal")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('plots/line_plot.png', dpi=500)
