import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator
from ase.io import read
from functions.util import repeate_cells

# lengths are in Å
# MoS2 electron effective mass: m_e/m=0.7
# WSe2 hole effective mass: m_h/m=0.45
# From https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.195403

m = 1.15  # COM mass in units of electron mass


def harmonic_osc(xy, w, c_x, c_y, off_set):
    x = xy[:, 0] - c_x
    y = xy[:, 1] - c_y
    return 0.5 * m * w**2 * (x**2 + y**2) + off_set


def get_points(x, y, r, c_x, c_y, data=None) -> tuple:
    dist2 = (x - c_x) ** 2 + (y - c_y) ** 2
    mask = dist2 <= r**2

    if data is None:
        return x[mask], y[mask]
    else:
        return x[mask], y[mask], data[mask]


# ################### Read the data and atoms #########################
data = np.genfromtxt(
    "../full-calculation/results_fixed_cell_fixed_TM_PBE.csv",
    dtype=float,
    skip_header=1,
    delimiter=",",
)
struc_path = "../../structures/MoS2-WSe2-MatterSim/1.11_2946/structure_ml.json"
atoms = read(struc_path)
v1 = atoms.cell[0, :2]
v2 = atoms.cell[1, :2]
print("Vectors:")
print(v1, v2)

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


# Start of the line at 2*v2
line_start = 2 * v2
line_end = 2 * v1  # end of the diagonal line across two moire cells
line_vec = line_end - line_start

# Place the first minimum at the halfway point along the diagonal
t1 = 0.5
gap_min_x_1, gap_min_y_1 = line_start + t1 * line_vec

# Place the second minimum 1/3 of the diagonal further along the line
t2 = t1 + 1 / 6  # 1/3 further from first minimum
gap_min_x_2, gap_min_y_2 = line_start + t2 * line_vec


_, _, gap_L = repeate_cells(x, y, gap, range(-1, 2), v1, v2)
x_L, y_L, corr_L = repeate_cells(x, y, corr, range(-1, 2), v1, v2)

# Interpolators
gap_interp = LinearNDInterpolator(list(zip(x_L, y_L)), gap_L)
corr_interp = LinearNDInterpolator(list(zip(x_L, y_L)), corr_L)


x_near_1, y_near_1, gap_near_1 = get_points(
    x_L, y_L, 10, gap_min_x_1, gap_min_y_1, data=gap_L
)
x_near_2, y_near_2, gap_near_2 = get_points(
    x_L, y_L, 10, gap_min_x_2, gap_min_y_2, data=gap_L
)

minima_lim = 10

popt_1, _ = curve_fit(
    harmonic_osc,
    np.column_stack((x_near_1, y_near_1)),
    gap_near_1,
    p0=[0.1, gap_min_x_1, gap_min_y_1, np.nanmin(gap)],
    bounds=(
        [0, gap_min_x_1 - minima_lim, gap_min_y_1 - minima_lim, -np.inf],
        [np.inf, gap_min_x_1 + minima_lim, gap_min_y_1 + minima_lim, np.inf],
    ),
)

popt_2, _ = curve_fit(
    harmonic_osc,
    np.column_stack((x_near_2, y_near_2)),
    gap_near_2,
    p0=[0.1, gap_min_x_2, gap_min_y_2, np.nanmin(gap)],
    bounds=(
        [0, gap_min_x_2 - minima_lim, gap_min_y_2 - minima_lim, -np.inf],
        [np.inf, gap_min_x_2 + minima_lim, gap_min_y_2 + minima_lim, np.inf],
    ),
)
print("Data range:")
print(max(gap_near_1), min(gap_near_1))
print(max(gap_near_2), min(gap_near_2))
print("\n")


print(f"Fit 1 guess: {gap_min_x_1 / 72:.2f}, {gap_min_y_1 / 72:.2f}")
print(f"Fit 2 guess: {gap_min_x_2 / 72:.2f}, {gap_min_y_2 / 72:.2f}")
print("Fit center 1 in moire coords:")
print(popt_1[1:3] / 72)
print("Fit center 2 in moire coords:")
print(popt_2[1:3] / 72)

line_points = np.linspace(2 * v2, 2 * v1, 400)
print("The line goes from")
print(2 * v2 / 72)
print("to")
print(2 * v1 / 72)

print("Frequencies:")
print(f"Frequency of fit 1: {popt_1[0]}")
print(f"Frequency of fit 2: {popt_2[0]}")

print("############## Non-corrected gap plot ################")
diag_len = np.linalg.norm(abs(v1) + abs(v2))

x_line = line_points[:, 0]
y_line = line_points[:, 1]
gap_line = gap_interp(x_line, y_line) * 1000
print(np.nanmin(gap_line), np.nanmax(gap_line))

fit_line_1 = harmonic_osc(line_points, *popt_1) * 1000
fit_line_2 = harmonic_osc(line_points, *popt_2) * 1000

x_axis = np.linspace(0, diag_len, len(fit_line_1[132:330]))
plt.plot(x_axis, gap_line[132:330], label="Gap data")
plt.plot(x_axis, fit_line_1[132:330], label="Fit 1")
plt.plot(x_axis, fit_line_2[132:330], label="Fit 2")
plt.legend()
plt.ylim(np.nanmin(gap * 1000), np.nanmax(gap * 1000))
plt.title("Non corrected gap and harmonic fits")
plt.xlabel("Diagonal length [Å]")
plt.ylabel("Band gap [meV]")
plt.tight_layout()
plt.grid()
plt.savefig("non-corrected-fits.png", dpi=500)

plt.hlines(
    min(gap_line[132:330]) + 44.9, 15, 70, colors="C1", linestyles="dashed", alpha=0.75
)
plt.hlines(
    min(gap_line[132:330]) + 22.7, 55, 110, colors="C2", linestyles="dashed", alpha=0.75
)
plt.savefig("non-corrected-fits-with-eig.png", dpi=500)

plt.close()

print("############## Corrected gap plot ################")
gap_c = gap + corr
gap_c_min_x = x[np.argmin(gap_c)]
gap_c_min_y = y[np.argmin(gap_c)]
x_near, y_near, gap_c_near = get_points(x, y, 10, gap_c_min_x, gap_c_min_y, data=gap_c)
popt_c, _ = curve_fit(
    harmonic_osc,
    np.column_stack((x_near, y_near)),
    gap_c_near,
    p0=[0.1, gap_c_min_x, gap_c_min_y, np.nanmin(gap_c)],
    bounds=(
        [0, gap_c_min_x - minima_lim, gap_c_min_y - minima_lim, -np.inf],
        [np.inf, gap_c_min_x + minima_lim, gap_c_min_y + minima_lim, np.inf],
    ),
)

s_line_points = np.linspace(v2, v1, 400)
s_x_line = s_line_points[:, 0]
s_y_line = s_line_points[:, 1]
gap_c_line = (gap_interp(s_x_line, s_y_line) + corr_interp(s_x_line, s_y_line)) * 1000
fit_line = harmonic_osc(s_line_points, *popt_c) * 1000

print(f"Corrected frequency: {popt_c[0]}")

# Wrap array
gap_c_line_w = np.concatenate((gap_c_line[265:], gap_c_line[:265]))
fit_line_w = np.concatenate((fit_line[265:], fit_line[:265]))
x_axis = np.linspace(0, diag_len, len(gap_c_line_w))

plt.plot(x_axis, gap_c_line_w, label="Corrected data")
plt.plot(x_axis, fit_line_w, label="Fit")
plt.legend()
plt.ylim(np.nanmin((gap + corr) * 1000), np.nanmax(gap + corr) * 1000)
plt.title("Corrected gap and harmonic fits")
plt.xlabel("Diagonal length [Å]")
plt.ylabel("Band gap [meV]")
plt.tight_layout()
plt.grid()
plt.savefig("corrected-fits.png", dpi=500)

plt.hlines(
    min(gap_c_line_w) + 37.5, 50, 110, colors="C1", linestyles="dashed", alpha=0.75
)
plt.savefig("corrected-fits-with eig.png", dpi=500)
plt.close()
