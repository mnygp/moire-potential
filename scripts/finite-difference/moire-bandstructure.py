import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from scipy.interpolate import LinearNDInterpolator

from functions.finite_difference import diag_hamiltonian
from functions.util import repeate_cells


def get_bz_kpath(a=1.0, n_points=100):
    """
    BZ k-path for a 60° rhomboid supercell with no internal symmetry.
    K and K' are inequivalent, so the full path Γ→M→K→Γ→M'→K'→Γ is needed.
    """
    b1 = (2 * np.pi / a) * np.array([1.0, -1.0 / np.sqrt(3)])
    b2 = (2 * np.pi / a) * np.array([0.0, 2.0 / np.sqrt(3)])

    Gamma = np.array([0.0, 0.0])
    M = b1 / 2
    Mp = b2 / 2
    K = (2 * b1 + b2) / 3
    Kp = (b1 + 2 * b2) / 3

    path_def = [
        ("Γ", Gamma),
        ("M", M),
        ("K", K),
        ("Γ", Gamma),
        ("M'", Mp),
        ("K'", Kp),
        ("Γ", Gamma),
    ]

    path_k, path_d, ticks = [], [], [0.0]
    d_total = 0.0

    for seg in range(len(path_def) - 1):
        label0, k0 = path_def[seg]
        label1, k1 = path_def[seg + 1]
        seg_pts = np.linspace(k0, k1, n_points, endpoint=False)
        seg_len = np.linalg.norm(k1 - k0)
        seg_d = np.linspace(0, seg_len, n_points, endpoint=False)
        seg_len = np.linalg.norm(k1 - k0)
        seg_d = np.linspace(0, seg_len, n_points, endpoint=False)
        path_k.append(seg_pts)
        path_d.append(d_total + seg_d)
        d_total += seg_len
        ticks.append(d_total)

    path_k.append(path_def[-1][1][np.newaxis])
    path_d.append([d_total])

    tick_labels = [p[0] for p in path_def]

    return (
        np.concatenate(path_d),
        np.vstack(path_k),
        ticks,
        tick_labels,
    )


path = "../../structures/MoS2-WSe2-MatterSim/1.11_2946/structure_ml.json"
atoms = read(path)
v1 = atoms.cell[0, :2]
v2 = atoms.cell[1, :2]

data = np.genfromtxt(
    "../full-calculation/results_fixed_cell_fixed_TM_PBE.csv",
    dtype=float,
    skip_header=1,
    delimiter=",",
)

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

cell_2d = atoms.cell[:2, :2]  # * (87/np.linalg.norm(v1))

x_L, y_L, gap_L = repeate_cells(
    x, y, gap + corr, range(-1, 2), atoms.cell[0, :2], atoms.cell[1, :2]
)
gap_L -= min(gap_L)
gap_interp = LinearNDInterpolator(list(zip(x_L, y_L)), gap_L)
print(f"Gap minimum=0 and maximum={max(gap_L * 1000):.2f}")

# Parameters
N_grid = 100
dr = np.linalg.norm(v1) / N_grid
m = 1.15

# Initialize grid and initial guess
x_lin = np.linspace(0, 1, N_grid, endpoint=False)
y_lin = np.linspace(0, 1, N_grid, endpoint=False)
X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

# Generate potential grid
points = np.column_stack((X.ravel(), Y.ravel()))
real_points = points @ cell_2d
V_grid = gap_interp(real_points).reshape(N_grid, N_grid)

d_plot, kpts, ticks, labels = get_bz_kpath(a=np.linalg.norm(v1), n_points=15)

# --- Compute band structure ---
n_bands = 13
kx_path = kpts[:, 0]
ky_path = kpts[:, 1]

all_eigvals = np.zeros((len(kx_path), n_bands))
for i, (kx, ky) in enumerate(zip(kx_path, ky_path)):
    eigvals, _ = diag_hamiltonian(
        V_grid, m, dr, True, 8, eigvals=n_bands + 6, kx=kx, ky=ky
    )
    all_eigvals[i] = eigvals[:n_bands]
    if True:  # i % 10 == 0:
        print(f"k-point {i + 1}/{len(kx_path)}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))
for band in range(n_bands):
    ax.scatter(d_plot, all_eigvals[:, band], color="red", s=2)

for t in ticks:
    ax.axvline(t, color="black", lw=0.8, linestyle="--", alpha=0.6)

ax.set_xticks(ticks, labels, fontsize=13)
ax.set_xlim(d_plot[0], d_plot[-1])
ax.set_ylabel("Energy (eV)")
ax.set_title("Band structure — moiré supercell (Γ→M→K→Γ→M'→K'→Γ)")
plt.tight_layout()
plt.savefig("band_structure_moire.png", dpi=350)
print("Saved band_structure.png")

k_idx, band_idx = np.unravel_index(np.argmin(all_eigvals), all_eigvals.shape)
gs_energy = all_eigvals[k_idx, band_idx]
gs_kx = kx_path[k_idx]
gs_ky = ky_path[k_idx]
print(f"Ground state energy is {gs_energy:.4f} eV at kx={gs_kx:.4f}, ky={gs_ky:.4f}")
print("Gamma point energies:")
print(all_eigvals[0, :])
