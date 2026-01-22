import numpy as np
from ase.io import read
from scipy.interpolate import LinearNDInterpolator
from functions.util import repeate_cells
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


CONVERSION_FACTOR = 3.80998211  # hbar²/(m_e*Å²)

coefficients = {
    2: [1, -2, 1],
    4: [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12],
    6: [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
    8: [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
}


def laplacian(N, dr, order=2):
    lap = lil_matrix((N * N, N * N))

    coeffs = coefficients[order]
    N_c = len(coeffs)
    shifts = range(-(N_c // 2), N_c // 2 + 1)

    for i in range(N):
        for j in range(N):
            for coeff, shift in zip(coeffs, shifts):
                new_j = (j + shift) % N
                new_i = (i + shift) % N

                lap[j + i * N, new_j + i * N] += coeff  # y direction
                lap[j + i * N, j + new_i * N] += coeff  # x direction
                lap[j + i * N, new_j + new_i * N] += coeff  # xy direction

    return 2 / 3 * lap / dr**2


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

############# The solving part of the code ##############
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
# real_points = points @ ([[1, 0], [0, 1]])
V_flat = gap_interp(real_points)
assert np.sum(np.isnan(V_flat)) == 0, "Potential contains NaN values"


L = laplacian(N_grid, dr, order=2)

# Kinetic prefactor
H = -CONVERSION_FACTOR / (2 * m) * L
# H = -1/(2*m)* L

# Add potential on the diagonal
H.setdiag(H.diagonal() + V_flat)

eigvals, eigvecs = eigsh(H, k=10, which="SM")
np.set_printoptions(linewidth=200, precision=3, suppress=True)
print(eigvals * 1000)


for i in range(8):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    psi = eigvecs[:, i].reshape((N_grid, N_grid))

    # ---- Left plot: potential ----
    im0 = axes[0].scatter(
        real_points[:, 0],
        real_points[:, 1],
        c=V_flat.reshape((N_grid, N_grid)),
        s=15,
        cmap="viridis",
    )

    axes[0].set_title("Corrected local band gap")
    axes[0].set_xlabel("x [Å]")
    axes[0].set_ylabel("y [Å]")
    fig.colorbar(im0, ax=axes[0], label="Potential")

    # ---- Right plot: |psi|^2 ----
    im1 = axes[1].scatter(
        real_points[:, 0],
        real_points[:, 1],
        c=psi * np.conj(psi),
        s=15,
        cmap="viridis",
    )

    axes[1].set_title(r"$|\psi|^2$" + f" eigval={eigvals[i] * 1000:.2f}")
    axes[1].set_xlabel("x [Å]")
    axes[1].set_ylabel("y [Å]")
    fig.colorbar(im1, ax=axes[1], label="Amplitude")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")

    plt.savefig(f"2D-schrodinger-results-diag_{i}.png", dpi=500)
    plt.close()
