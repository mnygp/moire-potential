import numpy as np
from ase.io import read
from scipy.interpolate import LinearNDInterpolator
from functions.util import repeate_cells

CONVERSION_FACTOR = 3.80998211  # hbar²/(m_e*Å²)

coefficients = {
    2: [1, -2, 1],
    4: [-1/12, 4/3, -5/2, 4/3, -1/12],
    6: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
    8: [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560],
}

def laplacian(psi, dr, order=2):
    total_deriv = np.zeros_like(psi)
    coeffs = coefficients[order]
    n_coeffs = len(coeffs)
    for i in range(n_coeffs):
        shift = i - n_coeffs // 2
        total_deriv += coeffs[i] * np.roll(psi, shift, axis=0)
        total_deriv += coeffs[i] * np.roll(psi, shift, axis=1)
        total_deriv += coeffs[i] * np.roll(np.roll(psi, shift, axis=0), shift, axis=1)
    return (2/3)/(dr**2)*total_deriv


# ################ Import part of the code ###########################
path = '../../structures/MoS2-WSe2-MatterSim/1.11_2946/structure_ml.json'
atoms = read(path)
v1 = atoms.cell[0, :2]
v2 = atoms.cell[1, :2]

data = np.genfromtxt('../full-calculation/results_fixed_cell_fixed_TM_PBE.csv',
                     dtype=float, skip_header=1, delimiter=',')

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

cell_2d = atoms.cell[:2,:2]

x_L, y_L, gap_L = repeate_cells(x, y, gap + corr, range(-1, 2),
                          atoms.cell[0, :2], atoms.cell[1, :2])

gap_interp = LinearNDInterpolator(list(zip(x_L, y_L)), gap_L)


############# The solving part of the code ##############
# Parameters
N_grid = 100
dr = np.linalg.norm(v1)/N_grid
m = 1.15
Nt = 1000
dt = 0.2*dr**2

# Convergence parameters
E_conv = 1e-6

# Initialize grid and initial guess
x_lin = np.linspace(0, 1, N_grid, endpoint=False)
y_lin = np.linspace(0, 1, N_grid, endpoint=False)
X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

# Generate potential grid
points = np.column_stack((X.ravel(), Y.ravel()))
real_points = points @ cell_2d
V_flat = gap_interp(real_points)
V = V_flat.reshape(N_grid, N_grid)
assert np.sum(np.isnan(V)) == 0, 'Potential contains NaN values'

# V[:] = 0

# Initial guess
psi = np.cos(2*np.pi*X) + np.cos(2*np.pi*Y)

psi = np.ones_like(X)

psi /= np.sqrt(np.sum(np.abs(psi)**2)*dr*dr)

old_E = 10000
i = 0

max_iter = 10000
# iterative loop
while True:
    lap = laplacian(psi, dr, order=8)
    Hpsi = (-(CONVERSION_FACTOR/(2*m))*lap + V*psi)
    E = np.sum(np.conj(psi)*Hpsi)*dr**2

    new_psi = psi - dt*Hpsi

    # avg_diff = np.sum(np.abs(new_psi - psi))*dr**2

    psi = new_psi / np.sqrt(np.sum(np.abs(new_psi)**2) * dr**2)
    # psi += (dt/(2*m))*lap - dt*V*psi
    # psi /= np.sqrt(np.sum(np.abs(psi)**2)*dr*dr)

    if i%25 == 0:
        print(f'iter: {i} with energy {E*1000:.2f}')
    i += 1 
    if i > max_iter:
        print("Max iterations reached")
        break
    if abs(E - old_E) < E_conv:
        print(f'iter: {i} with energy {E * 1000:.2f}')
        break
    else:
        old_E = E



import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# ---- Left plot: potential ----
im0 = axes[0].scatter(
    real_points[:, 0],
    real_points[:, 1],
    c=V,
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
    c=np.abs(psi)**2,
    s=15,
    cmap="viridis",
)

axes[1].set_title(r"$|\psi|^2$")
axes[1].set_xlabel("x [Å]")
axes[1].set_ylabel("y [Å]")
fig.colorbar(im1, ax=axes[1], label="Amplitude")

for ax in axes:
    ax.set_aspect("equal", adjustable="box")


plt.savefig("2D-schrodinger-results-V0.png", dpi=500)
