import numpy as np
from ase.io import read
from scipy.interpolate import LinearNDInterpolator
from functions.util import repeate_cells

def laplacian(psi, dr):
    """
    2D hexagonal (triangular lattice) Laplacian
    using 6 nearest neighbours with zero padding.
    """
    Nx, Ny = psi.shape

    # pad by 1 cell on all sides
    p = np.pad(psi, 1, mode="constant", constant_values=0.0)

    center = p[1:-1, 1:-1]

    lap = (
            p[2:, 1:-1] +  # +x
            p[:-2, 1:-1] +  # -x
            p[1:-1, 2:] +  # +y
            p[1:-1, :-2] +  # -y
            p[2:, 2:] +  # +x +y
            p[:-2, :-2]  # -x -y
            - 6.0 * center
    )

    return (2.0 / 3.0) * lap / dr**2

# Parameters
L = 5
N_grid = 100
dr = L/N_grid
m = 1.15
omega = 1.5  # angular frequency (adjust as needed)
Nt = 1000
dt = 0.2*dr**2

# Convergence parameters
E_conv = 1e-6

dx = dr
dy = np.sqrt(3)/2 * dr
Nx = Ny = N_grid

# Grid indices
i_idx, j_idx = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
grid_indices = np.column_stack((i_idx.ravel(), j_idx.ravel()))

# Lattice vectors for hexagonal unit cell
v1 = np.array([dx, 0])
v2 = np.array([-dx/2, dy])

# Map indices to Cartesian positions
real_points = grid_indices @ np.vstack([v1, v2])  # Nx*Ny x 2


X = np.zeros((Nx, Ny))
Y = np.zeros((Nx, Ny))
"""
for j in range(Ny):
    for i in range(Nx):
        X[i, j] = i*dx + (j % 2)*dx/2
        Y[i, j] = j*dy

# Center of harmonic trap
x0 = np.mean(X)
y0 = np.mean(Y)
"""

x0 =  np.mean(real_points[:,0])
y0 = np.mean(real_points[:,1])
V_flat = 0.5 * m * omega**2 * ((real_points[:,0]-x0)**2 + (real_points[:,1]-y0)**2)

# reshape into grid for computations
V = V_flat.reshape(Nx, Ny)

# Harmonic potential: V = 1/2 m omega^2 ((x-x0)^2 + (y-y0)^2)
# V = 0.5 * m * omega**2 * ((X - x0)**2 + (Y - y0)**2)


# Initial guess
psi = np.cos(2*np.pi*X) + np.cos(2*np.pi*Y)
psi /= np.sqrt(np.sum(np.abs(psi)**2)*dr*dr)

old_E = 10000
i = 0

max_iter = 10000
# iterative loop
while True:
    lap = laplacian(psi, dr)
    Hpsi = (-(1/(2*m))*lap + V*psi)
    E = np.sum(np.conj(psi)*Hpsi)*dr**2

    new_psi = psi - dt*Hpsi

    avg_diff = np.sum(np.abs(new_psi - psi))*dr**2

    psi = new_psi / np.sqrt(np.sum(np.abs(new_psi)**2) * dr**2)
    # psi += (dt/(2*m))*lap - dt*V*psi
    # psi /= np.sqrt(np.sum(np.abs(psi)**2)*dr*dr)


    print(f'iter: {i} with energy {E*1000:.2f}')
    i += 1
    if i > max_iter:
        print("Max iterations reached")
        break
    if abs(E - old_E) < E_conv:
        break
    else:
        old_E = E


import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# Plot potential
plt.subplot(1,2,1)
plt.scatter(real_points[:,0], real_points[:,1],
            c=V_flat, s=50, cmap="viridis")
plt.gca().set_aspect("equal")
plt.title("Potential V(x,y)")
plt.xlabel("x [Å]")
plt.ylabel("y [Å]")
plt.colorbar(label="V [eV]")

# Plot wavefunction
plt.subplot(1,2,2)
plt.scatter(real_points[:,0], real_points[:,1],
            c=np.abs(psi.ravel())**2, s=50, cmap="viridis")
plt.gca().set_aspect("equal")
plt.title(r"$|\psi|^2$" + f" Eigenvalue: {old_E:.2f}")
plt.xlabel("x [Å]")
plt.ylabel("y [Å]")
plt.colorbar(label="Probability density")
plt.tight_layout()

plt.savefig("2D-harmonic-results.png", dpi=500)