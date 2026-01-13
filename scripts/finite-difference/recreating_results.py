import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, e
from functions.finite_difference import diag_hamiltonian

def V(x, y, M, omega, delta, phase):
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    V = M * omega**2 / 2 * R**2 * (1 - delta * np.cos(3 * phi + phase))
    CONVERSION_FACTOR = 7.61996422  # hbar²/(m_e*eV*Å²)
    return V * CONVERSION_FACTOR

N_grid = 125
# MoS2 electron effective mass 0.7
# WSe2 hole effective mass 0.45
m = 1.15  # in electron masses
omega = 0.05483  # in eV
delta = 0.33

x_lin = np.linspace(-40, 40, N_grid)
y_lin = np.linspace(-40, 40, N_grid)
X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

dr = x_lin[1] - x_lin[0]

pot = V(X, Y, m, omega, delta, np.pi / 2)

eigvals, eigvecs = diag_hamiltonian(pot, m, dr, False, 4)

print(f"R0 is {np.sqrt(hbar * hbar / (omega * e * m * m_e))*1e10}")
print(eigvals)

for i in range(6):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    psi = eigvecs[:, i] / max(eigvecs[:, i])

    X_flat = X.flatten()
    Y_flat = Y.flatten()

    psi_real = np.real(psi)
    psi_imag = np.imag(psi)

    # ---- First plot: potential ----
    im0 = ax[0].scatter(
        X_flat,
        Y_flat,
        c=pot.flatten() * 1000,
        s=15,
        cmap="viridis",
    )

    ax[0].set_title("Potential")
    ax[0].set_xlabel("x [Å]")
    ax[0].set_ylabel("y [Å]")
    fig.colorbar(im0, ax=ax[0], label="Potential [meV")

    # ---- Second plot: Re(psi) ----
    im1 = ax[1].scatter(
        X_flat,
        Y_flat,
        c=psi,
        s=15,
        cmap="viridis",
    )

    ax[1].set_title(r"$\psi$" + f" eigval={eigvals[i] * 1000:.2f}")
    ax[1].set_xlabel("x [Å]")
    ax[1].set_ylabel("y [Å]")
    fig.colorbar(im1, ax=ax[1], label="Normalised Amplitude")

    for ax in ax:
        ax.set_aspect("equal", adjustable="box")

    plt.savefig(f"2D-fitted_potential_{i}.png", dpi=500)
    plt.close()
