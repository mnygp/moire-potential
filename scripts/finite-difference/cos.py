import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functions.finite_difference import diag_hamiltonian

x = np.linspace(0, 2 * np.pi, 75, endpoint=False)
x_L = np.linspace(0, 4 * np.pi, 150, endpoint=False)

for name, x_arr in zip(["single_cell", "2_by_2"], [x, x_L]):
    dr = x_arr[1] - x_arr[0]
    X, Y = np.meshgrid(x_arr, x_arr)

    cos = np.cos(X) + np.cos(Y)

    eigval, eigvec = diag_hamiltonian(cos, 10, dr, False, 2)
    print(eigval)

    fig, axes = plt.subplots(4, 2, figsize=(10, 18))
    fig.suptitle("2D Schrödinger Equation " + name, fontsize=14, fontweight="bold")

    # --- Top left: Potential ---
    ax = axes[0, 0]
    im = ax.pcolormesh(X, Y, cos, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title("Potential V(x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # --- 3 lowest eigenstates ---
    titles = [
        "Ground State |ψ|²",
        "1st Excited |ψ|²",
        "2nd Excited |ψ|²",
        "3rd Excited |ψ|²",
        "4th Excited |ψ|²",
        "5th Excited |ψ|²",
        "6th Excited |ψ|²",
    ]

    # Sort eigenvectors by eigenvalue (eigsh doesn't guarantee order)
    sort_idx = np.argsort(eigval)
    N = x_arr.shape[0]  # grid size

    for i, (ax, title) in enumerate(zip(axes.flat[1:], titles)):
        vec = eigvec[:, sort_idx[i]].reshape(N, N)
        prob = np.abs(vec) ** 2
        prob /= np.max(prob)
        im = ax.pcolormesh(X, Y, prob, cmap="viridis", shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{title}\nE = {eigval[sort_idx[i]]:.4f} eV")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("schrodinger_eigenstates_" + name + ".png", dpi=400)
    plt.close()
