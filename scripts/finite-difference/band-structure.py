import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from functions.finite_difference import diag_hamiltonian

# --- Grid ---
N = 75
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
dr = x[1] - x[0]
X, Y = np.meshgrid(x, x)
cos = np.cos(X) + np.cos(Y)

# --- K-path: Γ → X → M → Γ ---
N_k = 20
a = x[-1] - x[0]
kx_path = np.concatenate(
    [
        np.linspace(0, np.pi / a, N_k, endpoint=False),  # Γ → X
        np.linspace(np.pi / a, np.pi / a, N_k, endpoint=False),  # X → M
        np.linspace(np.pi / a, 0, N_k, endpoint=True),  # M → Γ
    ]
)
ky_path = np.concatenate(
    [
        np.linspace(0, 0, N_k, endpoint=False),  # Γ → X
        np.linspace(0, np.pi / a, N_k, endpoint=False),  # X → M
        np.linspace(np.pi / a, 0, N_k, endpoint=True),  # M → Γ
    ]
)

# --- Compute band structure ---
n_bands = 13
all_eigvals = np.zeros((len(kx_path), n_bands))

for i, (kx, ky) in enumerate(zip(kx_path, ky_path)):
    eigvals, _ = diag_hamiltonian(
        cos, 10, dr, False, 8, eigvals=n_bands + 6, kx=kx, ky=ky
    )
    all_eigvals[i] = eigvals[:n_bands]
    if True:  # i % 10 == 0:
        print(f"k-point {i + 1}/{len(kx_path)}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))

for band in range(n_bands):
    ax.scatter(range(len(kx_path)), all_eigvals[:, band], color="red", s=2)

# High-symmetry point markers
sym_ticks = [0, N_k, 2 * N_k, 3 * N_k - 1]
sym_labels = [r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"]
for t in sym_ticks:
    ax.axvline(t, color="black", lw=0.8, linestyle="--", alpha=0.6)
for t in sym_ticks:
    ax.axvline(t, color="black", lw=0.8, linestyle="--", alpha=0.6)
ax.set_xticks(sym_ticks, sym_labels, fontsize=13)

ax.set_xlim(0, 3 * N_k - 1)
ax.set_ylabel("Energy (eV)")
ax.set_title("Band structure — square lattice (Γ → X → M → Γ)")
plt.tight_layout()
plt.savefig("band_structure.png", dpi=350)
print("Saved band_structure.png")

k_idx, band_idx = np.unravel_index(np.argmin(all_eigvals), all_eigvals.shape)
gs_energy = all_eigvals[k_idx, band_idx]
gs_kx = kx_path[k_idx]
gs_ky = ky_path[k_idx]
print(f"Ground state energy is {gs_energy:.4f} eV at kx={gs_kx:.4f}, ky={gs_ky:.4f}")
print("Gamma point energies")
print(all_eigvals[0, :])

# ---- Check folding -----
N = 150
x = np.linspace(0, 4 * np.pi, N, endpoint=False)
dr = x[1] - x[0]
X, Y = np.meshgrid(x, x)
cos = np.cos(X) + np.cos(Y)
eigvals, _ = diag_hamiltonian(cos, 10, dr, False, 2, eigvals=n_bands + 6)
print(eigvals)

N = 75
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
dr = x[1] - x[0]
X, Y = np.meshgrid(x, x)
cos = np.cos(X) + np.cos(Y)
folded_eigvals = []
for kx, ky in [(0, 0), (np.pi / 2, 0), (0, np.pi / 2), (np.pi / 2, np.pi / 2)]:
    eigvals, _ = diag_hamiltonian(cos, 10, dr, False, 2, kx=kx, ky=ky)
    folded_eigvals.extend(eigvals[:6])
    print(f"k=({kx:.2f}, {ky:.2f}): {eigvals[:3]}")

print(np.sort(folded_eigvals)[:n_bands])
