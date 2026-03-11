import numpy as np
import json
from functions.finite_difference import diag_hamiltonian
from ase.io import read
from functions.util import repeate_cells
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt


def decode_ndarray(obj):
    if "__ndarray__" in obj:
        shape, dtype, data = obj["__ndarray__"]
        return np.array(data, dtype=dtype).reshape(shape)
    return obj


struct_list = ['0.00_4875', '1.06_3921', '1.84_2595',
               '2.98_1500', '3.96_978', '4.72_696']

Ns = range(20, 120, 10)
all_ground_states = np.zeros((len(struct_list), len(Ns)))
dr_list = np.zeros((len(struct_list), len(Ns)))

for name in struct_list:
    print(f'Starting {name}')
    gap_dict = json.load(open('../../multi-angle-calculation/calculations/'
                              f'tree/gen_wfs/{name}/'
                              'corrected_opt_gap/output.json'))
    x = decode_ndarray(gap_dict['x'])
    y = decode_ndarray(gap_dict['y'])
    gap = decode_ndarray(gap_dict['corr gap'])

    atoms = read(f'../../../structures/more-structures/{name}'
                 '/MatterSim_relaxed_extra_high_fid.json')

    cell_2d = atoms.cell[:2, :2]  # * (87/np.linalg.norm(v1))
    v1 = atoms.cell[0, :2]

    x_L, y_L, gap_L = repeate_cells(
        x, y, gap, range(-1, 2), atoms.cell[0, :2], atoms.cell[1, :2]
    )
    gap_L -= min(gap_L)
    gap_interp = LinearNDInterpolator(list(zip(x_L, y_L)), gap_L)

    print(f"Gap minimum=0 and maximum={max(gap_L * 1000):.2f}")

    eigvals_array = []

    for i, N in enumerate(Ns):
        dr = np.linalg.norm(v1) / N
        dr_list[struct_list.index(name), i] = dr
        m = 1.15

        # Initialize grid and initial guess
        x_lin = np.linspace(0, 1, N, endpoint=False)
        y_lin = np.linspace(0, 1, N, endpoint=False)
        X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

        # Generate potential grid
        points = np.column_stack((X.ravel(), Y.ravel()))
        real_points = points @ cell_2d
        # real_points = points @ ([[1, 0], [0, 1]])
        V_flat = gap_interp(real_points)
        V = V_flat.reshape((N, N))
        eigvals, eigvecs = diag_hamiltonian(V, m, dr, hexagonal=True, order=2)
        eigvals_array.append(list(eigvals[:4]))
        print(f'Finished N={N}')

    eigvals_array = np.array(eigvals_array)
    all_ground_states[struct_list.index(name), :] = eigvals_array[:, 0]
    Ns = np.asarray(Ns)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(7, 6),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # ---- top: eigenvalues vs grid size ----
    for i in range(4):
        ax_top.plot(Ns, eigvals_array[:, i]*1000, marker='o',
                    label=f"$\\lambda_{i + 1}$")

    ax_top.set_ylabel("Eigenvalue [meV]")
    ax_top.legend()
    ax_top.grid(True)

    # ---- bottom: successive differences ----
    dNs = Ns[1:]

    for i in range(4):
        delta = eigvals_array[1:, i] - eigvals_array[:-1, i]
        ax_bot.plot(dNs, delta*1000, marker='o',
                    label=f"$\\Delta \\lambda_{i + 1}$")

    ax_bot.set_xlabel("Grid size N")
    ax_bot.set_ylabel("Δ Eigenvalue [meV]")
    ax_bot.axhline(0, linestyle="--")
    ax_bot.grid(True)

    plt.tight_layout()
    plt.savefig(f'{name}_function_of_N.png')
    plt.close()

    # ---------------------------------------------
    drs = np.linalg.norm(v1) / Ns
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(7, 6),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # ---- top: eigenvalues vs grid size ----
    for i in range(4):
        ax_top.plot(drs, eigvals_array[:, i]*1000, marker='o',
                    label=f"$\\lambda_{i + 1}$")

    ax_top.set_ylabel("Eigenvalue [meV]")
    ax_top.legend()
    ax_top.grid(True)

    # ---- bottom: successive differences ----
    for i in range(4):
        delta = eigvals_array[1:, i] - eigvals_array[:-1, i]
        ax_bot.plot(drs[1:], delta*1000, marker='o',
                    label=f"$\\Delta \\lambda_{i + 1}$")
    ax_bot.set_xlabel("Grid spacing dr [Å]")
    ax_bot.set_ylabel("Δ Eigenvalue [meV]")
    ax_bot.axhline(0, linestyle="--")
    ax_bot.grid(True)

    plt.tight_layout()
    plt.savefig(f'{name}_function_of_dr.png')
    plt.close()

# ---------- Only ground state comparison between structures ----------

s = len(struct_list)
fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(7, 6),
        gridspec_kw={"height_ratios": [2, 1]}
    )

# ---- top: eigenvalues vs grid size ----
for i in range(s):
    ax_top.plot(Ns, all_ground_states[i, :]*1000, marker='o',
                label=f"{struct_list[i]}")

ax_top.set_ylabel("Eigenvalue [meV]")
ax_top.legend()
ax_top.grid(True)

# ---- bottom: successive differences ----
dNs = Ns[1:]

for i in range(s):
    delta = all_ground_states[i, 1:] - all_ground_states[i, :-1]
    ax_bot.plot(dNs, delta*1000, marker='o',
                label=f"$\\Delta$ for {struct_list[i]}")

ax_bot.set_xlabel("Grid size N")
ax_bot.set_ylabel("Δ Eigenvalue [meV]")
ax_bot.axhline(0, linestyle="--")
ax_bot.grid(True)

plt.tight_layout()
plt.savefig('ground_states_function_of_N.png')
plt.close()

# ---------------------------------------------
drs = np.linalg.norm(v1) / Ns
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(7, 6),
    gridspec_kw={"height_ratios": [2, 1]}
)

# ---- top: eigenvalues vs grid size ----
for i in range(s):
    ax_top.plot(dr_list[i, :], all_ground_states[i, :]*1000, marker='o',
                label=f"{struct_list[i]}")

ax_top.set_ylabel("Eigenvalue [meV]")
ax_top.legend()
ax_top.grid(True)

# ---- bottom: successive differences ----
for i in range(s):
    delta = all_ground_states[i, 1:] - all_ground_states[i, :-1]
    ax_bot.plot(dr_list[i, 1:], delta*1000, marker='o',
                label=f"$\\Delta$ for {struct_list[i]}")
ax_bot.set_xlabel("Grid spacing dr [Å]")
ax_bot.set_ylabel("Δ Eigenvalue [meV]")
ax_bot.axhline(0, linestyle="--")
ax_bot.grid(True)

plt.tight_layout()
plt.savefig('ground_states_function_of_dr.png')
plt.close()
