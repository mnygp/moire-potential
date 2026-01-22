import matplotlib.pyplot as plt
import numpy as np
from ase.dft.kpoints import bandpath
from ase.io.trajectory import Trajectory
from ase.parallel import parprint
from gpaw.new.ase_interface import GPAW
from gpaw.spinorbit import soc_eigenstates

from functions.util import generate_scissor_shifts

a = 3.2515
data = np.loadtxt(
    f"../diagonal-shift/gap_shift_{a:.2f}_lcao.csv", delimiter=",", skiprows=1
)
shift_arr = data[:, 0]
z_dist_arr = data[:, 1]
gap_arr = data[:, 2]
gap_arr_soc = data[:, 3]
gap_scissors_arr = data[:, 4]
gap_scissors_arr_soc = data[:, 5]


for i in range(14, 20):
    struct = Trajectory(
        f"../diagonal-shift/traj_files/lcao_opt_{a:.2f}_{shift_arr[i]:.2f}_high_fid.traj"
    )[-1]
    shifts = generate_scissor_shifts(struct)
    scissors = {"name": "scissors", "shifts": shifts}
    gpw_name = f"shift_{shift_arr[i]:.2f}.gpw"

    parprint(f"Beginning shift: {shift_arr[i]:.2f}")

    calc_load = GPAW(gpw_name).fixed_density(
        symmetry="off", kpts={"path": "GMKG", "npoints": 200}
    )
    ef = GPAW(gpw_name, txt=None).get_fermi_level()

    path = bandpath("GMKG", calc_load.atoms.cell, npoints=200)
    (x, X, labels) = path.get_linear_kpoint_axis()

    e_kn = np.array(
        [
            calc_load.get_eigenvalues(kpt=k)[:]
            for k in range(len(calc_load.get_ibz_k_points()))
        ]
    )
    e_nk = e_kn.T
    # e_nk -= ef

    # VBM normalisation
    e_kn = np.array(
        [
            calc_load.get_eigenvalues(kpt=k)
            for k in range(len(calc_load.get_ibz_k_points()))
        ]
    )
    f_kn = np.array(
        [
            calc_load.get_occupation_numbers(kpt=k)
            for k in range(len(calc_load.get_ibz_k_points()))
        ]
    )
    vbm = np.max(e_kn[f_kn > 1e-5])

    e_nk -= vbm

    for e_k in e_nk:
        plt.plot(x, e_k, "--", c="0.5")

    soc = soc_eigenstates(calc_load)
    e_mk = soc.eigenvalues().T
    # e_mk -= soc.fermi_level

    soc_occ = soc.occupation_numbers().T  # shape matches e_mk
    vbm_soc = np.max(e_km[soc_occ > 1e-5])
    # Normalize SOC bands to SOC VBM
    e_mk -= vbm_soc

    plt.xticks(X, [r"$\Gamma$", "M", "K", r"$\Gamma$"], size=20)
    plt.yticks(size=20)
    for j in range(len(X))[1:-1]:
        plt.plot(2 * [X[j]], [-1.5, 2.5], c="0.5", linewidth=0.5)

    for e_k in e_mk[::2]:
        plt.plot(x, e_k, c="b", lw=2)
    plt.plot([0.0, x[-1]], 2 * [0.0], c="0.5")

    plt.ylabel(r"$\varepsilon_n(k)$ [eV]", size=24)
    plt.axis([0, x[-1], -1, 2])
    plt.tight_layout()
    plt.savefig(f"bandstructure_shift_{shift_arr[i]:.2f}_SOC.png")
    plt.close()
    parprint("SOC done")
