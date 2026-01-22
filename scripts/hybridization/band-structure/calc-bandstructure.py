import matplotlib.pyplot as plt
import numpy as np
from ase.dft.kpoints import bandpath
from ase.io.trajectory import Trajectory
from ase.parallel import parprint
from ase.spectrum.band_structure import BandStructure
from gpaw import FermiDirac
from gpaw.new.ase_interface import GPAW
from gpaw.new.extensions import D3
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

    parprint(f"Beginning shift: {shift_arr[i]:.2f}")
    # No SOC
    calc = GPAW(
        mode="lcao",
        basis="dzp",
        xc="PBE",
        kpts={"size": (12, 12, 1)},
        symmetry="off",
        txt="gpaw.txt",
        eigensolver=scissors,
        occupations=FermiDirac(0.01),
        extensions=[D3(xc="PBE")],
    )
    struct.calc = calc
    struct.get_potential_energy()
    ef = struct.calc.get_fermi_level()

    gpw_name = f"shift_{shift_arr[i]:.2f}.gpw"
    calc.write(gpw_name, mode="all")

    calc = GPAW(gpw_name).fixed_density(
        symmetry="off", kpts={"path": "GMKG", "npoints": 200}
    )

    # VBM normalisation
    e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
    f_kn = np.array([calc.get_occupation_numbers(kpt=k) for k in range(len(calc.get_ibz_k_points()))])

    vbm = np.max(e_kn[f_kn > 1e-5])

    bs = calc.band_structure()
    # bs = bs.subtract_reference()

    bs_shift = BandStructure(bs.path, bs.energies, reference=vbm)
    bs_shift.subtract_reference()
    bs_shift.plot(filename=f"bandstructure_shift_{shift_arr[i]:.2f}.png", emax=2, emin=-1)
