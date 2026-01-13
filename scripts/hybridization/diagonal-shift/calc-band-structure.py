from gpaw import GPAW, PW
from ase.io import Trajectory
import numpy as np


for a in [3.184]:  # np.linspace(3.184, 3.319, 7):
    shift_arr = np.linspace(0, 1, 30)
    shift_arr = [0.03]
    for shift in shift_arr:
        traj_file = f"traj_files/opt_{a:.2f}_{shift:.2f}.traj"
        last_image = Trajectory(traj_file)[-1]  # take last frame

        calc = GPAW(
            mode=PW(500),
            xc="PBE",
            kpts={"size": (12, 12, 1)},
            symmetry="off",
            txt="gpaw.txt",
        )
        last_image.calc = calc

        last_image.get_potential_energy()

        ef = calc.get_fermi_level()

        calc = calc.fixed_density(
            nbands=100,
            symmetry="off",
            kpts={"path": "GKMG", "npoints": 200},
            convergence={"bands": 30},
        )

        bs = calc.band_structure()
        string = f"band_gap_files/band_structure_{a:.2f}_{shift:.2f}.png"
        bs.plot(string, emin=ef - 4, emax=ef + 4)
