import numpy as np
import csv
from gpaw import GPAW, PW
from ase.optimize import BFGS
from ase.calculators.dftd3 import DFTD3

from functions.structure import create_bilayer

# Parameters
shifts = np.linspace(0, 1, 6, endpoint=False)
a_values = np.linspace(3.184 * 0.995, 3.319 * 1.005, 30)

# Data containers
z_matrix = []  # each row: [a, z1, z2, z3, z4, z5, z6]


for shift in shifts:
    row = [shift]  # start with the shift value
    for a in a_values:
        struct = create_bilayer(
            z_dist=6.6,
            lattice_length=a,
            a_shift=shift,
            b_shift=shift,
            constrain=True,
            acute_corner=True,
        )

        calc = GPAW(mode=PW(500), xc="PBE", kpts={"size": (12, 12, 1)}, txt="gpaw.txt")
        d3_calc = DFTD3(dft=calc)
        struct.calc = d3_calc
        struct.get_potential_energy()

        opt = BFGS(struct, trajectory=f"traj_files/opt_{a:.2f}_{shift:.2f}.traj")
        opt.run(fmax=0.01)
        symbols = np.array(struct.get_chemical_symbols())
        z_dist = (
            struct[symbols == "W"].positions[0][2]
            - struct[symbols == "Mo"].positions[0][2]
        )

        row.append(z_dist)

    z_matrix.append(row)

with open("relaxed_z_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["a"] + [f"shift={s:.2f}" for s in shifts]
    writer.writerow(header)
    writer.writerows(z_matrix)
