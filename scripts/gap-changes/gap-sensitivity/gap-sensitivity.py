from ase.parallel import parprint
from ase.calculators.dftd3 import DFTD3
from gpaw import GPAW, PW
from functions.bandstructure import calc_gap
from functions.structure import create_bilayer
import numpy as np
import csv

z_arr = np.linspace(6, 7, 15)
strain_arr = np.linspace(0.99, 1.01, 14)

for shift in [0.0, 0.4]:
    data = []
    for z in z_arr:
        for strain in strain_arr:
            struct = create_bilayer(
                z,
                lattice_length=3.2515 * strain,
                a_shift=shift,
                b_shift=shift,
                constrain=True,
                acute_corner=True,
            )

            calc = GPAW(
                mode=PW(500),
                xc="PBE",
                kpts={"size": (12, 12, 1)},
                symmetry="off",
                txt="gpaw.txt",
            )
            d3_calc = DFTD3(dft=calc)
            struct.calc = d3_calc
            struct.get_potential_energy()

            gap = calc_gap(struct, kpts=36)
            data.append([shift, z, strain, gap])
            parprint(f"Shift: {shift:.2f}, z: {z:.2f}, strain: {strain:.4f} done")

    with open(
        f"gap_sensitivity_shift_{shift:.2f}.csv", mode="w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["shift", "z_dist", "strain", "gap"])
        writer.writerows(data)
    parprint(f"Shift {shift:.2f} data saved")
