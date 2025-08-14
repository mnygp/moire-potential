import numpy as np
from functions.structure import create_bilayer
from gpaw import GPAW, PW
from ase.calculators.dftd3 import DFTD3
from ase.filters import UnitCellFilter
from ase.optimize.bfgs import BFGS
import csv

average_lattice = 3.2515
average_cell = np.array([[1, 0], [-0.5, np.sqrt(3) / 2]]) * average_lattice

# Make sure the CSV file has a header
with open("results.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["i", "j", "v1_norm", "v2_norm", "z_dist"])


for i in np.linspace(0, 1, 15):
    for j in np.linspace(0, 1, 15):
        bilayer = create_bilayer(z_dist=6.6,  constrain=True,
                                 a_shift=i, b_shift=j)

        bilayer.calc = DFTD3(dft=GPAW(mode=PW(500),
                                      kpts=(6, 6, 1),
                                      xc='PBE',
                                      txt='bilayer.txt'))

        uf = UnitCellFilter(bilayer, mask=[1, 1, 0, 0, 0, 1])
        relax = BFGS(uf, trajectory=f'traj_files/opt_{i:.2f}_{j:.2f}.traj')
        relax.run(fmax=0.02)

        v1_norm = np.linalg.norm(bilayer.cell[0])
        v2_norm = np.linalg.norm(bilayer.cell[1])

        symbols = np.array(bilayer.get_chemical_symbols())
        Mo_z = bilayer.positions[symbols == 'Mo'][:, 2]
        W_z = bilayer.positions[symbols == 'W'][:, 2]

        z_dist = np.abs(W_z - Mo_z)[0]

        # TODO: Add mpi rank=0 check
        # Append results to CSV
        with open("results.csv", mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, j, v1_norm, v2_norm, z_dist])
