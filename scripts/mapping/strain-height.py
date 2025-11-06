import numpy as np
from functions.structure import create_bilayer
from functions.bandstructure import calc_gap
from gpaw import PW, GPAW
# from gpaw.new.ase_interface import GPAW
from ase.calculators.dftd3 import DFTD3
from ase.filters import UnitCellFilter
from ase.optimize.bfgs import BFGS
from ase.parallel import parprint
import csv

average_lattice = 3.2515
average_cell = np.array([[1, 0], [0.5, np.sqrt(3) / 2]]) * average_lattice

with open("results.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["i", "j", "v1_norm", "v2_norm", "z_dist", "gap"])


for i in np.linspace(0, 1, 25):
    for j in np.linspace(0, 1, 25):
        bilayer = create_bilayer(z_dist=6.6,  constrain=True,
                                 a_shift=i, b_shift=j)

        bilayer.calc = DFTD3(dft=GPAW(mode=PW(500),
                                      kpts=(8, 8, 1),
                                      xc='PBE',
                                      txt=None), xc='PBE')

        uf = UnitCellFilter(bilayer, mask=[1, 1, 0, 0, 0, 1])
        relax = BFGS(uf, trajectory=f'traj_files/opt_{i:.2f}_{j:.2f}.traj',
                     logfile=f'log_files/opt_{i:.2f}_{j:.2f}.log')
        relax.run(fmax=0.01, steps=300)

        gap = calc_gap(bilayer, kpts=30, soc=True)[0]

        v1_norm = np.linalg.norm(bilayer.cell[0])
        v2_norm = np.linalg.norm(bilayer.cell[1])

        symbols = np.array(bilayer.get_chemical_symbols())
        Mo_z = bilayer.positions[symbols == 'Mo'][:, 2]
        W_z = bilayer.positions[symbols == 'W'][:, 2]

        z_dist = np.abs(W_z - Mo_z)[0]

        parprint(f'Shift {i:.2f},{j:.2f} is done.')

        # Append results to CSV
        with open("results.csv", mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, j, v1_norm, v2_norm, z_dist, gap])
