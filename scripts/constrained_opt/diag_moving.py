from ase.build import mx2
from ase.constraints import FixedLine
from ase.optimize import BFGS
from ase.parallel import parprint
from ase.io import write
from ase import Atoms
from ase.calculators.dftd3 import DFTD3

from gpaw import GPAW, PW

from functions.bandstructure import calc_gap

import numpy as np
import csv


def create_structure() -> Atoms:
    MoS2_len = 3.184
    WSe2_len = 3.319

    MoS2 = mx2('MoS2', a=MoS2_len, vacuum=10.0)
    WSe2 = mx2('WSe2', a=WSe2_len, vacuum=10.0)

    MoS2.positions[:, 2] += 3.2
    WSe2.positions[:, 2] -= 3.2

    # Create the initial structure
    struct = WSe2 + MoS2
    struct.center(vacuum=10.0, axis=2)

    struct.positions += struct.cell[0]

    indices = [atom.index for atom in struct if (atom.symbol == 'W' or
                                                 atom.symbol == 'Mo')]
    struct.set_constraint(FixedLine(indices=indices, direction=[0, 0, 1]))

    struct.pbc = True
    struct.wrap()

    return struct


shift_arr: list[float] = []
post_z_dist_arr: list[float] = []
pre_gap_arr: list[float] = []
post_gap_arr: list[float] = []

for i in np.linspace(0, 1, 30):
    shift_arr.append(i)
    struct = create_structure()

    diag = - struct.cell[0] + struct.cell[1]

    for atom in struct:
        if atom.symbol == 'Mo' or atom.symbol == 'S':
            atom.position += i * diag
    struct.wrap()
    write(f'atoms_files/struct_{i:.2f}.xyz', struct)
    pre_relax_gap = calc_gap(struct, kpts=36)
    pre_gap_arr.append(pre_relax_gap)

    calc = GPAW(mode=PW(500),
                xc='PBE',
                kpts={'size': (12, 12, 1)},
                txt='gpaw.txt')
    d3_calc = DFTD3(dft=calc)
    struct.calc = d3_calc
    struct.get_potential_energy()

    opt = BFGS(struct, trajectory=f'trajectory_files/opt_{i:.2f}.traj')
    opt.run(fmax=0.01)

    post_relax_gap = calc_gap(struct, kpts=36)
    post_gap_arr.append(post_relax_gap)

    symb = np.array(struct.get_chemical_symbols())
    post_relax_z_dist = (struct[symb == 'W'].positions[0][2]
                         - struct[symb == 'Mo'].positions[0][2])
    post_z_dist_arr.append(abs(post_relax_z_dist))
    parprint(f'{i:.2f} shift done')


rows = zip(shift_arr, post_z_dist_arr, pre_gap_arr, post_gap_arr)

with open('gap_shift.csv', mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['shift', 'z_dist', 'pre_gap', 'post_gap'])
    writer.writerows(rows)
parprint('All done')
