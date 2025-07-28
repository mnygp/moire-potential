from ase.optimize import BFGS
from ase.parallel import parprint
from ase.calculators.dftd3 import DFTD3
from gpaw import GPAW, PW
from functions.bandstructure import calc_gap
from functions.structure import create_bilayer
import numpy as np
import csv

for a in [3.184, 3.2515, 3.319]:  # MoS2, average and WSe2 lattice constants

    shift_arr = np.linspace(0, 1, 30)
    z_dist_arr: list[float] = []
    gap_arr: list[float] = []

    for shift in shift_arr:
        struct = create_bilayer(6.6, lattice_length=a,
                                a_shift=shift, b_shift=shift,
                                constrain=True,
                                acute_corner=True)

        calc = GPAW(mode=PW(500),
                    xc='PBE',
                    kpts={'size': (12, 12, 1)},
                    txt='gpaw.txt')
        d3_calc = DFTD3(dft=calc)
        struct.calc = d3_calc
        struct.get_potential_energy()

        opt = BFGS(struct,
                   trajectory=f'traj_files/opt_{a:.2f}_{shift:.2f}.traj')
        opt.run(fmax=0.01)

        post_relax_gap = calc_gap(struct, kpts=36)
        gap_arr.append(post_relax_gap)

        symbols = np.array(struct.get_chemical_symbols())
        relaxed_z_dist = (struct[symbols == 'W'].positions[0][2]
                          - struct[symbols == 'Mo'].positions[0][2])
        z_dist_arr.append(abs(relaxed_z_dist))
        parprint(f'{shift:.2f} shift done')

    rows = zip(shift_arr, z_dist_arr, gap_arr)

    with open(f'gap_shift_{a:.2f}.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['shift', 'z_dist', 'gap'])
        writer.writerows(rows)
    parprint(f'a={a:.2f} done')
