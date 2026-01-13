from ase.optimize import BFGS
from ase.parallel import parprint
from gpaw.new.extensions import D3
from gpaw.new.ase_interface import GPAW
from gpaw import PW
from functions.bandstructure import calc_gap
from functions.structure import create_bilayer
import numpy as np
import csv

# [3.184, 3.2515, 3.319]  MoS2, average and WSe2 lattice constants
for a in [3.2515]:  # np.linspace(3.184, 3.319, 7):
    shift_arr = np.linspace(0, 1, 30, endpoint=False)
    z_dist_arr: list[float] = []
    gap_arr: list[float] = []
    gap_soc_arr: list[float] = []

    for shift in shift_arr:
        struct = create_bilayer(6.6, lattice_length=a,
                                a_shift=shift, b_shift=shift,
                                constrain=True,
                                acute_corner=True)

        calc = GPAW(mode=PW(500),
                    xc='PBE',
                    kpts={'size': (12, 12, 1)},
                    symmetry='off',
                    txt='gpaw.txt',
                    extensions=[D3(xc='PBE')])
        struct.calc = calc
        struct.get_potential_energy()

        opt = BFGS(struct,
                   trajectory=f'traj_files/lcao_opt_{a:.2f}_{shift:.2f}.traj',
                   logfile=f'log_files/lcao_opt_{a:.2f}_{shift:.2f}.log')
        opt.run(fmax=0.01)

        post_relax_gap = calc_gap(struct, kpts=36, functional='PBE', mode='lcao')[0]
        gap_arr.append(post_relax_gap)

        post_relax_gap_soc = calc_gap(struct, kpts=36, functional='PBE', mode='lcao', soc=True)[0]
        gap_soc_arr.append(post_relax_gap_soc)

        symbols = np.array(struct.get_chemical_symbols())
        relaxed_z_dist = (struct[symbols == 'W'].positions[0][2]
                          - struct[symbols == 'Mo'].positions[0][2])
        z_dist_arr.append(abs(relaxed_z_dist))
        parprint(f'{shift:.2f} shift done')

    rows = zip(shift_arr, z_dist_arr, gap_arr, gap_soc_arr)

    with open(f'gap_shift_{a:.2f}_lcao.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['shift', 'z_dist', 'gap', 'gap_soc'])
        writer.writerows(rows)
    parprint(f'a={a:.2f} done')
