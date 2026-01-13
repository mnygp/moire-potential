from ase.parallel import parprint
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from gpaw.new.extensions import D3
from gpaw.new.ase_interface import GPAW
from gpaw import PW
from functions.bandstructure import calc_gap
from functions.structure import create_bilayer
from functions.util import generate_scissor_shifts
import numpy as np
import csv

# [3.184, 3.2515, 3.319]  MoS2, average and WSe2 lattice constants
for a in [3.2515]:  # np.linspace(3.184, 3.319, 7):
    shift_arr = np.linspace(0, 1, 30)
    z_dist_arr: list[float] = []
    gap_arr: list[float] = []
    gap_soc_arr: list[float] = []
    gap_scissors_arr: list[float] = []
    gap_scissors_soc_arr: list[float] = []

    for shift in shift_arr:
        struct = Trajectory(f'traj_files/opt_{a:.2f}_{shift:.2f}.traj')[-1]


        calc = GPAW(mode=PW(1000),
                    xc='PBE',
                    kpts={'size': (12, 12, 1)},
                    symmetry='off',
                    txt='gpaw.txt',
                    extensions=[D3(xc='PBE')],
                    convergence={'forces': 5e-4,
                                 'density': 1e-5})
        struct.calc = calc

        opt = BFGS(struct,
                   trajectory=f'traj_files/lcao_opt_{a:.2f}_{shift:.2f}_high_fid.traj',
                   logfile=f'log_files/lcao_opt_{a:.2f}_{shift:.2f}_high_fid.log')
        opt.run(fmax=0.005)

        shifts = generate_scissor_shifts(struct)
        eigsolver = {'name': 'scissors', 'shifts': shifts}

        post_relax_gap = calc_gap(struct, kpts=36, functional='PBE', mode='lcao')[0]
        gap_arr.append(post_relax_gap)

        post_relax_gap_soc = calc_gap(struct, kpts=36, functional='PBE', mode='lcao', soc=True)[0]
        gap_soc_arr.append(post_relax_gap_soc)

        post_relax_gap_scissors = calc_gap(struct, kpts=36, functional='PBE', mode='lcao', eigensolver=eigsolver)[0]
        gap_scissors_arr.append(post_relax_gap_scissors)

        post_relax_gap_scissors_soc = calc_gap(struct, kpts=36, functional='PBE', mode='lcao', soc=True, eigensolver=eigsolver)[0]
        gap_scissors_soc_arr.append(post_relax_gap_scissors_soc)

        symbols = np.array(struct.get_chemical_symbols())
        relaxed_z_dist = (struct[symbols == 'W'].positions[0][2]
                          - struct[symbols == 'Mo'].positions[0][2])
        z_dist_arr.append(abs(relaxed_z_dist))
        parprint(f'{shift:.2f} shift done')

    rows = zip(shift_arr, z_dist_arr, gap_arr, gap_soc_arr, gap_scissors_arr, gap_scissors_soc_arr)

    with open(f'gap_shift_{a:.2f}_lcao_high_fidelity.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['shift', 'z_dist', 'gap', 'gap_soc', 'gap_scissors', 'gap_scissors_soc'])
        writer.writerows(rows)
    parprint(f'a={a:.2f} done')
