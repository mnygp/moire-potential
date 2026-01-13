from ase.parallel import parprint
from functions.bandstructure import calc_gap
from functions.structure import create_bilayer
from functions.util import generate_scissor_shifts
import numpy as np
import csv


dist_arr = np.linspace(6.2, 8, 20)
shift_arr: list[float] = []
z_dist_arr: list[float] = []
gap_arr: list[float] = []
gap_soc_arr: list[float] = []
gap_scissors_arr: list[float] = []
gap_scissors_soc_arr: list[float] = []

for dist in dist_arr:
    struct = create_bilayer(z_dist=dist,
                            a_shift=0.65,
                            b_shift=0.65)


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
    shift_arr.append(65)
    parprint(f'{dist:.2f} z dist done')

rows = zip(shift_arr, dist_arr, gap_arr, gap_soc_arr, gap_scissors_arr, gap_scissors_soc_arr)

parprint(z_dist_arr)

with open(f'gap_shift_lcao_3.25_constant_shift_0.65.csv', mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['shift', 'z_dist', 'gap', 'gap_soc', 'gap_scissors', 'gap_scissors_soc'])
    writer.writerows(rows)
