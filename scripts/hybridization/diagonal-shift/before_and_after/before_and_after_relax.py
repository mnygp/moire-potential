from __future__ import annotations

import csv

import numpy as np
from ase.optimize import BFGS
from ase.parallel import parprint
from gpaw import PW
from gpaw.new.ase_interface import GPAW
from gpaw.new.extensions import D3

from functions.bandstructure import calc_gap
from functions.structure import create_bilayer
from functions.util import generate_scissor_shifts


def get_gaps(struct):
    shifts = generate_scissor_shifts(struct)
    eigsolver = {"name": "scissors", "shifts": shifts}

    gap = calc_gap(struct, kpts=36, functional="PBE", mode="lcao")[0]
    gap_soc = calc_gap(struct, kpts=36, functional="PBE", mode="lcao", soc=True)[0]

    gap_scissors = calc_gap(
        struct, kpts=36, functional="PBE", mode="lcao", eigensolver=eigsolver
    )[0]

    gap_scissors_soc = calc_gap(
        struct, kpts=36, functional="PBE", mode="lcao", soc=True, eigensolver=eigsolver
    )[0]

    symbols = np.array(struct.get_chemical_symbols())
    z_dist = (
        struct[symbols == "W"].positions[0][2] - struct[symbols == "Mo"].positions[0][2]
    )

    return (z_dist, gap, gap_soc, gap_scissors, gap_scissors_soc)


# [3.184, 3.2515, 3.319]  MoS2, average and WSe2 lattice constants
for a in [3.2515]:  # np.linspace(3.184, 3.319, 7):
    shift_arr = np.linspace(0.5, 0.85, 10)

    z_dist_arr_before: list[float] = []
    gap_arr_before: list[float] = []
    gap_soc_arr_before: list[float] = []
    gap_scissors_arr_before: list[float] = []
    gap_scissors_soc_arr_before: list[float] = []

    z_dist_arr_after: list[float] = []
    gap_arr_after: list[float] = []
    gap_soc_arr_after: list[float] = []
    gap_scissors_arr_after: list[float] = []
    gap_scissors_soc_arr_after: list[float] = []

    for shift in shift_arr:
        struct = create_bilayer(
            6.3,
            lattice_length=a,
            a_shift=shift,
            b_shift=shift,
            constrain=True,
            acute_corner=True,
        )

        (
            pre_relax_z_dist,
            pre_relax_gap,
            pre_relax_gap_soc,
            pre_relax_gap_scissors,
            pre_relax_gap_scissors_soc,
        ) = get_gaps(struct)
        gap_arr_before.append(pre_relax_gap)
        gap_soc_arr_before.append(pre_relax_gap_soc)
        gap_scissors_arr_before.append(pre_relax_gap_scissors)
        gap_scissors_soc_arr_before.append(pre_relax_gap_scissors_soc)

        parprint(f"Pre gaps done for shift: {shift:.2f}")

        calc = GPAW(
            mode=PW(800),
            convergence={"forces": 5e-4},
            kpts={"size": (10, 10, 1)},
            txt="gpaw.txt",
            extensions=[D3(xc="PBE")],
            symmetry="off",
        )
        struct.calc = calc

        opt = BFGS(
            struct,
            trajectory=f"before_and_after/lcao_opt_{a:.2f}_{shift:.2f}_high_fid.traj",
            logfile=f"before_and_after/lcao_opt_{a:.2f}_{shift:.2f}_high_fid.log",
        )
        opt.run(fmax=0.005)
        parprint(f"Relaxation done for shift: {shift:.2f}")

        (
            post_relax_z_dist,
            post_relax_gap,
            post_relax_gap_soc,
            post_relax_gap_scissors,
            post_relax_gap_scissors_soc,
        ) = get_gaps(struct)
        gap_arr_after.append(post_relax_gap)
        gap_soc_arr_after.append(post_relax_gap_soc)
        gap_scissors_arr_after.append(post_relax_gap_scissors)
        gap_scissors_soc_arr_after.append(post_relax_gap_scissors_soc)

        parprint(f"Post relax gaps done for shift: {shift:.2f}")
        parprint("\n")

    rows = zip(
        shift_arr,
        z_dist_arr_before,
        gap_arr_before,
        gap_soc_arr_before,
        gap_scissors_arr_before,
        gap_scissors_soc_arr_before,
    )
    with open(
        f"gap_shift_{a:.2f}_lcao_before_relax.csv",
        mode="w",
        newline="",
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["shift", "z_dist", "gap", "gap_soc", "gap_scissors", "gap_scissors_soc"]
        )
        writer.writerows(rows)

    rows = zip(
        shift_arr,
        z_dist_arr_after,
        gap_arr_after,
        gap_soc_arr_after,
        gap_scissors_arr_after,
        gap_scissors_soc_arr_after,
    )
    with open(
        f"gap_shift_{a:.2f}_lcao_after_relax.csv", mode="w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["shift", "z_dist", "gap", "gap_soc", "gap_scissors", "gap_scissors_soc"]
        )
        writer.writerows(rows)
