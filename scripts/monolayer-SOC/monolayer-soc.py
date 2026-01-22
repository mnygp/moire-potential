from ase.build import mx2
import numpy as np
from ase.parallel import parprint
import csv
from gpaw.new.ase_interface import GPAW
from gpaw import FermiDirac, PW
from functions.bandstructure import generate_scissor_shifts
from ase import Atoms
from ase.io import read
from pathlib import Path
from gpaw.spinorbit import soc_eigenstates


def gap_and_kpts(atom_path, functional, kpts, occ_thresh=0.5, scissors=False):
    if isinstance(atom_path, Path):
        atoms = read(atom_path)
    elif isinstance(atom_path, Atoms):
        atoms = atom_path
    else:
        raise TypeError("atom_path must be a Path or Atoms object")

    if scissors:
        shift_arr = generate_scissor_shifts(atom_path)
        calc = GPAW(
            mode="lcao",
            basis="dzp",
            kpts=dict(size=(kpts, kpts, 1), gamma=True),
            eigensolver={"name": "scissors", "shifts": shift_arr},
            txt="gpaw.txt",
        )
    else:
        calc = GPAW(
            mode=PW(500),  # Basis set
            xc=functional,  # Functional
            kpts={"size": (kpts, kpts, 1)},  # k-points
            occupations=FermiDirac(0.01),
            txt="gpaw.txt",
        )

    atoms.calc = calc
    atoms.get_potential_energy()

    V = calc.get_electrostatic_potential()
    parprint(type(V), V.shape)

    V_avg = V.mean(axis=(0, 1))  # average over x and y
    vacuum_level = V_avg.max()

    bz_kpts = calc.get_bz_k_points()

    soc_eig = soc_eigenstates(calc)
    eigs = soc_eig.eigenvalues()
    occ = soc_eig.occupation_numbers()

    global_homo = -np.inf
    global_lumo = np.inf
    homo_kpt = None
    lumo_kpt = None

    # Loop over k-points to find global HOMO/LUMO
    for i, (kpt_eigs, kpt_occ, kpt) in enumerate(zip(eigs, occ, bz_kpts)):
        # Occupied/unoccupied masks
        occ_mask = kpt_occ > occ_thresh
        unocc_mask = kpt_occ <= occ_thresh

        if np.any(occ_mask):
            kpt_homo = np.max(kpt_eigs[occ_mask])
            if kpt_homo > global_homo:
                global_homo = kpt_homo
                homo_kpt = kpt

        if np.any(unocc_mask):
            kpt_lumo = np.min(kpt_eigs[unocc_mask])
            if kpt_lumo < global_lumo:
                global_lumo = kpt_lumo
                lumo_kpt = kpt

    return (
        vacuum_level,
        global_lumo,
        global_homo,
        lumo_kpt,
        homo_kpt,
    )


lattice_arr = np.linspace(3.184, 3.319, 10)
for name, thick in zip(["MoS2", "WSe2"], [3.13, 3.36]):
    lumo_arr = []
    homo_arr = []
    lumo_kpts_arr_x = []
    lumo_kpts_arr_y = []
    homo_kpts_arr_x = []
    homo_kpts_arr_y = []

    for lattice in lattice_arr:
        atoms = mx2(name, a=lattice, thickness=thick, vacuum=10)

        vacuum, lumo, homo, lumo_kpts, homo_kpts = gap_and_kpts(
            atoms, "PBE", 36, scissors=True
        )

        lumo_arr.append(lumo - vacuum)
        homo_arr.append(homo - vacuum)
        lumo_kpts_arr_x.append(lumo_kpts[0])
        lumo_kpts_arr_y.append(lumo_kpts[1])
        homo_kpts_arr_x.append(homo_kpts[0])
        homo_kpts_arr_y.append(homo_kpts[1])

        parprint(f"{name} with lattice constant {lattice:.2f} done")

    rows = zip(
        lattice_arr,
        lumo_arr,
        homo_arr,
        lumo_kpts_arr_x,
        lumo_kpts_arr_y,
        homo_kpts_arr_x,
        homo_kpts_arr_y,
    )

    with open(f"{name}_scissors_SOC.csv", mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "lattice",
                "lumo",
                "homo",
                "lumo kpts_x",
                "lumo kpts_y",
                "homo kpts_x",
                "homo kpts_y",
            ]
        )
        writer.writerows(rows)
