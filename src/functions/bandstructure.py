from ase.io import read
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac
from gpaw.spinorbit import soc_eigenstates
from pathlib import Path
import numpy as np


def calc_gap(
    atom_path: Path | Atoms,
    functional: str = "PBE",
    kpts: int = 18,
    pw_cut: float = 500,
    soc: bool = False,
) -> tuple[float, float, float]:

    if isinstance(atom_path, Path):
        atoms = read(atom_path)
    elif isinstance(atom_path, Atoms):
        atoms = atom_path
    else:
        raise TypeError("atom_path must be a Path or Atoms object")

    calc = GPAW(
        mode=PW(pw_cut),  # Basis set
        xc=functional,  # Functional
        kpts={"size": (kpts, kpts, 1)},  # k-points
        occupations=FermiDirac(0.01),
        txt=None,
    )

    atoms.calc = calc
    atoms.get_potential_energy()

    if soc:
        soc_eig = soc_eigenstates(calc)
        eigs = soc_eig.eigenvalues()
        occ = soc_eig.occupation_numbers()

        energies = eigs.ravel()
        occs = occ.ravel()

        occ_thresh = 0.5
        occupied_energies = energies[occs > occ_thresh]
        unoccupied_energies = energies[occs <= occ_thresh]

        homo = occupied_energies.max()
        lumo = unoccupied_energies.min()
    else:
        homo, lumo = calc.get_homo_lumo()

    return lumo - homo, lumo, homo


def get_vacuum_and_band_edges(gpw_file: str, soc=False):
    calc = GPAW(gpw_file, txt=None)
    V = calc.get_electrostatic_potential()
    V_avg = V.mean(axis=(0, 1))  # average over x and y
    vacuum_level = V_avg.max()
    ef = calc.get_fermi_level()  # in eV

    if soc:
        soc = soc_eigenstates(calc)
        eigs = soc.eigenvalues()
        occ = soc.occupation_numbers()

        energies = eigs.ravel()  # flatten eigenvalues into 1D array
        occs = occ.ravel()  # flatten occupation numbers

        occ_thresh = 0.5
        occupied_energies = energies[occs > occ_thresh]
        unoccupied_energies = energies[occs <= occ_thresh]

        homo = occupied_energies.max()
        lumo = unoccupied_energies.min()
    else:
        eigs = calc.get_eigenvalues(spin=0)  # first spin channel

        if calc.get_number_of_spins() == 2:
            eigs_spin1 = calc.get_eigenvalues(spin=1)
            eigs = np.concatenate([eigs, eigs_spin1])

        homo = eigs[eigs <= ef].max()
        lumo = eigs[eigs >= ef].min()

    # 5. Shift to vacuum level reference
    homo_rel = homo - vacuum_level
    lumo_rel = lumo - vacuum_level

    return {
        "vacuum_level": vacuum_level,
        "homo": homo_rel,
        "lumo": lumo_rel,
        "bandgap": lumo-homo,
    }


def gap_and_kpts(atom_path, functional, kpts, occ_thresh=0.5):
    if isinstance(atom_path, Path):
        atoms = read(atom_path)
    elif isinstance(atom_path, Atoms):
        atoms = atom_path
    else:
        raise TypeError("atom_path must be a Path or Atoms object")

    calc = GPAW(
        mode=PW(500),  # Basis set
        xc=functional,  # Functional
        kpts={"size": (kpts, kpts, 1)},  # k-points
        occupations=FermiDirac(0.01),
        txt='gpaw.txt',
    )

    atoms.calc = calc
    atoms.get_potential_energy()

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

    return {
        "gap": global_lumo - global_homo,
        "lumo": global_lumo,
        "homo": global_homo,
        "lumo_kpt": lumo_kpt,
        "homo_kpt": homo_kpt,
    }
