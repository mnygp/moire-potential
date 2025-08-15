from ase.io import read
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac
from pathlib import Path
import numpy as np


def calc_gap(atom_path: Path | Atoms, functional: str = "PBE",
             kpts: int = 18, pw_cut: float = 500) -> float:

    if isinstance(atom_path, Path):
        atoms = read(atom_path)
    elif isinstance(atom_path, Atoms):
        atoms = atom_path
    else:
        raise TypeError("atom_path must be a Path or Atoms object")

    calc = GPAW(mode=PW(pw_cut),  # Basis set
                xc=functional,  # Functional
                kpts={'size': (kpts, kpts, 1)},  # k-points
                occupations=FermiDirac(0.01),
                txt='gpaw_output.gpw')

    atoms.calc = calc
    atoms.get_potential_energy()

    homo, lumo = calc.get_homo_lumo()

    return lumo - homo


def get_vacuum_and_band_edges(gpw_file: str):
    calc = GPAW(gpw_file, txt=None)

    # 1. Get electrostatic potential
    V = calc.get_electrostatic_potential()
    V_avg = V.mean(axis=(0, 1))  # average over x and y
    vacuum_level = V_avg.max()

    # 2. Get Fermi level and eigenvalues
    ef = calc.get_fermi_level()  # in eV
    eigs = calc.get_eigenvalues(spin=0)  # first spin channel

    # For spin-polarized case, include both spin channels
    if calc.get_number_of_spins() == 2:
        eigs_spin1 = calc.get_eigenvalues(spin=1)
        eigs = np.concatenate([eigs, eigs_spin1])

    # 3. HOMO is the highest eigenvalue <= Fermi level
    homo = eigs[eigs <= ef].max()
    # 4. LUMO is the lowest eigenvalue >= Fermi level
    lumo = eigs[eigs >= ef].min()

    # 5. Shift to vacuum level reference
    homo_rel = homo - vacuum_level
    lumo_rel = lumo - vacuum_level

    return {
        "vacuum_level": vacuum_level,
        "homo": homo_rel,
        "lumo": lumo_rel,
        "bandgap": lumo - homo
    }
