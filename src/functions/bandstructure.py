from ase.io import read
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac
from pathlib import Path


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
