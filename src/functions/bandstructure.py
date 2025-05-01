from ase.io import read
from ase import Atoms
from ase.parallel import parprint
from gpaw import GPAW, PW, FermiDirac
from pathlib import Path


def calc_gap(atom_path: Path | Atoms, path: str, functional: str = "PBE",
             kpts: tuple[float, float, float] = (18, 18, 1),
             pw_cut: float = 500, filename: None | str = None) -> float:

    if isinstance(atom_path, Path):
        atoms = read(atom_path)
    elif isinstance(atom_path, Atoms):
        atoms = atom_path
    else:
        raise TypeError("atom_path must be a Path or Atoms object")

    if filename is not None:
        file = filename
        parprint(f'Filename is {file}')
    else:
        parprint('No filename provided, using default')
        file = "gap_calc.gpw"

    calc = GPAW(mode=PW(pw_cut),  # Basis set
                xc=functional,  # Functional
                kpts={'size': kpts},  # k-points
                occupations=FermiDirac(0.01),
                txt='gpaw_output.gpw')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.get_fermi_level()
    calc.write(file)
    file_path = Path(file).resolve()
    calc_fix = GPAW(file_path).fixed_density(symmetry='off',
                                             kpts={'path': path,
                                                   'npoints': 60})
    parprint("HOMO-LUMO raw:", calc_fix.get_homo_lumo())
    homo, lumo = calc_fix.get_homo_lumo()

    return lumo - homo
