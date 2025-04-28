from ase.io import read
from gpaw import GPAW, PW, FermiDirac
from pathlib import Path


def calc_gap(atom_path: Path, path: str, funcional: str = "PBE",
             kpts: tuple[float, float, float] = (18, 18, 1),
             filename: None | str = None) -> float:

    atoms = read(atom_path)

    if filename is not None:
        file = filename
        print(f'Filename is {file}')
    else:
        print('No filename provided, using default')
        file = "gap_calc.gpw"

    # Set up the calculator
    calc = GPAW(mode=PW(400),  # Basis set
                xc=funcional,  # Functional
                kpts={'size': kpts},  # k-points
                occupations=FermiDirac(0.01),
                txt='gpaw_output.gpw')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.get_fermi_level()
    calc.write(file)
    file_path = Path(file)
    calc = GPAW(file_path).fixed_density(symmetry='off',
                                         kpts={'path': path, 'npoints': 60})
    print("HOMO-LUMO raw:", calc.get_homo_lumo())
    homo, lumo = calc.get_homo_lumo()

    return lumo - homo
