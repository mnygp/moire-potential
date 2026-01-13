from ase import Atoms
from ase.build import mx2
from ase.constraints import FixedLine
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.parallel import parprint
from ase.calculators.dftd3 import DFTD3
from gpaw import PW, FermiDirac
from gpaw.new.ase_interface import GPAW
import numpy as np
import csv


def create_bilayer(
    z_dist: float,
    lattice_length: float = 3.2515,
    a_shift: float = 0,
    b_shift: float = 0,
    constrain: bool = False,
    acute_corner: bool = True,
) -> Atoms:
    MoS2 = mx2("MoS2", a=lattice_length, vacuum=6.0)
    WSe2 = mx2("WSe2", a=lattice_length, vacuum=6.0)

    # 6.6Å of distance between layers
    MoS2.positions[:, 2] += z_dist / 2
    WSe2.positions[:, 2] -= z_dist / 2

    # Create the initial structure
    struct = WSe2 + MoS2
    struct.center(vacuum=10.0, axis=2)

    if constrain:
        indices = [
            atom.index for atom in struct if (atom.symbol == "W" or atom.symbol == "Mo")
        ]
        struct.set_constraint(FixedLine(indices=indices, direction=[0, 0, 1]))

    if acute_corner:
        struct.positions += struct.cell[0]
        for atom in struct:
            if atom.symbol == "Mo" or atom.symbol == "S":
                atom.position -= a_shift * struct.cell[0]
                atom.position += b_shift * struct.cell[1]
    else:
        for atom in struct:
            if atom.symbol == "Mo" or atom.symbol == "S":
                atom.position += a_shift * struct.cell[0]
                atom.position += b_shift * struct.cell[1]

    struct.pbc = True
    struct.wrap()

    return struct


def calc_gap(
    atoms, functional: str = "PBE", kpts: int = 18, pw_cut: float = 500
) -> tuple[float, float, float]:
    calc = GPAW(
        mode=PW(pw_cut),  # Basis set
        xc=functional,  # Functional
        kpts={"size": (kpts, kpts, 1)},  # k-points
        occupations=FermiDirac(0.01),
        txt=None,
    )

    atoms.calc = calc
    atoms.get_potential_energy()

    homo, lumo = calc.get_homo_lumo()

    return lumo - homo


a = 3.2515  # Average lattice length

shift_arr = np.linspace(0, 1, 30)
z_arr = []
vec1_arr = []
vec2_arr = []
gap_arr = []

for shift in shift_arr:
    struct = create_bilayer(
        6.6,
        lattice_length=a,
        a_shift=shift,
        b_shift=shift,
        constrain=True,
        acute_corner=True,
    )
    struct.pbc = [1, 1, 1]

    calc = GPAW(
        mode=PW(500),  # Basis set
        xc="PBE",  # Functional
        kpts={"size": (8, 8, 1)},  # k-points
        occupations=FermiDirac(0.01),
        txt=None,
    )
    struct.calc = DFTD3(dft=calc)

    opt_filter = FrechetCellFilter(struct, mask=[1, 1, 0, 0, 0, 1])
    opt = BFGS(
        opt_filter,
        logfile=f"logs/relax_{shift:.2f}.log",
        trajectory=f"traj_files/relax_{shift:.2f}.traj",
    )
    opt.run(fmax=0.002, steps=200)

    parprint("Starting gap calc")
    gap = calc_gap(struct, kpts=30)
    parprint("The gap is:")
    parprint(gap)

    symbols = np.array(struct.get_chemical_symbols())
    relaxed_z_dist = abs(
        struct[symbols == "W"].positions[0][2] - struct[symbols == "Mo"].positions[0][2]
    )
    z_arr.append(relaxed_z_dist)
    vec1_arr.append(np.linalg.norm(struct.cell[0, :2]))
    vec2_arr.append(np.linalg.norm(struct.cell[1, :2]))
    gap_arr.append(gap)

    parprint(f"Shift: {shift}")
    parprint(relaxed_z_dist)
    parprint(struct.cell[0, :2])
    parprint(struct.cell[1, :2])
    parprint(gap)


# Write to CSV
with open("DFT_data.csv", mode="w", newline="") as f:
    writer = csv.writer(f)

    # Optional: write header
    writer.writerow(["shift", "z", "vec1", "vec2", "gap"])

    # Write data rows
    for s, z, v1, v2, g in zip(shift_arr, z_arr, vec1_arr, vec2_arr, gap_arr):
        writer.writerow([s, z, v1, v2, g])
