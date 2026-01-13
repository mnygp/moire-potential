from ase import Atoms
from ase.build import mx2
from ase.parallel import parprint
import numpy as np
from functions.bandstructure import calc_gap
from ase.optimize import BFGS
from gpaw import GPAW, PW
import csv
from ase.calculators.dftd3 import DFTD3


MoS2_len = 3.184
WSe2_len = 3.319


def create_structure(lattice_length: float, shift: float) -> Atoms:
    MoS2 = mx2("MoS2", a=lattice_length, vacuum=6.0)
    WSe2 = mx2("WSe2", a=lattice_length, vacuum=6.0)

    # 6.6Å of distance between layers
    MoS2.positions[:, 2] += 3.3
    WSe2.positions[:, 2] -= 3.3

    # Create the initial structure
    struct = WSe2 + MoS2
    struct.center(vacuum=10.0, axis=2)

    struct.positions += struct.cell[0]

    # indices = [atom.index for atom in struct if (atom.symbol == 'W' or
    #                                              atom.symbol == 'Mo')]
    # struct.set_constraint(FixedLine(indices=indices, direction=[0, 0, 1]))

    for atom in struct:
        if atom.symbol == "Mo" or atom.symbol == "S":
            atom.position += shift * struct.cell[0] + shift * struct.cell[1]

    struct.pbc = True
    struct.wrap()

    return struct


def attach_gpaw_calculator(struct: Atoms, shift: float):
    calc_dft = GPAW(
        mode=PW(500),
        xc="PBE",
        kpts={"size": (8, 8, 1)},
        txt=f"gpaw_output_{shift:.2f}.txt",
    )
    calc = DFTD3(dft=calc_dft)
    struct.calc = calc


def relax_structure(struct: Atoms, shift: float, traj: str | None = None) -> Atoms:
    attach_gpaw_calculator(struct, shift)
    if traj is not None:
        relaxer = BFGS(struct, trajectory=traj)
    else:
        relaxer = BFGS(struct)
    relaxer.run(fmax=0.01)
    return struct


def get_z(struct: Atoms) -> float:
    Mo_index = [atom.index for atom in struct if atom.symbol == "Mo"][0]
    Mo_z = struct.positions[Mo_index][2]
    W_index = [atom.index for atom in struct if atom.symbol == "W"][0]
    W_z = struct.positions[W_index][2]
    # Mo_z = next(atom.position[2] for atom in struct if atom.symbol == 'Mo')
    # W_z = next(atom.position[2] for atom in struct if atom.symbol == 'W')
    return abs(Mo_z - W_z)


lattice_lengths = np.linspace(MoS2_len, WSe2_len, 20)
shifts = [0, 0.333, 0.667]

for shift in shifts:
    results = []  # [(shift, lattice_length, gap_before, z_before, gap_after, z_after)]
    for a in lattice_lengths:
        struct = create_structure(a, shift=shift)

        attach_gpaw_calculator(struct, shift=shift)
        gap_before = calc_gap(struct)
        parprint(
            f"Shift: {shift:.3f}, Lattice length: {a:.3f} Å,"
            + f" Gap before relaxation: {gap_before:.3f} eV"
        )
        z_before = get_z(struct)

        relaxed_struct = relax_structure(
            struct, shift=shift, traj=f"traj_files/traj_{shift:.2f}_{a:.2f}.traj"
        )
        gap_after = calc_gap(relaxed_struct)
        parprint(
            f"Shift: {shift:.3f}, Lattice length: {a:.3f} Å,"
            + f" Gap after relaxation: {gap_after:.3f} eV"
        )

        z_after = get_z(relaxed_struct)

        results.append((a, gap_before, z_before, gap_after, z_after))

    # Write CSV for this shift
    filename = f"bandgap_shift_{shift:.2f}.csv"
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Lattice_length(Å)",
                "Gap_before_relax(eV)",
                "Z_distance_before",
                "Gap_after_relax(eV)",
                "Z_distance_after",
            ]
        )
        for a, gap_before, z_before, gap_after, z_after in results:
            writer.writerow(
                [
                    f"{a:.3f}",
                    f"{gap_before:.4f}",
                    f"{z_before:.4f}",
                    f"{gap_after:.4f}",
                    f"{z_after:.4f}",
                ]
            )

    parprint(f"Results saved to {filename}")
