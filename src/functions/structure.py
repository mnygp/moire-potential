from ase.build import mx2
from ase import Atoms
from ase.constraints import FixedLine


def create_bilayer(z_dist: float, lattice_length: float = 3.2515,
                   a_shift: float = 0, b_shift: float = 0,
                   constrain: bool = False,
                   acute_corner: bool = False) -> Atoms:

    MoS2 = mx2('MoS2', a=lattice_length, vacuum=6.0)
    WSe2 = mx2('WSe2', a=lattice_length, vacuum=6.0)

    # 6.6Å of distance between layers
    MoS2.positions[:, 2] += z_dist/2
    WSe2.positions[:, 2] -= z_dist/2

    # Create the initial structure
    struct = WSe2 + MoS2
    struct.center(vacuum=10.0, axis=2)

    if constrain:
        indices = [atom.index for atom in struct if (atom.symbol == 'W' or
                                                     atom.symbol == 'Mo')]
        struct.set_constraint(FixedLine(indices=indices, direction=[0, 0, 1]))

    if acute_corner:
        struct.positions += struct.cell[0]
        for atom in struct:
            if atom.symbol == 'Mo' or atom.symbol == 'S':
                atom.position -= a_shift * struct.cell[0]
                atom.position += b_shift * struct.cell[1]
    else:
        for atom in struct:
            if atom.symbol == 'Mo' or atom.symbol == 'S':
                atom.position += a_shift*struct.cell[0]
                atom.position += b_shift*struct.cell[1]

    struct.pbc = True
    struct.wrap()

    return struct
