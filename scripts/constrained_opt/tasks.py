import taskblaster as tb

import numpy as np
from pathlib import Path
import csv
from functions.bandstructure import calc_gap  # noqa F401

from gpaw import GPAW, PW

from ase.io import read, write
from ase import Atoms
from ase.build import mx2
from ase.constraints import FixedLine
from ase.optimize import BFGS
from ase.filters import UnitCellFilter


@tb.dynamical_workflow_generator_task
def generate_wfs_task(coords):
    for i in coords:
        wf = single_cell(x=i[0], y=i[1])  # type:ignore
        name = f'gap_{i[0]:.2f}_{i[1]:.2f}'
        yield name, wf


@tb.workflow
class single_cell:
    x = tb.var()
    y = tb.var()

    @tb.task
    def create_struct_task(self):
        return tb.node('create_structure',
                       i=self.x,
                       j=self.y)

    @tb.task
    def pre_relax_gap_task(self):
        return tb.node('calc_gap',
                       atom_path=self.create_struct_task,
                       kpts=36)

    @tb.task
    def relaxation_task(self):
        return tb.node('relaxation',
                       atom_path=self.create_struct_task,
                       i=self.x,
                       j=self.y)

    @tb.task
    def post_relax_gap_task(self):
        return tb.node('calc_gap',
                       atom_path=self.relaxation_task['path'],  # type:ignore
                       kpts=36)

    @tb.task
    def return_dict_task(self):
        return tb.node('return_as_dict',
                       i=self.x,
                       j=self.y,
                       pre_relax=self.pre_relax_gap_task,
                       post_relax=self.post_relax_gap_task)


def create_tuple_list(x: int, y: int) -> list[tuple[float, float]]:
    return [(i, j) for i in np.linspace(0, 1, x, endpoint=False)
            for j in np.linspace(0, 1, y, endpoint=False)]


def return_as_dict(i: int, j: int, pre_relax: float, post_relax: float):
    return {'x': i, 'y': j, 'pre': pre_relax, 'post': post_relax}


def write_results_to_csv(results_dict: dict, csv_name: str) -> Path:
    rows = []
    for name, d in results_dict.items():
        x = d['x']
        y = d['y']
        pre_relax = d['pre']
        post_relax = d['post']

        rows.append({
            "x": x,
            "y": y,
            "pre": pre_relax,
            "post": post_relax
        })

    csv_path = Path(csv_name)
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=["x", "y", "pre", "post"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def create_structure(i: float, j: float) -> Atoms:
    MoS2_len = 3.184
    WSe2_len = 3.319

    average_len = (MoS2_len + WSe2_len)/2

    MoS2 = mx2('MoS2', a=average_len, vacuum=6.0)
    WSe2 = mx2('WSe2', a=average_len, vacuum=6.0)

    # 6.6Å of distance between layers
    MoS2.positions[:, 2] += 3.3
    WSe2.positions[:, 2] -= 3.3

    # Create the initial structure
    struct = WSe2 + MoS2
    struct.center(vacuum=10.0, axis=2)

    struct.positions += struct.cell[0]

    indices = [atom.index for atom in struct if (atom.symbol == 'W' or
                                                 atom.symbol == 'Mo')]
    struct.set_constraint(FixedLine(indices=indices, direction=[0, 0, 1]))

    for atom in struct:
        if atom.symbol == 'Mo' or atom.symbol == 'S':
            atom.position += i * struct.cell[0] + j * struct.cell[1]

    struct.pbc = True
    struct.wrap()

    root = get_root_path()

    folder_path = root / 'atoms_files_pre_relax/'
    file_name = f'MoS2WSe2_{i:.2f}_{j:.2f}.xyz'
    # Write to external directory for organising
    write(folder_path / file_name, struct)
    # Write to internal directory for taskblasters sake
    write(file_name, struct)
    return Path(file_name)


def relaxation(atom_path: Path, i: int, j: int) -> dict:
    atoms = read(atom_path)

    atoms.calc = GPAW(mode=PW(500), xc='PBE', kpts={'size': (8, 8, 1)})

    root = get_root_path()

    uf = UnitCellFilter(atoms)
    traj_path = root / 'trajectory_files' / f'opt_{i:.2f}_{j:.2f}.traj'
    relax = BFGS(uf, trajectory=traj_path)
    relax.run(fmax=0.01)

    folder_path = root / 'atoms_files_post_relax/'
    file_name = f'MoS2WSe2_{i:.2f}_{j:.2f}_relaxed.xyz'

    # Write to external directory for organising
    write(folder_path / file_name, atoms)
    # Write to internal directory for taskblasters sake
    write(file_name, atoms)

    return {'path': Path(file_name),
            'lengths': atoms.cell.lengths()}


def get_root_path() -> Path:
    current_path = Path(__file__).resolve()
    print(f"Current path: {current_path}")

    for parent in current_path.parents:
        if parent.name == 'constrained_opt':
            base_dir = parent
            break
    else:
        raise FileNotFoundError(
            f"Could not find a directory named 'constrained_opt' in the path "
            f"{current_path}"
        )

    return base_dir
