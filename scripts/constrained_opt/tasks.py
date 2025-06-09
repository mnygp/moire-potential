import taskblaster as tb

import numpy as np
from pathlib import Path
import csv
# import os
from functions.bandstructure import calc_gap  # noqa F401
import functions.util as util  # noqa F401

from gpaw import GPAW, PW

from ase.io import read, write
from ase import Atoms
from ase.build import mx2
from ase.constraints import FixedLine
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from ase.calculators.dftd3 import DFTD3


@tb.dynamical_workflow_generator_task
def generate_wfs_task(input: dict[str, list[list[float]]]):
    for i, c in zip(input['points'], input['centers']):
        wf = single_cell(x=i[0], y=i[1], center=c)  # type:ignore
        name = f'gap_{i[0]:.2f}_{i[1]:.2f}'
        yield name, wf


@tb.workflow
class single_cell:
    x = tb.var()
    y = tb.var()
    center = tb.var()

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
    def relax_no_strain_task(self):
        return tb.node('relaxation',
                       atom_path=self.create_struct_task,
                       i=self.x,
                       j=self.y,
                       fixed_cell=True)

    @tb.task
    def post_relax_no_strain_gap_task(self):
        return tb.node('calc_gap',
                       atom_path=self.relax_no_strain_task,
                       kpts=36)

    @tb.task
    def distance_no_strain_task(self):
        return tb.node('get_z_dist', atom_path=self.relax_no_strain_task)

    @tb.task
    def relax_with_strain_task(self):
        return tb.node('relaxation',
                       atom_path=self.create_struct_task,
                       i=self.x,
                       j=self.y,
                       fixed_cell=False)

    @tb.task
    def post_relax_with_strain_gap_task(self):
        return tb.node('calc_gap',
                       atom_path=self.relax_with_strain_task,
                       kpts=36)

    @tb.task
    def distance_with_strain_task(self):
        return tb.node('get_z_dist', atom_path=self.relax_with_strain_task)

    @tb.task
    def return_dict_task(self):
        return tb.node('return_as_dict',
                       i=self.x,
                       j=self.y,
                       pre_relax=self.pre_relax_gap_task,
                       post_relax_no_strain=self.post_relax_no_strain_gap_task,
                       z_dist_no_strain=self.distance_no_strain_task,
                       post_relax_with_strain=self.post_relax_with_strain_gap_task,  # noqa: E501
                       z_dist_with_strain=self.distance_with_strain_task,
                       center=self.center)


def get_shifts(structure_path: str,
               root: str) -> dict[str, list[list[float]]]:
    atoms = read(root + structure_path)
    v1_arr, v2_arr, origins = util.get_cells(atoms)
    atoms_arr, cell_center = util.get_atom_obj(atoms, origins, v1_arr, v2_arr)

    x = []
    y = []

    for a, o in zip(atoms_arr, origins):
        a.wrap()
        cell_vectors = a.cell[:2, :2]
        # cell_vectors[:, 1] *= -1
        Mo_postion = a.positions[a.get_chemical_symbols().index('Mo')]
        # distance = util.dist(Mo_postion, o)
        a.wrap()

        crystal_pos = np.linalg.solve(cell_vectors.T, Mo_postion[:2])
        x.append(crystal_pos[0])
        y.append(crystal_pos[1])

        # write(f'test_atom_files/atoms_{x[-1]:.2f}_{y[-1]:.2f}.xyz', a)

    combined = np.stack((x, y), axis=-1).tolist()

    return {'points': combined, 'centers': cell_center.tolist()}  # type:ignore


def return_as_dict(i: int, j: int, pre_relax: float,
                   post_relax_no_strain: float, z_dist_no_strain: float,
                   post_relax_with_strain: float, z_dist_with_strain: float,
                   center: list[float]) -> dict[str, float | int
                                                | list[float]]:
    return {'x': i, 'y': j, 'center': center, 'pre': pre_relax,
            'post no strain': post_relax_no_strain,
            'distance no strain': z_dist_no_strain,
            'post with strain': post_relax_with_strain,
            'distance with strain': z_dist_with_strain}


def write_results_to_csv(results_dict: dict, csv_name: str) -> Path:
    rows = []
    for name, d in results_dict.items():
        x = d['x']
        y = d['y']
        cx = d['center'][0]
        cy = d['center'][1]
        pre_relax = d['pre']
        post_relax_no_strain = d['post no strain']
        dist_no_strain = d['distance no strain']
        post_relax_with_strain = d['post no strain']
        dist_with_strain = d['distance no strain']

        rows.append({
            "x": x,
            "y": y,
            "center x": cx,
            "center y": cy,
            "pre": pre_relax,
            "post no strain": post_relax_no_strain,
            "dist no strain": dist_no_strain,
            "post with strain": post_relax_with_strain,
            "dist with strain": dist_with_strain
        })

    csv_path = Path(csv_name)
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=["x", "y", "center x", "center y",
                                            "pre",
                                            "post no strain",
                                            "dist no strain",
                                            "post with strain",
                                            "dist with strain"])
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

    root = get_root_path('constrained_opt')

    folder_path = root + '/atoms_files_pre_relax/'
    file_name = f'MoS2WSe2_{i:.3f}_{j:.3f}.traj'
    # Write to external directory for organising
    write(folder_path + file_name, struct)
    # Write to internal directory for taskblasters sake
    write(file_name, struct)
    return Path(file_name)


def relaxation(atom_path: str, i: int, j: int, fixed_cell: bool) -> Path:
    atoms = read(atom_path)

    calc = GPAW(mode=PW(500), xc='PBE', kpts={'size': (8, 8, 1)})
    d3_calc = DFTD3(dft=calc)
    atoms.calc = d3_calc

    root = get_root_path('constrained_opt')

    traj_name = f'opt_{i:.3f}_{j:.3f}.traj'

    if fixed_cell:
        opt = atoms
        traj_path = root + '/traj_files_no_strain/' + traj_name
    else:
        # mask makes it so the unitcell is only optimised in x and y
        opt = UnitCellFilter(atoms, mask=[1, 1, 0, 0, 0, 1])
        traj_path = root + '/traj_files_with_strain/' + traj_name

    relax = BFGS(opt, trajectory=traj_path)
    relax.run(fmax=0.01)

    folder_path = root + '/atoms_files_post_relax/'
    file_name = f'MoS2WSe2_{i:.3f}_{j:.3f}_relaxed.traj'

    # Write to external directory for organising
    write(folder_path + file_name, atoms)
    # Write to internal directory for taskblasters sake
    write(file_name, atoms)

    return Path(file_name)


def get_z_dist(atom_path: Path):
    atoms = read(atom_path)
    symb = np.array(atoms.get_chemical_symbols())
    z_dist = (atoms[symb == 'W'].positions[0][2]
              - atoms[symb == 'Mo'].positions[0][2])
    return z_dist


def get_root_path(directory: str) -> str:
    current_path = Path(__file__).resolve()
    print(f"Current path: {current_path}")

    for parent in current_path.parents:
        if parent.name == directory:
            return str(parent)
    else:
        raise FileNotFoundError(
            f"Could not find a directory named 'moire-potential' in the path "
            f"{current_path}")
