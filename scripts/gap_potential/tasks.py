import taskblaster as tb

import numpy as np
from pathlib import Path
import csv

from ase.io import read, write
from ase import Atoms

from functions.util import closest_index, check_formula, dist
from functions.bandstructure import calc_gap


@tb.dynamical_workflow_generator_task
def generate_wfs_task(result_dict):

    atoms = result_dict['atoms']
    centers = result_dict['centers']

    for i in range(len(atoms)):
        wf = SubWorkflow(atom=atoms[i], center=centers[i], number=i)
        name = f'gap_{i}'
        yield name, wf


@tb.workflow
class SubWorkflow:
    atom = tb.var()
    center = tb.var()
    number = tb.var()

    @tb.task
    def calc_gap_task(self):
        return tb.node('calculate_gap', atom=self.atom, center=self.center,
                       number=self.number)


def calculate_gap(atom: Path, center: list[float], number: int):
    atoms_obj = read(atom)
    path_map_2d = {
                   'ORCC': 'GXSX1YG',
                   'HEX':  'GMKG',
                   'MCL':  'GYHCH1XH2G',
                    }

    cell_type = atoms_obj.cell.get_bravais_lattice().name
    BZ_path = path_map_2d[cell_type]
    gap = calc_gap(atom, path=BZ_path, filename=f'gap_calc_{number}.gpw')
    return {'gap': gap, 'center': center, 'number': number}


def write_results_to_csv(results_dict) -> Path:
    rows = []
    for name, d in results_dict.items():
        center = d['center']
        gap = d['gap']

        if 'gap' in name:
            rows.append({
                "name": name,
                "gap": gap,
                "x": center[0],
                "y": center[1],
                "z": center[2]
            })

    csv_path = Path("results.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=["name", "gap", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def create_atoms_list(file: str) -> dict[str, list[Path | list[float]]]:
    atoms = read(file)

    v1, v2, origins = get_cells(atoms)

    atoms_arr, centers = get_atom_obj(atoms, origins, v1, v2)

    output_dir = Path("atom_files")
    output_dir.mkdir(exist_ok=True)

    atoms_paths = []
    for i, atoms in enumerate(atoms_arr):
        write(f'atom_files/atoms_{i}.xyz', atoms)
        atoms_paths.append(Path(f'atom_files/atoms_{i}.xyz'))

    return {'atoms': atoms_paths,
            'centers': centers.tolist(),
            'origins': origins.tolist()}


def get_path(structure_name: str) -> str:
    current_path = Path(__file__).resolve()
    print(f"Current path: {current_path}")

    for parent in current_path.parents:
        if parent.name == 'moire-potential':
            base_dir = parent
            break
    else:
        raise FileNotFoundError("Could not find a directory" /
                                "named moire-potential in the" /
                                f" path {current_path}")
    print(f"Base directory: {base_dir}")

    file = base_dir / 'structures' / structure_name / 'structure_ml.json'
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist.")
    print(f"File path: {file}")

    return str(file)


def get_cells(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atoms_L = atoms.repeat((3, 3, 1))

    W_pos = atoms.positions[atoms.symbols == "W"]
    W_pos += (atoms.cell[0] + atoms.cell[1])

    W_L_pos = atoms_L.positions[atoms_L.symbols == "W"]

    approx_v1 = W_pos[closest_index(W_pos[0], W_pos, index=1)] - W_pos[0]

    rot_matrix = np.array([[0.5, -np.sqrt(3)/2, 0],
                           [np.sqrt(3)/2, 0.5, 0],
                           [0, 0, 1]])

    approx_v2 = rot_matrix@approx_v1

    v1_arr = np.array([])
    v2_arr = np.array([])
    origins = np.array([])

    for pos in W_pos:
        v1_end = W_L_pos[closest_index(pos + approx_v1, W_L_pos)]
        v2_end = W_L_pos[closest_index(pos + approx_v2, W_L_pos)]

        v1 = (v1_end - pos)*0.999
        v2 = (v2_end - pos)*0.999

        v1_arr = np.append(v1_arr, v1)
        v2_arr = np.append(v2_arr, v2)
        origins = np.append(origins, pos)

    v1_arr = v1_arr.reshape(-1, 3)
    v2_arr = v2_arr.reshape(-1, 3)
    origins = origins.reshape(-1, 3) - (atoms.cell[0] + atoms.cell[1])

    v1_arr[:, 2] = 0
    v2_arr[:, 2] = 0

    return v1_arr, v2_arr, origins


def get_atom_obj(atoms, origins, v1, v2):
    moire_v1 = atoms.cell[0]
    moire_v2 = atoms.cell[1]

    atoms_L = atoms.repeat((3, 3, 1))
    atoms_L_pos = atoms_L.positions

    origins += (moire_v1 + moire_v2)

    V = np.stack((v1[:, :2], v2[:, :2]), axis=-1)
    V_inv = np.linalg.inv(V)

    dist = atoms_L_pos[:, None, :2] - origins[:, :2]
    # Transform coordinates to the small cell basis
    coeffs = np.matmul(V_inv[None, :, :, :], dist[:, :, :, None])[:, :, :, 0]

    atoms_cell = Atoms()

    atoms_arr = []
    cell_centers = []

    for i in range(np.shape(coeffs)[1]):
        atoms_cell = Atoms()
        atom_coeffs = coeffs[:, i, :]
        inside_cell_indices = np.where(np.all(atom_coeffs >= 0, axis=1)
                                       & np.all(atom_coeffs < 1, axis=1))

        center_of_cell = v1[i]/2 + v2[i]/2 + origins[i] - (moire_v1 + moire_v2)

        for j in inside_cell_indices:
            atoms_cell += atoms_L[j]
            atoms_cell.positions = origins[i] - atoms_cell.positions
            atoms_cell.cell = np.array([v1[i], v2[i], np.array([0, 0, 24])])
            atoms_cell.center(axis=2)
            atoms_cell.pbc = [True, True, True]

        if not check_formula(atoms_cell.get_chemical_symbols()):
            atoms_cell = fix_cell(atoms_cell, origins[i])

        cell_centers.append(center_of_cell)
        atoms_arr.append(atoms_cell)

    return atoms_arr, np.array(cell_centers)


def fix_cell(atoms: Atoms, origin: np.ndarray) -> Atoms:
    chemical_number = {'Mo': 1, 'W': 1, 'S': 2, 'Se': 2}

    while not check_formula(atoms.get_chemical_symbols()):
        for atom_type in chemical_number.keys():
            symb = np.array(atoms.get_chemical_symbols())
            if len(symb[symb == atom_type]) != chemical_number[atom_type]:
                indices = [i for i, x in enumerate(symb) if x == atom_type]

                distances = np.array([dist(atoms.positions[i], origin)
                                     for i in indices])

                furthest_index = indices[np.argmax(distances)]
                del atoms[furthest_index]

    return atoms
