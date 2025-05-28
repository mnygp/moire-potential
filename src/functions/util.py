import numpy as np
from ase import Atoms
from pathlib import Path


def closest_index(position: np.ndarray,
                  particles: np.ndarray,
                  index: int = 0,
                  twoD: bool = True) -> int:
    if twoD:
        r_diff = particles[:, :2] - position[:2]
    else:
        r_diff = particles - position

    distances = np.sqrt(np.sum(r_diff**2, axis=1))

    return np.argsort(distances)[index]


def repeate_cells(x: np.ndarray, y: np.ndarray, data: np.ndarray,
                  n_cells: range, vector1: np.ndarray,
                  vector2: np.ndarray) -> tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray]:

    # data_original = data.copy()
    x_original = x.copy()
    y_original = y.copy()
    data_original = data.copy()

    for i in n_cells:
        for j in n_cells:
            if i == 0 and j == 0:
                continue

            data = np.concatenate((data, data_original), axis=0)
            new_x = x_original + i*vector1[0] + j*vector2[0]
            new_y = y_original + i*vector1[1] + j*vector2[1]
            x = np.concatenate((x, new_x), axis=0)
            y = np.concatenate((y, new_y), axis=0)

    return x, y, data


def check_formula(chemicals: np.ndarray):
    chem = ["W", "Se", "Mo", "S"]
    number = [1, 2, 1, 2]
    chemicals = np.array(chemicals)
    for i, j in zip(number, chem):
        if i != np.sum(chemicals == j):
            return False

    return True


def add_distance(atoms: Atoms, distance: float) -> Atoms:

    positions = atoms.positions
    average_z = np.mean(positions[:, 2], axis=0)

    # Move the atoms above the center of mass up and below down
    for i in range(len(positions)):
        if positions[i, 2] > average_z:
            positions[i, 2] += distance/2
        else:
            positions[i, 2] -= distance/2

    atoms.cell[2, 2] += distance
    atoms.center(axis=2)

    # Set the new positions of the atoms
    # atoms.set_positions(positions)

    return atoms


def dist(a: np.ndarray, b: np.ndarray, twoD: bool = True) -> float:
    if twoD:
        a = a[:2]
        b = b[:2]
    return np.sqrt(np.sum((a - b)**2))


def get_root_path(directory: str) -> Path:
    current_path = Path(__file__).resolve()
    print(f"Current path: {current_path}")

    for parent in current_path.parents:
        if parent.name == directory:
            base_dir = parent
            break
    else:
        raise FileNotFoundError("Could not find a directory" +
                                f"named {directory} in the" +
                                f" path {current_path}")

    return base_dir


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


def get_atom_obj(atoms: Atoms, origins: np.ndarray,
                 v1: np.ndarray, v2: np.ndarray) -> tuple[list[Atoms],
                                                          np.ndarray]:
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
            # Check whether the number of atoms is correct
            if len(symb[symb == atom_type]) != chemical_number[atom_type]:
                indices = [i for i, x in enumerate(symb) if x == atom_type]
                # For S and Se, if there are 3 atoms, we want to keep
                # the pair of atoms that are above each other
                # and remove the one that is isolated
                if len(indices) == 3 and (atom_type == 'S'
                                          or atom_type == 'Se'):
                    remove_isolated(atoms, indices)
                else:
                    distances = np.array([dist(atoms.positions[i], origin)
                                          for i in indices])

                    furthest_index = indices[np.argmax(distances)]
                    del atoms[furthest_index]

    return atoms


def remove_isolated(atoms: Atoms, indices: list[int]) -> Atoms:
    distances = []
    for i in indices:
        d = atoms.get_distances(i, indices)
        distances.append(sum(d))
    # Remove the most isolated atom
    del atoms[indices[np.argmax(distances)]]
    return atoms
