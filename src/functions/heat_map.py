import numpy as np
import warnings
from functions.util import closest_index, repeate_cells
from ase import Atoms


def height(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symbols = np.array(atoms.get_chemical_symbols())
    positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    Se_atoms = positions[symbols == "Se"]
    S_atoms = positions[symbols == "S"]

    Se_average = np.mean(Se_atoms[:, 2])
    S_average = np.mean(S_atoms[:, 2])

    # Mask out the atoms not in the top or bottom layer
    top_layer = Se_atoms[Se_atoms[:, 2] > Se_average]
    bottom_layer = S_atoms[S_atoms[:, 2] < S_average]

    x, y, z = repeate_cells(bottom_layer[:, 0],
                            bottom_layer[:, 1],
                            bottom_layer[:, 2],
                            range(-1, 2), vector1, vector2)

    bottom_layer_large = np.array([x, y, z]).T

    for top_pos in top_layer:
        close = closest_index(top_pos, bottom_layer_large)
        top_pos[2] = top_pos[2] - bottom_layer_large[close, 2]

    return top_layer[:, 0], top_layer[:, 1], top_layer[:, 2]


def horizontal_distance(atoms: Atoms) -> tuple[np.ndarray,
                                               np.ndarray,
                                               np.ndarray]:

    symbols = np.array(atoms.get_chemical_symbols())
    positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    # Mask out the atoms not in the top or bottom layer
    Se_atoms = positions[symbols == "Se"]
    S_atoms = positions[symbols == "S"]

    Se_average = np.mean(Se_atoms[:, 2])
    S_average = np.mean(S_atoms[:, 2])

    # Calculate the horizontal distance between the two middle layers
    Se_inner_atoms = Se_atoms[Se_atoms[:, 2] < Se_average]
    S_inner_atoms = S_atoms[S_atoms[:, 2] > S_average]

    x, y, z = repeate_cells(S_inner_atoms[:, 0],
                            S_inner_atoms[:, 1],
                            S_inner_atoms[:, 2],
                            range(-1, 2),
                            vector1, vector2)

    S_atoms_large = np.array([x, y, z]).T

    for pos in Se_inner_atoms:
        close = closest_index(pos, S_atoms_large)
        closest_particle = S_atoms_large[close]
        xy_distance = np.sqrt((pos[0] - closest_particle[0])**2
                              + (pos[1] - closest_particle[1])**2)
        pos[2] = xy_distance

    return Se_inner_atoms[:, 0], Se_inner_atoms[:, 1], Se_inner_atoms[:, 2]


def modified_h_dist(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symbols = np.array(atoms.get_chemical_symbols())
    positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    # Mask out the atoms not in the top or bottom layer
    Mo_atoms = positions[symbols == "Mo"]
    W_atoms = positions[symbols == "W"]

    x, y, z = repeate_cells(W_atoms[:, 0],
                            W_atoms[:, 1],
                            W_atoms[:, 2],
                            range(-1, 2),
                            vector1, vector2)

    W_atoms_large = np.array([x, y, z]).T

    for pos in Mo_atoms:
        close = closest_index(pos, W_atoms)
        closest_particle = W_atoms_large[close]
        xy_distance = np.sqrt((pos[0] - closest_particle[0])**2
                              + (pos[1] - closest_particle[1])**2)
        pos[2] = xy_distance

        diag_len = np.linalg.norm(vector1+vector2)
        if (closest_particle[0] > pos[0]):
            pos[2] = diag_len - xy_distance

    return Mo_atoms[:, 0], Mo_atoms[:, 1], Mo_atoms[:, 2]/diag_len


def strain(atoms: Atoms, atom_type: str) -> tuple[np.ndarray,
                                                  np.ndarray,
                                                  np.ndarray]:

    if (atom_type not in ['W', 'Mo']):
        raise ValueError("Input not a valid atom type."
                         + " Choose either 'W' or 'Mo'.")

    symbols = np.array(atoms.get_chemical_symbols())
    positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    # Choose what transitions metal to look at
    T_metal = positions[symbols == atom_type]

    if (atom_type == 'W'):
        ideal_len = 3.319
    elif (atom_type == 'Mo'):
        ideal_len = 3.184

    T_metal_large = T_metal.copy()

    # Periodic boundary conditions
    x, y, z = repeate_cells(T_metal[:, 0], T_metal[:, 1], T_metal[:, 2],
                            range(-1, 2), vector1, vector2)

    T_metal_large = np.array([x, y, z]).T

    strain_arr = np.zeros(len(T_metal[:, 0]))

    for i, pos in enumerate(T_metal):

        # Find the closest particles to the unstrained positions in [Å]
        diff = T_metal_large - pos
        distances = np.sqrt(np.sum(diff**2, axis=1))

        # Six closest particles excluding the particle itself
        six_closest_indices = np.argsort(distances)[1:7]

        six_closest = distances[six_closest_indices]  # in [Å]

        sum_strain = (six_closest - ideal_len)/ideal_len

        strain_arr[i] = np.mean(sum_strain)

    if (max(abs(strain_arr)) > 0.1):
        w = f"Suspiciously high strain of {max(abs(strain_arr)):.3f}"
        warnings.warn(w)
        print("")

    return T_metal[:, 0], T_metal[:, 1], strain_arr


def layer_thicknsess(atoms: Atoms, atom_type: str) -> tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray]:

    if (atom_type not in ['S', 'Se']):
        raise ValueError("Input not a valid atom type."
                         + " Choose either 'S' or 'Se'.")

    symbols = np.array(atoms.get_chemical_symbols())
    positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    chalcogen = positions[symbols == atom_type]

    center = np.mean(chalcogen[:, 2])

    top_layer = chalcogen[chalcogen[:, 2] > center]
    bottom_layer = chalcogen[chalcogen[:, 2] < center]

    x, y, z = repeate_cells(bottom_layer[:, 0],
                            bottom_layer[:, 1],
                            bottom_layer[:, 2],
                            range(-1, 2), vector1, vector2)

    bottom_layer_large = np.array([x, y, z]).T

    for i, pos in enumerate(top_layer):
        close = closest_index(pos, bottom_layer_large)
        top_layer[i, 2] = pos[2] - bottom_layer_large[close, 2]

    return top_layer[:, 0], top_layer[:, 1], top_layer[:, 2]


def interlayer_distance(atoms: Atoms) -> tuple[np.ndarray,
                                               np.ndarray,
                                               np.ndarray]:

    symbols = np.array(atoms.get_chemical_symbols())
    positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    Mo_atoms = positions[symbols == "Mo"]
    W_atoms = positions[symbols == "W"]

    x, y, z = repeate_cells(W_atoms[:, 0], W_atoms[:, 1], W_atoms[:, 2],
                            range(-1, 2), vector1, vector2)

    W_large = np.array([x, y, z]).T

    for pos in Mo_atoms:
        close = closest_index(pos, W_large)
        closest_particle = W_large[close]
        z_distance = abs(pos[2] - closest_particle[2])
        pos[2] = z_distance

    return Mo_atoms[:, 0], Mo_atoms[:, 1], Mo_atoms[:, 2]
