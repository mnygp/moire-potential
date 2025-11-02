import numpy as np
import warnings
from functions.util import closest_index, repeate_cells
from ase import Atoms


def get_masks(atoms: Atoms, TM: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the top and bottom transition metals"""
    # Mask out the non trasition metals
    symbols = atoms.get_chemical_symbols()
    TM_mask = np.array(symbols) == TM
    T_metals = atoms[TM_mask]

    average_TM_z = np.mean(T_metals.positions[:, 2])

    top_mask = atoms.positions[:, 2] > average_TM_z
    bottom_mask = atoms.positions[:, 2] < average_TM_z

    top_TM_mask = top_mask & TM_mask
    bottom_TM_mask = bottom_mask & TM_mask

    return top_TM_mask, bottom_TM_mask


def strain(
    atoms: Atoms, atom_type: str, layer: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if layer not in ["top", "bottom"]:
        raise ValueError(
            "Input not a valid layer type."
            + " Choose either 'top' or 'bottom'."
        )
    if atom_type not in ["W", "Mo"]:
        raise ValueError("Input not a valid atom type."
                         + " Choose either 'W' or 'Mo'.")

    symbols = np.array(atoms.get_chemical_symbols())
    positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    # Choose what transitions metal to look at
    T_metal = positions[symbols == atom_type]
    if layer == "top":
        T_metal = T_metal[T_metal[:, 2] > np.mean(T_metal[:, 2])]
    elif layer == "bottom":
        T_metal = T_metal[T_metal[:, 2] < np.mean(T_metal[:, 2])]

    if atom_type == "W":
        ideal_len = 3.319
    elif atom_type == "Mo":
        ideal_len = 3.184

    T_metal_large = T_metal.copy()

    # Periodic boundary conditions
    x, y, z = repeate_cells(
        T_metal[:, 0], T_metal[:, 1], T_metal[:, 2],
        range(-1, 2), vector1, vector2
    )
    T_metal_large = np.array([x, y, z]).T
    strain_arr = np.zeros(len(T_metal[:, 0]))

    for i, pos in enumerate(T_metal):
        # Find the closest particles to the unstrained positions in [Å]
        diff = T_metal_large - pos
        distances = np.sqrt(np.sum(diff**2, axis=1))

        # Six closest particles excluding the particle itself
        six_closest_indices = np.argsort(distances)[1:7]
        six_closest = distances[six_closest_indices]  # in [Å]
        sum_strain = (six_closest - ideal_len) / ideal_len

        strain_arr[i] = np.mean(sum_strain)

    if max(abs(strain_arr)) > 0.1:
        w = f"Suspiciously high strain of {max(abs(strain_arr)):.3f}"
        warnings.warn(w)
        print("")

    return T_metal[:, 0], T_metal[:, 1], strain_arr


def interlayer_distance(
    atoms: Atoms, TM: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    top_mask, bottom_mask = get_masks(atoms, TM)
    top_atoms = positions[top_mask]
    bottom_atoms = positions[bottom_mask]

    x, y, z = repeate_cells(
        bottom_atoms[:, 0],
        bottom_atoms[:, 1],
        bottom_atoms[:, 2],
        range(-1, 2),
        vector1,
        vector2,
    )

    bottom_large = np.array([x, y, z]).T

    for pos in top_atoms:
        close = closest_index(pos, bottom_large)
        closest_particle = bottom_large[close]
        z_distance = abs(pos[2] - closest_particle[2])
        pos[2] = z_distance

    return top_atoms[:, 0], top_atoms[:, 1], top_atoms[:, 2]
