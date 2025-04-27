import numpy as np
from ase.io import read
import functions.heat_map as f
import matplotlib.pyplot as plt
from pathlib import Path


def contour_plot_fill(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                      filename: str, title: str, color_map: str,
                      label: str, strings=None, levels=100):

    fig, ax = plt.subplots()
    contour = ax.tricontourf(X, Y, Z, levels=levels, cmap=color_map)

    fig.colorbar(contour, ax=ax, label=label)
    ax.set_xlabel("X Position [Å]")
    ax.set_ylabel("Y Position [Å]")
    ax.axis('equal')
    ax.set_title(title)

    if (strings is not None):
        text = '\n'.join(strings)
        ax.text(0.03, 0.17, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top')

    fig.savefig("plots/" + filename + ".png", dpi=300,
                bbox_inches='tight')
    plt.close(fig)


def contour_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                 filename: str, title: str, color_map: str,
                 label: str, strings=None, levels=7):

    level_values = np.linspace(min(Z)+0.01, max(Z)-0.01, levels)

    fig, ax = plt.subplots()
    contour = ax.tricontour(X, Y, Z, level_values, cmap=color_map)

    fig.colorbar(contour, ax=ax, label=label)
    ax.set_xlabel("X Position [Å]")
    ax.set_ylabel("Y Position [Å]")
    ax.axis('equal')
    ax.set_title(title)

    if (strings is not None):
        text = '\n'.join(strings)
        ax.text(0.03, 0.17, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top')

    fig.savefig("plots/" + filename + ".png", dpi=300,
                bbox_inches='tight')
    plt.close(fig)


path_to_structure = file_path = Path(__file__).resolve().parents[2]

# Parameters
structure = "1.05_3027"
structures = np.array([
    "1.00_2967", "1.05_3027", "1.09_3093", "1.11_2940", "1.13_3165",
    "1.15_3012", "1.19_3078", "1.00_2973", "1.05_3033", "1.09_3099",
    "1.11_2946", "1.13_3171", "1.16_3000"])

number_of_structures = len(structures)
counter = 0

top_type = "Se"
n_cells = 3


for structure in [structure]:
    counter += 1
    twist_angle, natoms = structure.split("_")

    # Read the structure
    atoms = read(path_to_structure /
                 str("structures/" + structure + "/structure_ml.json"))

    # symbols = np.array(atoms.get_chemical_symbols())
    # positions = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell())
    vector1, vector2 = cell[0], cell[1]

    # Length of the vectors
    a = np.linalg.norm(vector1)
    b = np.linalg.norm(vector2)

    angle = np.arccos(np.dot(vector1, vector2)/(a*b))

    print(f"{natoms}: Vector ratio {max(a, b)/min(a, b):.2f}"
          + f" with internal angle {angle*180/np.pi:.2f}")

    text = [f"a={max(a, b):.2f}Å", f"b={min(a, b):.2f}Å",
            f"Twist angle: {twist_angle}°", f"Atoms: {natoms}"]

    # Calculate the height of the top layer above the bottom layer
    top_layer_x, top_layer_y, height = f.height(atoms)
    top_layer_x, top_layer_y, height = f.repeate_cells(top_layer_x,
                                                       top_layer_y,
                                                       height,
                                                       range(-1, 2),
                                                       vector1, vector2)

    contour_plot_fill(top_layer_x, top_layer_y, height,
                      "relative_height_map_" + structure,
                      "Se height above bottom S layer",
                      "copper", "Height [Å]", strings=text)

    # Calculate the horizontal distance between the two middle layers
    h_distance_x, h_distance_y, h_distance = f.horizontal_distance(atoms)
    h_distance_x, h_distance_y, h_distance = f.repeate_cells(h_distance_x,
                                                             h_distance_y,
                                                             h_distance,
                                                             range(-1, 2),
                                                             vector1, vector2)

    contour_plot_fill(h_distance_x, h_distance_y, h_distance,
                      "horizontal_distance_map_" + structure,
                      "Horizontal distance from Se to nearest S",
                      "RdGy", "Horizontal distance [Å]", strings=text)

    # Calculate the strain in the W layer
    W_x, W_y, W_strain = f.strain(atoms, 'W')
    W_x, W_y, W_strain = f.repeate_cells(W_x, W_y, W_strain, range(-1, 2),
                                         vector1, vector2)

    contour_plot_fill(W_x, W_y, W_strain*100,
                      "strain_map_W_" + structure,
                      "Average deviation from ideal distance of W atoms",
                      "RdGy_r", "Average Displacement [\%]", strings=text)

    # Calculate the strain in the Mo layer
    W_x, W_y, W_strain = f.strain(atoms, 'Mo')
    W_x, W_y, W_strain = f.repeate_cells(W_x, W_y, W_strain, range(-1, 2),
                                         vector1, vector2)

    contour_plot_fill(W_x, W_y, W_strain*100,
                      "strain_map_Mo_" + structure,
                      "Average deviation from ideal distance of Mo atoms",
                      "RdGy_r", "Average Displacement [\%]", strings=text)

    # Calculate the thickness of the S layer
    S_x, S_y, S_thickness = f.layer_thicknsess(atoms, "S")
    S_x, S_y, S_thickness = f.repeate_cells(S_x, S_y, S_thickness,
                                            range(-1, 2), vector1, vector2)

    contour_plot_fill(S_x, S_y, S_thickness,
                      "S_thickness_map_" + structure,
                      "Thickness of the MoS2 layer",
                      "RdGy", "Thickness [Å]", strings=text)

    # Calculate the thickness of the Se layer
    Se_x, Se_y, Se_thickness = f.layer_thicknsess(atoms, "Se")
    Se_x, Se_y, Se_thickness = f.repeate_cells(Se_x, Se_y, Se_thickness,
                                               range(-1, 2), vector1, vector2)

    contour_plot_fill(Se_x, Se_y, Se_thickness,
                      "Se_thickness_map_" + structure,
                      "Thickness of the WSe2 layer",
                      "RdGy", "Thickness [Å]", strings=text)

    # Calculate the distance between the two layers
    inter_x, inter_y, inter_distance = f.interlayer_distance(atoms)
    inter_x, inter_y, inter_distance = f.repeate_cells(inter_x, inter_y,
                                                       inter_distance,
                                                       range(-1, 2),
                                                       vector1, vector2)

    contour_plot_fill(inter_x, inter_y, inter_distance,
                      "inter_distance_map_" + structure,
                      "Distance between the W and Mo layer",
                      "RdGy_r", "Distance [Å]", strings=text)

    contour_plot(inter_x, inter_y, inter_distance,
                 "inter_distance_map_contour_" + structure,
                 "Distance between the W and Mo layer",
                 "gist_earth", "Distance [Å]", strings=text,
                 levels=5)    

    contour_plot(W_x, W_y, W_strain*100,
                 "strain_map_Mo_contour_" + structure,
                 "Average deviation from ideal distance of Mo atoms",
                 "gist_earth", "Average Displacement [%]",
                 strings=text, levels=5)

    # Calculate the thickness of the Se layer
    Mo_x, Mo_y, Mo_mod_shift = f.modified_h_dist(atoms)
    # Mo_x, Mo_y, Mo_mod_shift = f.repeate_cells(Se_x, Se_y, Se_thickness,
    #                                           range(-1, 2), vector1, vector2)

    contour_plot_fill(Mo_x, Mo_y, Mo_mod_shift,
                      "Mo_modified_shift_map_" + structure,
                      "Amount of shift",
                      "RdGy", "Shift procentage", strings=text)

    print(f"({counter}/{number_of_structures}): {structure}"
          + " calculation completed. Strain"
          + f" range [{min(W_strain):.3f},{max(W_strain):.3f}]")
