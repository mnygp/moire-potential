import numpy as np
from ase.io import read
import functions.geometry as f
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker


def contour_plot_fill(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    filename: str,
    title: str,
    color_map: str,
    label: str,
    strings=None,
    levels=100,
    cbar_limits=None,
    position=None,
):
    fig, ax = plt.subplots()

    if cbar_limits is not None and len(cbar_limits) == 2:
        # contour = ax.tricontourf(X, Y, Z, levels=levels, cmap=color_map,
        #                         vmin=cbar_limits[0], vmax=cbar_limits[1])
        vmin, vmax = cbar_limits
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Generate levels spanning the full color range
        lvl = np.linspace(vmin, vmax, levels)
        contour = ax.tricontourf(X, Y, Z, levels=lvl, cmap=color_map, norm=norm)
    else:
        contour = ax.tricontourf(X, Y, Z, levels=levels, cmap=color_map)

    cbar = fig.colorbar(contour, ax=ax, label=label)
    cbar.formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    cbar.update_ticks()
    ax.set_xlabel("X Position [Å]")
    ax.set_ylabel("Y Position [Å]")
    ax.axis("equal")
    ax.set_title(title)

    if strings is not None:
        text = "\n".join(strings)
        if position is not None and len(position) == 2:
            ax.text(
                position[0],
                position[1],
                text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )
        else:
            ax.text(
                0.03,
                0.17,
                text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )

    fig.savefig("plots-MatterSim/" + filename + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


path_to_structure = file_path = Path(__file__).resolve().parents[2]

# Parameters
structure = "1.11_2946"
structures = np.array(
    [
        "1.00_2967",
        "1.05_3027",
        "1.09_3093",
        "1.11_2940",
        "1.13_3165",
        "1.15_3012",
        "1.19_3078",
        "1.00_2973",
        "1.05_3033",
        "1.09_3099",
        "1.11_2946",
        "1.13_3171",
        "1.16_3000",
    ]
)

for structure in [structure]:
    twist_angle, natoms = structure.split("_")

    # Read the structure
    atoms = read(
        path_to_structure
        / str("structures/MoS2-WSe2-MatterSim/" + structure + "/structure_ml.json")
    )

    # Length of the vectors
    vector1, vector2 = atoms.cell[0], atoms.cell[1]  # type: ignore
    a = np.linalg.norm(vector1)
    b = np.linalg.norm(vector2)

    angle = np.round(atoms.cell.angles()[2], 2)  # type: ignore

    print(
        f"{natoms}: Vector ratio {max(a, b) / min(a, b):.2f}"
        + f" with internal angle {angle}"
    )

    text = [
        f"a={max(a, b):.2f}Å",
        f"b={min(a, b):.2f}Å",
        f"Twist angle: {twist_angle}°",
        f"Atoms: {natoms}",
    ]

    # Calculate the height of the top layer above the bottom layer
    top_layer_x, top_layer_y, height = f.height(atoms)
    top_layer_x, top_layer_y, height = f.repeate_cells(
        top_layer_x, top_layer_y, height, range(-1, 2), vector1, vector2
    )

    contour_plot_fill(
        top_layer_x,
        top_layer_y,
        height,
        "relative_height_map_" + structure,
        "Se height above bottom S layer",
        "viridis",
        "Height [Å]",
        strings=text,
    )

    # Calculate the horizontal distance between the two middle layers
    h_distance_x, h_distance_y, h_distance = f.horizontal_distance(atoms)
    h_distance_x, h_distance_y, h_distance = f.repeate_cells(
        h_distance_x, h_distance_y, h_distance, range(-1, 2), vector1, vector2
    )

    contour_plot_fill(
        h_distance_x,
        h_distance_y,
        h_distance,
        "horizontal_distance_map_" + structure,
        "Horizontal distance from Se to nearest S",
        "viridis",
        "Horizontal distance [Å]",
        strings=text,
    )

    # Calculate the strain in the W layer
    W_x, W_y, W_strain = f.strain(atoms, "W")
    W_x, W_y, W_strain = f.repeate_cells(
        W_x, W_y, W_strain, range(-1, 2), vector1, vector2
    )
    W_strain_text = text.copy()
    W_strain_text.append(f"Max strain: {max(W_strain) * 100:.2f} %")
    W_strain_text.append(f"Min strain: {min(W_strain) * 100:.2f} %")

    contour_plot_fill(
        W_x,
        W_y,
        W_strain * 100,
        "strain_map_W_" + structure,
        "Average deviation from ideal distance of W atoms",
        "viridis",
        "Average Displacement [%]",
        strings=W_strain_text,
        cbar_limits=(-0.25, 0.65),
        position=(0.03, 0.25),
    )

    # Calculate the strain in the Mo layer
    Mo_x, Mo_y, Mo_strain = f.strain(atoms, "Mo")
    Mo_x, Mo_y, Mo_strain = f.repeate_cells(
        Mo_x, Mo_y, Mo_strain, range(-1, 2), vector1, vector2
    )

    Mo_strain_text = text.copy()
    Mo_strain_text.append(f"Max strain: {max(Mo_strain) * 100:.2f} %")
    Mo_strain_text.append(f"Min strain: {min(Mo_strain) * 100:.2f} %")

    contour_plot_fill(
        Mo_x,
        Mo_y,
        Mo_strain * 100,
        "strain_map_Mo_" + structure,
        "Average deviation from ideal distance of Mo atoms",
        "viridis",
        "Average Displacement [%]",
        strings=Mo_strain_text,
        cbar_limits=(-0.25, 0.65),
        position=(0.03, 0.25),
    )

    # Calculate the thickness of the S layer
    S_x, S_y, S_thickness = f.layer_thicknsess(atoms, "S")
    S_x, S_y, S_thickness = f.repeate_cells(
        S_x, S_y, S_thickness, range(-1, 2), vector1, vector2
    )

    contour_plot_fill(
        S_x,
        S_y,
        S_thickness,
        "S_thickness_map_" + structure,
        "Thickness of the MoS2 layer",
        "viridis",
        "Thickness [Å]",
        strings=text,
    )

    # Calculate the thickness of the Se layer
    Se_x, Se_y, Se_thickness = f.layer_thicknsess(atoms, "Se")
    Se_x, Se_y, Se_thickness = f.repeate_cells(
        Se_x, Se_y, Se_thickness, range(-1, 2), vector1, vector2
    )

    contour_plot_fill(
        Se_x,
        Se_y,
        Se_thickness,
        "Se_thickness_map_" + structure,
        "Thickness of the WSe2 layer",
        "viridis",
        "Thickness [Å]",
        strings=text,
    )

    # Calculate the distance between the two layers
    inter_x, inter_y, inter_distance = f.interlayer_distance(atoms)
    inter_x, inter_y, inter_distance = f.repeate_cells(
        inter_x, inter_y, inter_distance, range(-1, 2), vector1, vector2
    )

    contour_plot_fill(
        inter_x,
        inter_y,
        inter_distance,
        "inter_distance_map_" + structure,
        "Distance between the W and Mo layer",
        "viridis",
        "Distance [Å]",
        strings=text,
    )

    shift_dict = f.get_shifts(atoms)
    diag_shifts = np.linalg.norm(shift_dict["shifts"], axis=1)
    x_shifts = shift_dict["shifts"][:, 0]
    y_shifts = shift_dict["shifts"][:, 1]

    plt.scatter(
        shift_dict["origins"][:, 0],
        shift_dict["origins"][:, 1],
        c=y_shifts,
        cmap="viridis",
        s=40,
    )
    plt.colorbar(label="Data value")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("2D Scatter Plot with y shift")
    plt.axis("equal")
    plt.grid()
    plt.savefig(
        "plots-MatterSim/scatter_map_y_shift_" + structure + ".png",
        dpi=300,
        bbox_inches="tight",
    )
