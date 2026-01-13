import numpy as np
from ase.io import read
import functions.homo_layer_geometry as f
from functions.plotting import contour_plot_fill
from pathlib import Path


folders = np.array(
    [
        "10.42_546",
        "13.17_114",
        "17.90_186",
        "5.09_762",
        "7.93_942",
        "10.99_654",
        "15.18_258",
        "18.73_906",
        "6.01_546",
        "8.61_798",
        "11.64_438",
        "16.43_294",
        "19.65_618",
        "7.34_366",
        "9.43_222",
    ]
)

folder_subset = np.array(["5.09_762", "7.93_942", "8.61_798", "10.99_654", "18.73_906"])

parent_path = file_path = Path(__file__).resolve().parents[3]

path_to_struct = Path(parent_path / "structures/WSe2-WSe2/")

for i in folder_subset:
    atoms = read(str(path_to_struct / i / "structure_ml.json"))
    twist_angle, natoms = i.split("_")

    top_strain_x, top_strain_y, top_strain = f.strain(atoms, "W", "top")
    print("Top strain done")
    bottom_strain_x, bottom_strain_y, bottom_strain = f.strain(atoms, "W", "bottom")
    print("Bottom strain done")
    interlayer_x, interlayer_y, interlayer_dist = f.interlayer_distance(atoms, "W")
    print("Interlayer distance done")
    interlayer_x, interlayer_y, interlayer_dist = f.interlayer_distance(atoms, "W")

    contour_plot_fill(
        top_strain_x,
        top_strain_y,
        top_strain * 100,
        "top_strain_" + i,
        "Top W layer strain",
        "RdGy_r",
        "Strain",
        strings=[f"Twist angle: {twist_angle}°", f"Atoms: {natoms}"],
    )

    contour_plot_fill(
        bottom_strain_x,
        bottom_strain_y,
        bottom_strain * 100,
        "bottom_strain_" + i,
        "Bottom W layer strain",
        "RdGy_r",
        "Strain",
        strings=[f"Twist angle: {twist_angle}°", f"Atoms: {natoms}"],
    )

    contour_plot_fill(
        interlayer_x,
        interlayer_y,
        interlayer_dist,
        "interlayer_distance_" + i,
        "Interlayer distance",
        "RdGy_r",
        "Distance [Å]",
        strings=[f"Twist angle: {twist_angle}°", f"Atoms: {natoms}"],
    )
