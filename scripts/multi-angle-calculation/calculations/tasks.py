from pathlib import Path
import taskblaster as tb
from ase.io import read
from ase import Atoms
from numpy.typing import NDArray
import numpy as np

from functions.geometry import strain, interlayer_distance, shifts_and_z
from functions.util import repeate_cells

from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


# TODO: Check if util function does the same
def get_root_path(root: str, target: str) -> str:
    current_path = Path(__file__).resolve()
    print(f"Current path: {current_path}")

    for parent in current_path.parents:
        if parent.name == root:
            full_path = Path(parent) / target.lstrip("/")
            full_path = full_path.resolve()
            print(f"Resolved structure path: {full_path}")
            return str(full_path)

    raise FileNotFoundError(
        f"Could not find a directory named {root} in {current_path}"
    )


def get_dirs() -> list[str]:
    dir = get_root_path("moire-potential", "structures/more-structures")
    # dir = '../../../'
    dir_arr = [str(p.resolve()) for p in Path(dir).iterdir() if p.is_dir()]
    return dir_arr


def read_atoms(p: str) -> Atoms:
    atoms = read(p + "/MatterSim_relaxed.json")
    return atoms


def get_geometry(atoms: Atoms):
    MoS2_strain = strain(atoms, "Mo")
    WSe2_strain = strain(atoms, "W")
    shifts_z = shifts_and_z(atoms)

    # --- Filter copied atoms to W only ---
    atoms_W = Atoms(
        f"W{len(shifts_z['x'])}",
        positions=np.zeros((len(shifts_z["x"]), 3)),
        pbc=True,
        cell=atoms.cell.copy(),
    )
    atoms_W.positions[:, 0] = shifts_z["x"]
    atoms_W.positions[:, 1] = shifts_z["y"]
    atoms_W.positions[:, 2] = 0.0
    atoms_W.wrap()

    # Write wrapped coordinates back
    shifts_z["x"] = atoms_W.positions[:, 0]
    shifts_z["y"] = atoms_W.positions[:, 1]

    W_coords = np.column_stack((WSe2_strain["x"], WSe2_strain["y"]))
    shift_coords = np.column_stack((shifts_z["x"], shifts_z["y"]))

    # Find indices in W_coords that match each shift_coords row
    indices = []
    for s in shift_coords:
        # Compute distances to all W_coords
        dist = np.linalg.norm(W_coords - s, axis=1)
        idx = np.argmin(dist)
        if dist[idx] < 1e-6:
            indices.append(idx)

    # Use indices to filter WSe2_strain
    filtered = {
        "x": WSe2_strain["x"][indices],
        "y": WSe2_strain["y"][indices],
        "strain": WSe2_strain["strain"][indices],
    }

    print(min(filtered["x"]), max(filtered["x"]))
    print(min(shifts_z["x"]), max(shifts_z["x"]))

    print("")
    print(min(filtered["y"]), max(filtered["y"]))
    print(min(shifts_z["y"]), max(shifts_z["y"]))

    print(len(filtered["y"]))
    print(len(shifts_z["y"]))

    return {
        "Mo_x": MoS2_strain["x"],
        "Mo_y": MoS2_strain["y"],
        "Mo_strain": MoS2_strain["strain"],
        "W_x": filtered["x"],
        "W_y": filtered["y"],
        "W_strain": filtered["strain"],
        "z_dist": shifts_z["z"],
        "shift v1": shifts_z["shift v1"],
        "shift v2": shifts_z["shift v2"],
    }


def strain_ref():
    # Create a function that takes a list of points in and in this function
    # Just parse [x] amd [y]
    return 1


def strain_correction(
    input_dict: dict[str, NDArray], ref: float, atoms: Atoms
) -> NDArray:
    Mo_x = input_dict["Mo_x"]
    Mo_y = input_dict["Mo_y"]
    Mo_strain = input_dict["Mo_strain"]
    W_x = input_dict["W_x"]
    W_y = input_dict["W_y"]
    W_strain = input_dict["W_strain"]

    (x_Mo_large, y_Mo_large, Mo_strain_large) = repeate_cells(
        Mo_x,
        Mo_y,
        Mo_strain,
        range(-1, 2),
        atoms.cell[0, :2],  # type: ignore
        atoms.cell[1, :2],  # type: ignore
    )
    ######## Load strain data here #######
    # TODO: Perhaps use SOC calculated strain correction???
    data_path = get_root_path("calculations", "band_edges_medium_soc.csv")
    data = np.genfromtxt(data_path, skip_header=1, dtype=float, delimiter=",")

    strain_data = data[:, 0] - 1
    MoS2_lumo = data[:, 2]
    WSe2_homo = data[:, 3]

    lumo_grid, homo_grid = np.meshgrid(MoS2_lumo, WSe2_homo)
    band_gap_grid = lumo_grid - homo_grid
    ref_gap = MoS2_lumo[-1] - WSe2_homo[0]
    band_gap_correction = (lumo_grid - homo_grid) - ref

    Mo_strain_interp = LinearNDInterpolator(
        list(zip(x_Mo_large, y_Mo_large)), Mo_strain_large
    )

    interp_Mo_strain = Mo_strain_interp(W_x, W_y)
    MoS2_grid, WSe2_grid = np.meshgrid(strain_data, strain_data)
    points = np.column_stack([MoS2_grid.ravel(), WSe2_grid.ravel()])  # shape (N,2)
    values = (band_gap_grid - ref).ravel()  # shape (N,)
    correction_interp = LinearNDInterpolator(points, values)

    corrections = correction_interp(list(zip(interp_Mo_strain, W_strain)))

    print(f"Length of correction array {len(corrections)}")
    return {"x": W_x, "y": W_y, "strain_corr": corrections}


def optimized_z_gap(input_dict: dict[str, NDArray]) -> NDArray:
    # data_path = get_root_path('multi-angle-cancluation', 'reference-values/optimized_z.npy')
    # data = np.load(data_path)
    data = np.ones((20, 20))

    data_flat = data.flatten()
    shift_arr_1 = np.linspace(0, 1, len(data[0, :]))
    shift_arr_2 = np.linspace(0, 1, len(data[:, 0]))
    shift_grid_1, shift_grid_2 = np.meshgrid(shift_arr_1, shift_arr_2, indexing="ij")
    points = np.column_stack([shift_grid_1.ravel(), shift_grid_2.ravel()])
    gap_interpolator = LinearNDInterpolator(points, data.ravel())
    gaps = gap_interpolator(list(zip(input_dict["shift v1"], input_dict["shift v2"])))
    print(f"Length of opt z gap array {len(gaps)}")
    return {"gap": gaps, "x": input_dict["W_x"], "y": input_dict["W_y"]}


def parameter_z_gap(input_dict: dict[str, NDArray]) -> NDArray:
    # data_path = get_root_path('multi-angle-cancluation', 'reference-values/parameter_z.npy')
    # data = np.load(data_path)
    data = np.ones((15, 20, 20))

    data_flat = data.flatten()
    z_arr = np.linspace(6.0, 6.9, len(data[:, 0, 0]))
    shift_arr_1 = np.linspace(0, 1, len(data[0, :, 0]))
    shift_arr_2 = np.linspace(0, 1, len(data[0, 0, :]))

    print(input_dict["z_dist"].min(), input_dict["z_dist"].max())
    print(input_dict["shift v1"].min(), input_dict["shift v1"].max())
    print(input_dict["shift v2"].min(), input_dict["shift v2"].max())

    z_grid, shift_grid_1, shift_grid_2 = np.meshgrid(
        z_arr, shift_arr_1, shift_arr_2, indexing="ij"
    )
    points = np.column_stack(
        [z_grid.ravel(), shift_grid_1.ravel(), shift_grid_2.ravel()]
    )
    gap_interpolator = LinearNDInterpolator(points, data.ravel())
    gaps = gap_interpolator(
        list(zip(input_dict["z_dist"], input_dict["shift v1"], input_dict["shift v2"]))
    )
    print(f"Length of param z gap array {len(gaps)}")
    return {"gap": gaps, "x": input_dict["W_x"], "y": input_dict["W_y"]}


def collect_gaps(gap: dict[str, NDArray], correction: dict[str, NDArray]) -> NDArray:
    gap_x_round = np.round(gap["x"], 5)
    gap_y_round = np.round(gap["y"], 5)

    corr_x_round = np.round(correction["x"], 5)
    corr_y_round = np.round(correction["y"], 5)

    gap_mask = np.where((gap_x_round == corr_x_round) & (gap_y_round == corr_y_round))

    return {
        "x": gap["x"][gap_mask],
        "y": gap["y"][gap_mask],
        "corr gap": gap["gap"][gap_mask] + correction["strain_corr"],
    }


@tb.dynamical_workflow_generator_task
def generate_wfs(paths: list[str], strain_ref_E: float):
    for p in paths:
        name = p.split("/")[-1]
        wf = Sub_wf(path=p, strain_ref=strain_ref_E)
        yield name, wf


@tb.workflow
class Sub_wf:
    path = tb.var()
    strain_ref = tb.var()

    @tb.task
    def get_atoms(self):
        return tb.node("read_atoms", p=self.path)

    @tb.task
    def geometry(self):
        return tb.node("get_geometry", atoms=self.get_atoms)

    @tb.task
    def strain_correction(self):
        return tb.node(
            "strain_correction",
            ref=self.strain_ref,
            input_dict=self.geometry,
            atoms=self.get_atoms,
        )

    @tb.task
    def gap_opt_z(self):
        return tb.node("optimized_z_gap", input_dict=self.geometry)

    @tb.task
    def gap_param_z(self):
        return tb.node("parameter_z_gap", input_dict=self.geometry)

    @tb.task
    def corrected_opt_gap(self):
        return tb.node(
            "collect_gaps", gap=self.gap_opt_z, correction=self.strain_correction
        )

    @tb.task
    def corrected_param_gap(self):
        return tb.node(
            "collect_gaps", gap=self.gap_param_z, correction=self.strain_correction
        )
