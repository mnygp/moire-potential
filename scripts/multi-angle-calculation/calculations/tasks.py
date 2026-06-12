from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import taskblaster as tb
from ase import Atoms
from ase.io import read
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from functions.bandstructure import LDOS, scissors_gpw_file
from functions.finite_difference import diag_hamiltonian
from functions.geometry import shifts_and_z, strain
from functions.util import repeate_cells


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
    atoms = read(p + "/MatterSim_relaxed_high_fid.json")
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

    print(len(filtered["y"]))
    print(len(shifts_z["y"][indices]))

    print(len(WSe2_strain["x"]), len(shifts_z["z"]))
    print(len(indices))

    return {
        "Mo_x": MoS2_strain["x"],
        "Mo_y": MoS2_strain["y"],
        "Mo_strain": MoS2_strain["strain"],
        "W_x": filtered["x"],
        "W_y": filtered["y"],
        "W_strain": filtered["strain"],
        "z_dist": shifts_z["z"][indices],
        "shift v1": shifts_z["shift v1"][indices],
        "shift v2": shifts_z["shift v2"][indices],
    }


def strain_corr(input_dict: dict[str, NDArray], atoms: Atoms) -> NDArray:
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
    # ####### Load strain data here #######
    data_path = get_root_path("calculations", "band_edges_medium_soc.csv")
    data = np.genfromtxt(data_path, skip_header=1, dtype=float, delimiter=",")

    strain_data = data[:, 0] - 1
    MoS2_lumo = data[:, 2]
    WSe2_homo = data[:, 3]

    lumo_grid, homo_grid = np.meshgrid(MoS2_lumo, WSe2_homo)
    band_gap_grid = lumo_grid - homo_grid
    ref_gap = MoS2_lumo[-1] - WSe2_homo[0]

    Mo_strain_interp = LinearNDInterpolator(
        list(zip(x_Mo_large, y_Mo_large)), Mo_strain_large
    )

    interp_Mo_strain = Mo_strain_interp(W_x, W_y)
    MoS2_grid, WSe2_grid = np.meshgrid(strain_data, strain_data)
    points = np.column_stack([MoS2_grid.ravel(), WSe2_grid.ravel()])
    values = (band_gap_grid - ref_gap).ravel()  # shape (N,)
    correction_interp = LinearNDInterpolator(points, values)

    corrections = correction_interp(list(zip(interp_Mo_strain, W_strain)))

    return {"x": W_x, "y": W_y, "strain_corr": corrections}


def optimized_z_gap(input_dict: dict[str, NDArray]) -> NDArray:
    raw = np.genfromtxt(
        get_root_path(
            "calculations",
            "optimized_z_gaps_0_005.csv",
        ),
        delimiter=",",
        names=True,
    )
    gap = raw["gap"]
    shift1 = raw["shift_1"]
    shift2 = raw["shift_2"]
    print(max(shift1), max(shift2))

    shift1_vals = np.unique(shift1)
    shift2_vals = np.unique(shift2)

    order = np.lexsort((shift2, shift1))

    # reshape with original sizes
    gap_grid = gap[order].reshape(len(shift1_vals), len(shift2_vals))

    # Append 1.0 to shift arrays to match padded grid
    shift1_vals = np.append(shift1_vals, 1.0)
    shift2_vals = np.append(shift2_vals, 1.0)

    # To be removed once i have the better data set
    gap_grid = np.pad(gap_grid, ((0, 1), (0, 1)), mode="wrap")
    interp = RegularGridInterpolator(
        (shift1_vals, shift2_vals),
        gap_grid,
        bounds_error=False,
        fill_value=np.nan,
    )

    gaps = interp(
        np.column_stack(
            [
                input_dict["shift v1"],
                input_dict["shift v2"],
            ]
        )
    )

    return {
        "gap": gaps,
        "x": input_dict["W_x"],
        "y": input_dict["W_y"],
    }


def parameter_z_gap(input_dict: dict[str, NDArray]) -> NDArray:
    # data_path = get_root_path('multi-angle-cancluation',
    #                           'reference-values/parameter_z.npy')
    # data = np.load(data_path)
    data = np.ones((15, 20, 20))

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


def finite_diff(
    atoms: Atoms, input_dict: dict[str, NDArray], N_grid: int
) -> dict[str, NDArray]:
    v1 = atoms.cell[0, :2]

    x = input_dict["x"]
    y = input_dict["y"]
    gap = input_dict["corr gap"]

    mask = ~np.isnan(gap)
    x = x[mask]
    y = y[mask]
    gap = gap[mask]

    cell_2d = atoms.cell[:2, :2]  # * (87/np.linalg.norm(v1))

    x_L, y_L, gap_L = repeate_cells(
        x, y, gap, range(-1, 2), atoms.cell[0, :2], atoms.cell[1, :2]
    )
    gap_L -= min(gap_L)
    gap_interp = LinearNDInterpolator(list(zip(x_L, y_L)), gap_L)

    print(f"Gap minimum=0 and maximum={max(gap_L * 1000):.2f}")

    # N_grid = 100
    dr = np.linalg.norm(v1) / N_grid
    m = 1.15

    # Initialize grid and initial guess
    x_lin = np.linspace(0, 1, N_grid, endpoint=False)
    y_lin = np.linspace(0, 1, N_grid, endpoint=False)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

    # Generate potential grid
    points = np.column_stack((X.ravel(), Y.ravel()))
    real_points = points @ cell_2d
    # real_points = points @ ([[1, 0], [0, 1]])
    V_flat = gap_interp(real_points)

    V = V_flat.reshape((N_grid, N_grid))

    eigvals, eigvecs = diag_hamiltonian(V, m, dr, hexagonal=True, order=2)
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    print(eigvals * 1000)
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "points": real_points,
        "pot": V_flat,
        "N_grid": N_grid,
    }


def z_diff(geometry_dict: dict[str, NDArray]) -> dict[str, NDArray]:
    x = geometry_dict["W_x"]
    y = geometry_dict["W_y"]
    s1_geo = geometry_dict["shift v1"]
    s2_geo = geometry_dict["shift v2"]
    z_geo = geometry_dict["z_dist"]

    raw = np.genfromtxt(
        get_root_path(
            "calculations",
            "optimized_z_gaps_0_005.csv",
        ),
        delimiter=",",
        names=True,
    )
    # gap = raw["gap"]
    shift1 = raw["shift_1"]
    shift2 = raw["shift_2"]
    opt_z = raw["z"]

    points = np.column_stack((shift1, shift2))
    z_interp_func = LinearNDInterpolator(points, opt_z)

    # Evaluate interpolated z at each geometry shift pair
    z_interp = z_interp_func(np.column_stack((s1_geo, s2_geo)))

    # Difference
    diff = z_geo - z_interp

    return {"x": x, "y": y, "z_geo": z_geo, "z_interp": z_interp, "diff": diff}


def LCAO_PDOS(gpw_file: Path | str) -> dict:
    Mo_data = LDOS(symbol="Mo", gpw_file=gpw_file)
    W_data = LDOS(symbol="W", gpw_file=gpw_file)
    return {"Mo": Mo_data, "W": W_data}


@tb.dynamical_workflow_generator_task
def generate_wfs(paths: list[str]):
    for p in paths:
        name = p.split("/")[-1]
        wf = Sub_wf(path=p)
        yield name, wf


@tb.workflow
class Sub_wf:
    path = tb.var()

    @tb.task
    def get_atoms(self):
        return tb.node("read_atoms", p=self.path)

    @tb.task
    def geometry(self):
        return tb.node("get_geometry", atoms=self.get_atoms)

    @tb.task
    def strain_correction(self):
        return tb.node(
            "strain_corr",
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

    @tb.task
    def fd_opt(self):
        return tb.node(
            "finite_diff",
            input_dict=self.corrected_opt_gap,
            atoms=self.get_atoms,
            N_grid=100,
        )

    @tb.task
    def compare_z(self):
        return tb.node("z_diff", geometry_dict=self.geometry)

    @tb.task
    def LCAO_gpw(self):
        return tb.node(
            "scissors_gpw_file",
            atom_path=self.get_atoms,
            kpts_dens=4,
            gpw_file="MoS2WSe2",
        )

    @tb.task
    def LCAO_projection(self):
        return tb.node("LCAO_PDOS", gpw_file=self.LCAO_gpw)


# ################## Plotting functions #####################
def plot_z(input, atoms):
    angle = []
    natoms = []
    max_z = []
    min_z = []
    atoms_arr = []

    for task_name, a in atoms.items():
        atoms_arr.append(a)

    for task_name, d in input.items():
        x = d["W_x"]
        y = d["W_y"]
        z_dist = d["z_dist"]

        print(min(y), max(y))

        angle_natoms = task_name.split("/")[0]
        min_z.append(min(z_dist))
        max_z.append(max(z_dist))
        angle.append(angle_natoms.split("_")[0])
        natoms.append(angle_natoms.split("_")[1])

        # Extract in-plane lattice vectors
        a = atoms[angle_natoms + "/get_atoms"]
        v1 = a.cell[0, :2]
        v2 = a.cell[1, :2]

        x_L, y_L, z_L = repeate_cells(x, y, z_dist, range(-1, 2), v1, v2)
        interp = LinearNDInterpolator(np.column_stack((x_L, y_L)), z_L)

        # Scatter plots
        X, Y = np.meshgrid(
            np.linspace(min(x_L), max(x_L), 600), np.linspace(min(y_L), max(y_L), 600)
        )
        interpolated_z = interp(X, Y)

        plt.figure(figsize=(6, 5))

        v_min = np.nanmin((interpolated_z))
        v_max = np.nanmax((interpolated_z))

        im = plt.imshow(
            (interpolated_z),
            origin="lower",
            extent=[np.min(x_L), np.max(x_L), np.min(y_L), np.max(y_L)],
            cmap="viridis",
            aspect="equal",
            vmin=v_min,
            vmax=v_max,
        )

        plt.colorbar(im, label="Z dist")
        plt.title(f"Interlayer distance for twist angle {angle_natoms.split('_')[0]}")
        plt.xlabel("x [Å]")
        plt.ylabel("y [Å]")
        plt.tight_layout()
        plt.savefig(f"z_{angle_natoms}.png", dpi=500)
        plt.close()

    # Max min plot
    fig, ax_bottom = plt.subplots(figsize=(8, 5))

    # Bottom x-axis: angle
    ax_bottom.plot(angle, min_z, "-o", label="min z")
    ax_bottom.plot(angle, max_z, "-o", label="max z")
    ax_bottom.set_xlabel("Angle")
    ax_bottom.set_ylabel("z distance")
    ax_bottom.grid(True)

    ax_bottom.set_xticks(range(len(angle)))
    ax_bottom.set_xticklabels(angle, rotation=90)

    # Top x-axis: natoms
    ax_top = ax_bottom.twiny()
    ax_top.set_xlim(ax_bottom.get_xlim())
    ax_top.set_xticks(range(len(natoms)))
    ax_top.set_xticklabels(natoms, rotation=90)
    ax_top.set_xlabel("Number of atoms")

    ax_bottom.legend()
    plt.tight_layout()
    plt.savefig("z_dist_angle_natoms.png", dpi=500)
    plt.close()


def plot_strain(input, atoms):
    angle = []
    natoms = []
    max_Mo_strain = []
    min_Mo_strain = []
    max_W_strain = []
    min_W_strain = []
    atoms_arr = []

    for task_name, a in atoms.items():
        atoms_arr.append(a)

    for task_name, d in input.items():
        Mo_x = d["Mo_x"]
        Mo_y = d["Mo_y"]
        Mo_strain = d["Mo_strain"]
        W_x = d["W_x"]
        W_y = d["W_y"]
        W_strain = d["W_strain"]

        print("Mo strain")
        print(np.round((min(Mo_strain), max(Mo_strain)), 5))

        print("W strain")
        print(np.round((min(W_strain), max(W_strain)), 5))

        angle_natoms = task_name.split("/")[0]
        angle.append(angle_natoms.split("_")[0])
        natoms.append(angle_natoms.split("_")[1])

        max_Mo_strain.append(max(Mo_strain))
        min_Mo_strain.append(min(Mo_strain))
        max_W_strain.append(max(W_strain))
        min_W_strain.append(min(W_strain))

        # Extract in-plane lattice vectors
        a = atoms[angle_natoms + "/get_atoms"]
        v1 = a.cell[0, :2]
        v2 = a.cell[1, :2]

        fig, (ax_Mo, ax_W) = plt.subplots(1, 2, figsize=(12, 5))

        # --- MO ---
        Mo_x_L, Mo_y_L, Mo_strain_L = repeate_cells(
            Mo_x, Mo_y, np.array(Mo_strain) * 100, range(-1, 2), v1, v2
        )
        Mo_interp = LinearNDInterpolator(np.column_stack((Mo_x_L, Mo_y_L)), Mo_strain_L)

        X_Mo, Y_Mo = np.meshgrid(
            np.linspace(Mo_x_L.min(), Mo_x_L.max(), 600),
            np.linspace(Mo_y_L.min(), Mo_y_L.max(), 600),
        )

        Mo_field = Mo_interp(X_Mo, Y_Mo)

        # --- W ---
        W_x_L, W_y_L, W_strain_L = repeate_cells(
            W_x, W_y, np.array(W_strain) * 100, range(-1, 2), v1, v2
        )
        W_interp = LinearNDInterpolator(np.column_stack((W_x_L, W_y_L)), W_strain_L)

        X_W, Y_W = np.meshgrid(
            np.linspace(W_x_L.min(), W_x_L.max(), 600),
            np.linspace(W_y_L.min(), W_y_L.max(), 600),
        )

        W_field = W_interp(X_W, Y_W)

        # Shared color limits
        vmin = np.nanmin([Mo_field, W_field])
        vmax = np.nanmax([Mo_field, W_field])

        # Plot MO
        im0 = ax_Mo.imshow(
            Mo_field,
            origin="lower",
            extent=[Mo_x_L.min(), Mo_x_L.max(), Mo_y_L.min(), Mo_y_L.max()],
            cmap="viridis",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax_Mo.set_title(f"Mo strain for twist angle {angle_natoms.split('_')[0]}")
        ax_Mo.set_xlabel("x [Å]")
        ax_Mo.set_ylabel("y [Å]")

        # Plot W
        im1 = ax_W.imshow(
            W_field,
            origin="lower",
            extent=[W_x_L.min(), W_x_L.max(), W_y_L.min(), W_y_L.max()],
            cmap="viridis",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax_W.set_title(f"W strain for twist angle {angle_natoms.split('_')[0]}")
        ax_W.set_xlabel("x [Å]")

        # One shared colorbar
        cbar = fig.colorbar(im0, ax=[ax_Mo, ax_W], shrink=0.85)
        cbar.set_label("Strain")

        fig.tight_layout()
        fig.savefig(f"strain_{angle_natoms}.png", dpi=500)
        plt.close(fig)

    # Compute differences
    diff_Mo = np.array(max_Mo_strain) - np.array(min_Mo_strain)
    diff_W = np.array(max_W_strain) - np.array(min_W_strain)

    # Create figure with two rows (top bigger)
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

    # --- Top panel: max/min strains ---
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(
        angle, np.array(max_Mo_strain) * 100, "-o", label="Max Mo strain", color="C0"
    )
    ax_top.plot(
        angle, np.array(min_Mo_strain) * 100, "-^", label="Min Mo strain", color="C0"
    )
    ax_top.plot(
        angle, np.array(max_W_strain) * 100, "-o", label="Max W strain", color="C1"
    )
    ax_top.plot(
        angle, np.array(min_W_strain) * 100, "-^", label="Min W strain", color="C1"
    )

    ax_top.set_ylabel("Strain [%]")
    ax_top.grid(True)

    # Top x-axis: number of atoms
    ax_top_tw = ax_top.twiny()
    ax_top_tw.set_xlim(ax_top.get_xlim())
    ax_top_tw.set_xticks(range(len(natoms)))
    ax_top_tw.set_xticklabels(natoms, rotation=90)
    ax_top_tw.set_xlabel("Number of atoms")

    ax_top.legend()

    # --- Bottom panel: differences ---
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bottom.plot(angle, diff_Mo * 100, "-o", color="C0", label="ΔMo strain")
    ax_bottom.plot(angle, diff_W * 100, "-o", color="C1", label="ΔW strain")
    ax_bottom.set_ylabel("ΔStrain [%]")
    ax_bottom.set_xlabel("Angle")
    ax_bottom.grid(True)
    ax_bottom.legend()
    ax_bottom.set_xticks(range(len(angle)))
    ax_bottom.set_xticklabels(angle, rotation=90)
    # Hide top x-axis labels on top panel to avoid overlap
    plt.setp(ax_top.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig("strain_angle_natoms_with_diff.png", dpi=500)
    plt.close()


def plot_strain_correction(input: dict[str, NDArray], atoms):
    angle = []
    natoms = []
    min_strain_corr = []
    max_strain_corr = []
    atoms_arr = []

    for task_name, a in atoms.items():
        atoms_arr.append(a)

    for task_name, d in input.items():
        x = d["x"]
        y = d["y"]
        strain_corr = np.array(d["strain_corr"]) * 1000

        print("Strain correction [meV]")
        print(np.round((min(strain_corr), max(strain_corr)), 5))

        angle_natoms = task_name.split("/")[0]
        angle.append(angle_natoms.split("_")[0])
        natoms.append(angle_natoms.split("_")[1])

        max_strain_corr.append(max(strain_corr))
        min_strain_corr.append(min(strain_corr))

        # Extract in-plane lattice vectors
        a = atoms[angle_natoms + "/get_atoms"]
        v1 = a.cell[0, :2]
        v2 = a.cell[1, :2]

        x_L, y_L, strain_L = repeate_cells(x, y, strain_corr, range(-1, 2), v1, v2)
        interp = LinearNDInterpolator(np.column_stack((x_L, y_L)), strain_L)

        # Scatter plots
        X, Y = np.meshgrid(
            np.linspace(min(x_L), max(x_L), 600), np.linspace(min(y_L), max(y_L), 600)
        )
        interpolated_strain = interp(X, Y)

        plt.figure(figsize=(6, 5))

        v_min = np.nanmin((interpolated_strain))
        v_max = np.nanmax((interpolated_strain))

        im = plt.imshow(
            (interpolated_strain),
            origin="lower",
            extent=[np.min(x_L), np.max(x_L), np.min(y_L), np.max(y_L)],
            cmap="viridis",
            aspect="equal",
            vmin=v_min,
            vmax=v_max,
        )

        plt.colorbar(im, label="Strain correction [meV]")
        plt.title(f"Interlayer distance for twist angle {angle_natoms.split('_')[0]}")
        plt.xlabel("x [Å]")
        plt.ylabel("y [Å]")
        plt.tight_layout()
        plt.savefig(f"strain_{angle_natoms}.png", dpi=500)
        plt.close()

    # Convert to numpy arrays
    max_strain_corr = np.array(max_strain_corr)
    min_strain_corr = np.array(min_strain_corr)

    # --- Two-panel plot for max/min + difference ---
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

    # Top panel: max/min strain correction
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(angle, min_strain_corr, "-o", label="Min strain")
    ax_top.plot(angle, max_strain_corr, "-o", label="Max strain")
    ax_top.set_ylabel("Strain correction [meV]")
    ax_top.grid(True)

    # Top x-axis: number of atoms
    ax_top_tw = ax_top.twiny()
    ax_top_tw.set_xlim(ax_top.get_xlim())
    ax_top_tw.set_xticks(range(len(natoms)))
    ax_top_tw.set_xticklabels(natoms, rotation=90)
    ax_top_tw.set_xlabel("Number of atoms")

    ax_top.legend()

    # Bottom panel: difference
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bottom.plot(angle, max_strain_corr - min_strain_corr, "-o", color="red")
    ax_bottom.set_ylabel("ΔStrain [meV]")
    ax_bottom.set_xlabel("Angle")
    ax_bottom.grid(True)
    ax_bottom.set_xticks(range(len(angle)))
    ax_bottom.set_xticklabels(angle, rotation=90)

    # Hide top x-axis labels
    plt.setp(ax_top.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig("strain_correction_angle_natoms_with_diff.png", dpi=500)
    plt.close()


def plot_gap(input, atoms):
    angle = []
    natoms = []
    max_gap = []
    min_gap = []
    atoms_arr = []

    for task_name, a in atoms.items():
        atoms_arr.append(a)

    for task_name, d in input.items():
        x = d["x"]
        y = d["y"]
        gap = np.array(d["corr gap"]) * 1000

        mask = ~np.isnan(gap)
        x = x[mask]
        y = y[mask]
        gap = gap[mask]

        print(task_name.split("/")[0])
        print(np.nanmin(gap), np.nanmax(gap))

        angle_natoms = task_name.split("/")[0]
        min_gap.append(np.nanmin(gap))
        max_gap.append(np.nanmax(gap))
        # angle.append(float(angle_natoms.split("_")[0]))
        angle.append(angle_natoms.split("_")[0])
        natoms.append(angle_natoms.split("_")[1])

        # Extract in-plane lattice vectors
        a = atoms[angle_natoms + "/get_atoms"]
        v1 = a.cell[0, :2]
        v2 = a.cell[1, :2]

        x_L, y_L, gap_L = repeate_cells(x, y, gap, range(-1, 2), v1, v2)
        interp = LinearNDInterpolator(np.column_stack((x_L, y_L)), gap_L)

        # Scatter plots
        X, Y = np.meshgrid(
            np.linspace(min(x_L), max(x_L), 600), np.linspace(min(y_L), max(y_L), 600)
        )
        interpolated_gap = interp(X, Y)

        angle = [float(a) for a in angle]

        plt.figure(figsize=(6, 5))

        v_min = np.nanmin((interpolated_gap))
        v_max = np.nanmax((interpolated_gap))

        im = plt.imshow(
            (interpolated_gap),
            origin="lower",
            extent=[np.min(x_L), np.max(x_L), np.min(y_L), np.max(y_L)],
            cmap="viridis",
            aspect="equal",
            vmin=v_min,
            vmax=v_max,
        )

        plt.colorbar(im, label="Band gap")
        plt.title(
            f"Strain corrected band gap for twist angle {angle_natoms.split('_')[0]}"
        )
        plt.xlabel("x [Å]")
        plt.ylabel("y [Å]")
        plt.tight_layout()
        plt.savefig(f"gap_{angle_natoms}.png", dpi=500)
        plt.close()

    # Max min plot
    max_gap = np.array(max_gap)
    min_gap = np.array(min_gap)
    # Create figure with two rows, first row bigger
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

    # Top panel: max/min gaps
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(angle, min_gap, "-o", label="Min gap")
    ax_top.plot(angle, max_gap, "-o", label="Max gap")
    ax_top.set_ylabel("Gap [meV]")
    ax_top.grid(True)

    # Bottom panel: gap difference
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bottom.plot(angle, max_gap - min_gap, "-o", color="red")
    ax_bottom.set_ylabel("ΔGap [meV]")
    ax_bottom.set_xlabel("Angle")
    ax_bottom.grid(True)

    plt.tight_layout()
    plt.savefig("gap_angle_with_diff_even_space.png", dpi=500)
    plt.close()


def plot_energy(gap_input, fd_input):
    angle = []
    natoms = []
    fd_energy = []
    min_gap = []

    for task_name, a in fd_input.items():
        fd_energy.append(a["eigvals"][0] * 1000)

    for task_name, d in gap_input.items():
        x = d["x"]
        y = d["y"]
        gap = np.array(d["corr gap"]) * 1000

        mask = ~np.isnan(gap)
        x = x[mask]
        y = y[mask]
        gap = gap[mask]

        print(task_name.split("/")[0])
        print(np.nanmin(gap), np.nanmax(gap))

        angle_natoms = task_name.split("/")[0]
        min_gap.append(np.nanmin(gap))
        # fd_energy.append(fd_input[angle_natoms]["eigvals"][0] * 1000)
        angle.append(float(angle_natoms.split("_")[0]))
        # angle.append(angle_natoms.split("_")[0])
        natoms.append(angle_natoms.split("_")[1])

    # Max min plot
    min_gap = np.array(min_gap)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )

    # Plot all curves on both axes
    for ax in (ax_top, ax_bot):
        ax.plot(angle, min_gap + fd_energy, "-o", label="FD + Min gap")
        ax.plot(angle, min_gap, "-o", label="Min gap")
        ax.plot(angle, fd_energy, "-o", label="FD")
        ax.grid(True)

    # Set y-limits (adjust numbers to your data)

    ax_top.set_ylim(1100, 1250)
    ax_bot.set_ylim(0, 75)

    # Hide spines between axes
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    # Diagonal break marks
    d = 0.015
    kwargs = dict(color="k", clip_on=False)

    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)

    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)

    # Labels and legend
    ax_bot.set_xlabel("Angle")
    ax_top.set_ylabel("Gap [meV]")
    ax_top.legend()

    plt.tight_layout()
    plt.savefig("energy.png", dpi=500)
    plt.close()

    # Simple plot to comapre with experimentalists
    plt.plot(angle, (min_gap + fd_energy) / 1000, "-o", label="FD + Min gap")
    plt.ylim(bottom=0.9)
    plt.xlabel("Angle")
    plt.ylabel("Gap [eV]")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("energy_single.png", dpi=500)
    plt.close()


def plot_wavefunction(input, atoms):
    angle = []
    natoms = []
    eigvals_4 = []
    atoms_arr = []

    for task_name, a in atoms.items():
        atoms_arr.append(a)

    for task_name, d in input.items():
        angle_natoms = task_name.split("/")[0]
        angle.append(angle_natoms.split("_")[0])
        natoms.append(angle_natoms.split("_")[1])

        print(d["eigvals"])
        eigvals_4.append(d["eigvals"][:4])  # meV
        # print(eigvals_4)
        # Extract in-plane lattice vectors
        # a = atoms[angle_natoms + "/get_atoms"]
        # v1 = a.cell[0, :2]
        # v2 = a.cell[1, :2]

        N = d["N_grid"]
        V = d["pot"].reshape((N, N))
        # psi = d["eigvecs"][:, :3].reshape((N, N, 3))

        pts = d["points"]  # (N_grid^2, 2)
        V = d["pot"]  # (N_grid^2,)
        eigvecs = d["eigvecs"]  # (N_grid^2, N_states)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()  # flatten to 1D array for easy indexing

        # --- Potential ---
        im0 = axs[0].tricontourf(
            pts[:, 0],
            pts[:, 1],
            V,
            levels=100,
            cmap="viridis",
        )
        axs[0].set_title("Potential")
        axs[0].set_aspect("equal")
        fig.colorbar(im0, ax=axs[0], shrink=0.8)

        # --- Lowest 3 eigenstates ---
        for i in range(3):
            im = axs[i + 1].tricontourf(
                pts[:, 0],
                pts[:, 1],
                np.abs(eigvecs[:, i]) ** 2,
                levels=100,
                cmap="viridis",
            )
            axs[i + 1].set_title(f"|ψ{i}|²")
            axs[i + 1].set_aspect("equal")
            fig.colorbar(im, ax=axs[i + 1], shrink=0.8)

        fig.tight_layout()

        angle_natoms = task_name.split("/")[0]
        fig.suptitle(angle_natoms)

        fig.tight_layout()
        fig.savefig(f"eigstates_realspace_{angle_natoms}.png", dpi=500)
        plt.close(fig)

    eigvals_4 = np.array(eigvals_4)
    fig, ax_bottom = plt.subplots(figsize=(8, 5))

    for i in range(4):
        ax_bottom.plot(eigvals_4[:, i], "-o", label=f"E_{i}")

    ax_bottom.set_xlabel("Angle")
    ax_bottom.set_ylabel("Energy [meV]")
    ax_bottom.grid(True)

    ax_bottom.set_xticks(range(len(angle)))
    ax_bottom.set_xticklabels(angle, rotation=90)

    ax_top = ax_bottom.twiny()
    ax_top.set_xlim(ax_bottom.get_xlim())
    ax_top.set_xticks(range(len(natoms)))
    ax_top.set_xticklabels(natoms, rotation=90)
    ax_top.set_xlabel("Number of atoms")

    ax_bottom.legend()
    plt.tight_layout()
    plt.savefig("eigvals_angle_natoms.png", dpi=500)
    plt.close()


def plot_z_diff(input):
    angle = []
    natoms = []
    z_diff_avg = []
    z_diff_std = []

    for task_name, d in input.items():
        z_diff = d["diff"]
        z_diff = z_diff[~np.isnan(z_diff)]

        angle_natoms = task_name.split("/")[0]
        angle.append(float(angle_natoms.split("_")[0]))
        natoms.append(angle_natoms.split("_")[1])

        z_diff_avg.append(np.mean(z_diff))
        z_diff_std.append(np.std(z_diff))

    print(z_diff_avg)
    z_diff_avg = np.array(z_diff_avg)
    z_diff_std = np.array(z_diff_std)

    # Max min plot
    fig, ax_bottom = plt.subplots(figsize=(8, 5))

    # Bottom x-axis: angle
    ax_bottom.plot(angle, z_diff_avg, "-o", label="Avg. z diff", color="blue")
    ax_bottom.fill_between(
        angle, z_diff_avg - z_diff_std, z_diff_avg + z_diff_std, alpha=0.2, color="blue"
    )
    ax_bottom.set_xlabel("Angle")
    ax_bottom.set_ylabel("z difference [Å]")
    ax_bottom.grid(True)

    # ax_bottom.set_xticks(range(len(angle)))
    # ax_bottom.set_xticklabels(angle, rotation=90)

    # Top x-axis: natoms
    """ax_top = ax_bottom.twiny()
    ax_top.set_xlim(ax_bottom.get_xlim())
    ax_top.set_xticks(range(len(natoms)))
    ax_top.set_xticklabels(natoms, rotation=90)
    ax_top.set_xlabel("Number of atoms")
    """
    ax_bottom.legend()
    plt.tight_layout()
    plt.savefig("z_diff_angle_natoms.png", dpi=500)
    plt.close()


def plot_local_gap(input, atoms):
    angle = []
    natoms = []
    min_gap = []
    max_gap = []

    for task_name, d in input.items():
        angle_natoms = task_name.split("/")[0]
        a = atoms[angle_natoms + "/get_atoms"]
        v1 = a.cell[0, :2]
        v2 = a.cell[1, :2]

        # Mo LUMO field
        x_Mo = np.asarray(d["Mo"]["x"])
        y_Mo = np.asarray(d["Mo"]["y"])
        lumo_Mo = np.asarray(d["Mo"]["lumo"])
        x_Mo_L, y_Mo_L, lumo_L = repeate_cells(
            x_Mo, y_Mo, lumo_Mo, range(-1, 2), v1, v2
        )
        interp_lumo = LinearNDInterpolator(np.column_stack((x_Mo_L, y_Mo_L)), lumo_L)

        # W HOMO field
        x_W = np.asarray(d["W"]["x"])
        y_W = np.asarray(d["W"]["y"])
        homo_W = np.asarray(d["W"]["homo"])
        x_W_L, y_W_L, homo_L = repeate_cells(x_W, y_W, homo_W, range(-1, 2), v1, v2)
        interp_homo = LinearNDInterpolator(np.column_stack((x_W_L, y_W_L)), homo_L)

        # Common grid covering both tiled point clouds
        x_min = min(np.min(x_Mo_L), np.min(x_W_L))
        x_max = max(np.max(x_Mo_L), np.max(x_W_L))
        y_min = min(np.min(y_Mo_L), np.min(y_W_L))
        y_max = max(np.max(y_Mo_L), np.max(y_W_L))
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, 600),
            np.linspace(y_min, y_max, 600),
        )

        gap = interp_lumo(X, Y) - interp_homo(X, Y)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            gap,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="viridis",
            aspect="equal",
            vmin=np.nanmin(gap),
            vmax=np.nanmax(gap),
        )
        plt.colorbar(im, label="Local gap [eV]  (Mo LUMO − W HOMO)")
        plt.title(f"Local interlayer gap, twist angle {angle_natoms.split('_')[0]}")
        plt.xlabel("x [Å]")
        plt.ylabel("y [Å]")
        plt.tight_layout()
        plt.savefig(f"gap_{angle_natoms}.png", dpi=500)
        plt.close()

        angle.append(angle_natoms.split("_")[0])
        natoms.append(angle_natoms.split("_")[1])
        min_gap.append(np.nanmin(gap))
        max_gap.append(np.nanmax(gap))

    # Summary: min/max local gap vs angle
    fig, ax_bottom = plt.subplots(figsize=(8, 5))
    ax_bottom.plot(angle, min_gap, "-o", label="min local gap")
    ax_bottom.plot(angle, max_gap, "-o", label="max local gap")
    ax_bottom.set_xlabel("Angle")
    ax_bottom.set_ylabel("Local gap [eV]")
    ax_bottom.grid(True)
    ax_bottom.set_xticks(range(len(angle)))
    ax_bottom.set_xticklabels(angle, rotation=90)

    ax_top = ax_bottom.twiny()
    ax_top.set_xlim(ax_bottom.get_xlim())
    ax_top.set_xticks(range(len(natoms)))
    ax_top.set_xticklabels(natoms, rotation=90)
    ax_top.set_xlabel("Number of atoms")

    ax_bottom.legend()
    plt.tight_layout()
    plt.savefig("gap_angle_natoms.png", dpi=500)
    plt.close()
