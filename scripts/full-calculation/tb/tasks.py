import taskblaster as tb

from pathlib import Path

from scipy.interpolate import LinearNDInterpolator

import numpy as np

import csv

from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms, FixedLine
from ase.filters import UnitCellFilter
from ase.calculators.dftd3 import DFTD3

from gpaw import PW  # type: ignore
from gpaw.new.ase_interface import GPAW

from functions.structure import create_bilayer
from functions.bandstructure import calc_gap, gap_and_kpts
from functions.geometry import interlayer_distance, get_shifts, strain
from functions.util import get_z_dist, repeate_cells


@tb.dynamical_workflow_generator_task
def generate_wfs_task(
    input: dict, fixed_atom: bool, fixed_cell: bool, structure_path: Path
):
    for x, y, w_str, z, i, j in zip(
        input["x_W"],
        input["y_W"],
        input["WSe2_strain"],
        input["interlayer_dist"],
        input["i_shifts"],
        input["j_shifts"],
    ):
        Mo_strain_arr = input["MoS2_strain"]
        x_Mo_arr = input["x_Mo"]
        y_Mo_arr = input["y_Mo"]
        wf = single_cell(
            x=x,
            y=y,
            i=i,
            j=j,
            W_strain=w_str,
            ml_z_dist=z,
            x_Mo_arr=x_Mo_arr,
            y_Mo_arr=y_Mo_arr,
            Mo_strain_arr=Mo_strain_arr,
            fixed_cell=fixed_cell,
            fixed_atom=fixed_atom,
            structure_path=structure_path,
        )
        name = f"{x:.3f}_{y:.3f}"
        yield name, wf


@tb.workflow
class single_cell:
    x = tb.var()
    y = tb.var()
    i = tb.var()
    j = tb.var()
    W_strain = tb.var()
    ml_z_dist = tb.var()
    x_Mo_arr = tb.var()
    y_Mo_arr = tb.var()
    Mo_strain_arr = tb.var()
    fixed_cell = tb.var()
    fixed_atom = tb.var()
    structure_path = tb.var()

    @tb.task
    def relax(self):
        return tb.node(
            "relaxation",
            z_dist=self.ml_z_dist,
            a_shift=self.i,
            b_shift=self.j,
            x=self.x,
            y=self.y,
            fixed_cell=self.fixed_cell,
            fixed_atom=self.fixed_atom,
        )

    @tb.task
    def calculate_gap_and_z(self):
        return tb.node("gap_and_z_dist", atom_path=self.relax)

    @tb.task
    def correction(self):
        return tb.node(
            "strain_correction",
            atom_path=self.relax,
            x=self.x,
            y=self.y,
            WSe2_strain=self.W_strain,
            x_Mo=self.x_Mo_arr,
            y_Mo=self.y_Mo_arr,
            Mo_strain_arr=self.Mo_strain_arr,
            structure_path=self.structure_path,
        )

    @tb.task
    def return_dict(self):
        return tb.node(
            "return_as_dict",
            x=self.x,
            y=self.y,
            z_ml=self.ml_z_dist,
            gap_and_z=self.calculate_gap_and_z,
            i=self.i,
            j=self.j,
            correction=self.correction,
            W_strain=self.W_strain,
        )


def create_structure(z_dist: float, a_shift: float, b_shift: float) -> Path:
    struct = create_bilayer(
        z_dist, lattice_length=3.2515, a_shift=a_shift, b_shift=b_shift
    )
    write("bilayer.json", struct)
    return Path("bilayer.json")


def relaxation(
    x: int,
    y: int,
    z_dist: float,
    a_shift: float,
    b_shift: float,
    fixed_cell: bool,
    fixed_atom: bool,
) -> Path:
    atoms = create_bilayer(
        z_dist, lattice_length=3.2515, a_shift=a_shift, b_shift=b_shift
    )
    file_name = f"opt_{x:.3f}_{y:.3f}"

    indices = [
        atom.index for atom in atoms if (atom.symbol == "W" or atom.symbol == "Mo")
    ]
    if fixed_atom:
        atoms.set_constraint(FixAtoms(indices=indices))
        file_name = file_name + "_fixed_TM"
    else:
        atoms.set_constraint(FixedLine(indices=indices, direction=[0, 0, 1]))

    calc = GPAW(mode=PW(500), xc="PBE", kpts={"size": (8, 8, 1)})
    d3_calc = DFTD3(dft=calc, xc="PBE")
    atoms.calc = d3_calc

    if fixed_cell:
        opt = atoms
        file_name = file_name + "_fixed_cell"
    else:
        # mask makes it so the unitcell is only optimised in x, y and xy
        opt = UnitCellFilter(atoms, mask=[1, 1, 0, 0, 0, 1])

    relax = BFGS(opt, trajectory=file_name + ".traj", logfile=file_name + ".log")
    relax.run(fmax=0.01)

    relaxed_name = "relaxed" + file_name[3:] + ".json"
    write(relaxed_name, atoms)

    return Path(relaxed_name)


def gap_z_distand_kpts(atom_path) -> list[float]:
    gap = gap_and_kpts(atom_path, kpts=30, soc=True, functional="HSE06")
    z_dist = get_z_dist(atom_path)
    return {
        "gap": gap["gap"],
        "z_dist": z_dist,
        "lumo_kpt": gap["lumo_kpt"],
        "homo_kpt": gap["homo_kpt"],
    }


def get_root_path(directory: str, path_str: str) -> str:
    current_path = Path(__file__).resolve()
    print(f"Current path: {current_path}")

    for parent in current_path.parents:
        if parent.name == directory:
            full_path = Path(parent) / path_str.lstrip("/")
            full_path = full_path.resolve()
            print(f"Resolved structure path: {full_path}")
            return str(full_path)

    raise FileNotFoundError(
        f"Could not find a directory named '{directory}' in {current_path}"
    )


def get_geometry(atom_path):
    atoms = read(atom_path)
    x_MoS2, y_MoS2, MoS2_strain = strain(atoms, "Mo")
    x_WSe2, y_WSe2, WSe2_strain = strain(atoms, "W")
    _, _, interlayer_dist = interlayer_distance(atoms)

    shift_dict = get_shifts(atoms)
    x_shifts = shift_dict["shifts"][:, 0]
    y_shifts = shift_dict["shifts"][:, 1]

    return {
        "x_W": x_WSe2,
        "y_W": y_WSe2,
        "x_Mo": x_MoS2,
        "y_Mo": y_MoS2,
        "MoS2_strain": MoS2_strain,
        "WSe2_strain": WSe2_strain,
        "interlayer_dist": interlayer_dist,
        "i_shifts": x_shifts,
        "j_shifts": y_shifts,
    }


def strain_correction(
    atom_path: Path,
    structure_path: Path,
    WSe2_strain: float,
    x: float,
    y: float,
    x_Mo: list[float],
    y_Mo: list[float],
    Mo_strain_arr: list[float],
) -> list[float]:
    atoms = read(atom_path)
    lattice_length = np.mean(np.linalg.norm(atoms.cell[:2, :2], axis=1))
    MoS2_ref = (lattice_length - 3.184) / 3.184 + 1
    WSe2_ref = (lattice_length - 3.319) / 3.319 + 1

    ref_strains = [MoS2_ref, WSe2_ref]

    err_str = (
        "Excessive strain or wrong unit."
        + " Strain must be in decimal and within +/- 3.5%."
    )
    assert 0.965 < ref_strains[0] < 1.035, err_str
    assert 0.965 < ref_strains[1] < 1.035, err_str

    struct = read(structure_path)
    # Create the Mo strain interpolator and get the value
    x_Mo_L, y_Mo_L, Mo_strain_L = repeate_cells(
        x_Mo, y_Mo, Mo_strain_arr, range(-1, 2), struct.cell[0], struct.cell[1]
    )

    MoS2_strain_intp = LinearNDInterpolator(
        np.column_stack((x_Mo_L, y_Mo_L)), Mo_strain_L
    )
    MoS2_strain = MoS2_strain_intp([x, y])[0]

    # The medium data set goes from -2% to 2% along both axis
    csv_file = find_file(
        filename="band_edges_large_soc.csv", root_dir_name="full-calculation"
    )
    data = np.genfromtxt(csv_file, skip_header=1, dtype=float, delimiter=",")

    strain_data = data[:, 0]
    MoS2_lumo = data[:, 2]
    WSe2_homo = data[:, 3]

    # Create the band gab grid and interpolator
    lumo_grid, homo_grid = np.meshgrid(MoS2_lumo, WSe2_homo)
    band_gap_grid = lumo_grid - homo_grid
    MoS2_grid, WSe2_grid = np.meshgrid(strain_data, strain_data)
    points = np.column_stack([MoS2_grid.ravel(), WSe2_grid.ravel()])
    values = band_gap_grid.ravel()  # shape (N,)
    correction_interp = LinearNDInterpolator(points, values)

    ref_val = correction_interp(ref_strains)

    correction_val = correction_interp([MoS2_strain + 1, WSe2_strain + 1])

    return [(correction_val - ref_val)[0], MoS2_strain]


def return_as_dict(
    x: float,
    y: float,
    z_ml: float,
    gap_and_z: list[float],
    i: int,
    j: int,
    correction: list[float],
    W_strain: float,
    lumo_kpts: tuple[float, float, float],
    homo_kpts: tuple[float, float, float],
) -> dict:
    return {
        "x": x,
        "y": y,
        "z_ml": z_ml,
        "z_dft": gap_and_z[1],
        "i": i,
        "j": j,
        "gap": gap_and_z[0],
        "correction": correction[0],
        "Mo_strain": correction[1],
        "W_strain": W_strain,
        "lumo_kpts": lumo_kpts,
        "Homo_kpts": homo_kpts,
    }


def find_file(filename, root_dir_name) -> Path:
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if parent.name == root_dir_name:
            csv_path = parent / filename
            if csv_path.exists():
                return csv_path.resolve()
            else:
                raise FileNotFoundError(
                    f"{filename} not found inside {root_dir_name} at {parent}"
                )

    raise FileNotFoundError(
        f"Could not find a directory named '{root_dir_name}' from {current_path}"
    )


def write_results_to_csv(results_dict: dict, csv_name: str) -> Path:
    """Write results from TaskBlaster outputs to a CSV file."""
    rows = []
    for name, d in results_dict.items():
        rows.append(
            {
                "x": d["x"],
                "y": d["y"],
                "i": d["i"],
                "j": d["j"],
                "z_ml": d["z_ml"],
                "z_dft": d["z_dft"],
                "gap": d["gap"],
                "correction": d["correction"],
                "Mo_strain": d["Mo_strain"],
                "W_strain": d["W_strain"],
            }
        )

    # Write the CSV
    with open(csv_name, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "x",
                "y",
                "i",
                "j",
                "z_ml",
                "z_dft",
                "gap",
                "correction",
                "Mo_strain",
                "W_strain",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return Path(csv_name)


def write_Mo_pos_to_csv(results_dict: dict, csv_name: str) -> Path:
    rows = []
    for name, d in results_dict.items():
        rows.append({"x_Mo": d["x_Mo"], "y_Mo": d["y_Mo"]})

    # Write the CSV
    with open(csv_name, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x_Mo", "y_Mo"])
        writer.writeheader()
        writer.writerows(rows)

    return Path(csv_name)


def write_kpts_to_csv(results_dict: dict, csv_name: str) -> Path:
    rows = []
    for name, d in results_dict.items():
        rows.append(
            {
                "x": d["x"],
                "y": d["y"],
                "lumo kx": d["lumo_kpts"][0],
                "lumo ky": d["lumo_kpts"][1],
                "lumo kz": d["lumo_kpts"][2],
                "homo kx": d["homo_kpts"][0],
                "homo ky": d["homo_kpts"][1],
                "homo kz": d["homo_kpts"][2],
            }
        )

    with open(csv_name, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "x",
                "y",
                "lumo kx",
                "lumo ky",
                "lumo kz",
                "homo kx",
                "homo ky",
                "homo kz",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return Path(csv_name)
