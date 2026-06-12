import csv
from pathlib import Path

import numpy as np
import taskblaster as tb
from ase import Atoms
from ase.optimize import BFGS
from gpaw.new.ase_interface import GPAW
from gpaw.new.extensions import D3

from functions.bandstructure import calc_gap
from functions.structure import create_bilayer
from functions.util import generate_scissor_shifts, get_path


def gap_calculation(atoms: Atoms, indices: list[int], params: list[float]) -> dict:
    Mo_index = [atom.index for atom in atoms if atom.symbol == "Mo"][0]
    Mo_z = atoms.positions[Mo_index][2]
    W_index = [atom.index for atom in atoms if atom.symbol == "W"][0]
    W_z = atoms.positions[W_index][2]
    z_dist = abs(Mo_z - W_z)

    Mo_path = get_path("reference-values", "MoS2_shifts.npy")
    W_path = get_path("reference-values", "WSe2_shifts.npy")

    Mo_shifts = np.load(Mo_path)
    W_shifts = np.load(W_path)
    Mo_unocc_shift = Mo_shifts[:, 0]
    Mo_occ_shift = Mo_shifts[:, 1]
    W_unocc_shift = W_shifts[:, 0]
    W_occ_shift = W_shifts[:, 1]
    dist_shift = np.linspace(6.2, 7.2, 21)
    shifts = np.array([])
    for s in [Mo_unocc_shift, Mo_occ_shift, W_unocc_shift, W_occ_shift]:
        interp = np.interp(z_dist, dist_shift, s)
        shifts = np.append(shifts, interp)

    shifts = generate_scissor_shifts(atoms, shifts)
    gap = calc_gap(
        atoms,
        kpts=36,
        soc=True,
        mode="lcao",
        eigensolver={"name": "scissors", "shifts": shifts},
    )[0]

    return {"gap": gap, "z": z_dist, "indices": indices, "params": params}


def relax_z_dist(atoms: Atoms, fmax: float) -> Atoms:
    calc = GPAW(
        mode="lcao",
        basis="dzp",
        xc="PBE",
        kpts={"size": (12, 12, 1)},
        txt="gpaw.txt",
        extensions=[D3(xc="PBE")],
        convergence={"forces": 1e-5, "density": 1e-6},
    )
    atoms.calc = calc

    opt = BFGS(atoms, logfile="chalc_relax.log", trajectory="chalc_relax.traj")
    opt.run(fmax=fmax)
    return atoms


def write_results_to_csv(results_dict: dict, csv_name: str) -> str:
    rows = []
    for name, d in results_dict.items():
        rows.append(
            {
                "gap": d["gap"],
                "shift 1": d["params"][0],
                "shift 2": d["params"][1],
                "z": d["z"],
            }
        )

    csv_path = Path(csv_name)
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["gap", "shift 1", "shift 2", "z"])
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path.resolve())


@tb.workflow
class gap_Wf:
    indices = tb.var()
    s1 = tb.var()
    s2 = tb.var()

    @tb.task
    def create_struct(self):
        return tb.node(
            "create_bilayer",
            z_dist=6.6,
            a_shift=self.s1,
            b_shift=self.s2,
            constrain=True,
        )

    @tb.task
    def relax_z(self):
        return tb.node("relax_z_dist", atoms=self.create_struct, fmax=0.005)

    @tb.task
    def gap_calc(self):
        return tb.node(
            "gap_calculation",
            atoms=self.relax_z,
            indices=self.indices,
            params=[self.s1, self.s2],
        )


@tb.dynamical_workflow_generator_task
def wfs(inputs):
    for j, s1 in enumerate(inputs["shift 1"]):
        for k, s2 in enumerate(inputs["shift 2"]):
            wf = gap_Wf(indices=[j, k], s1=s1, s2=s2)
            name = f"{s1:.2f}_{s2:.2f}"
            yield name, wf
