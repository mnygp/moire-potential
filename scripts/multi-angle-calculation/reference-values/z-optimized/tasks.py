import taskblaster as tb
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from gpaw.new.ase_interface import GPAW
from gpaw.new.extensions import D3

from functions.bandstructure import calc_gap
from functions.structure import create_bilayer
from functions.util import generate_scissor_shifts


def gap_calculation(atoms: Atoms, indices: list[int], params: list[float]) -> float:
    scissor = generate_scissor_shifts(atoms)
    gap = calc_gap(
        atoms, kpts=36, soc=True, eigensolver={"name": "scissors", "shifts": scissor}
    )[0]

    Mo_index = [atoms.index for atom in atoms if atom.symbol == "Mo"][0]
    Mo_z = atoms.positions[Mo_index][2]
    W_index = [atoms.index for atom in atoms if atom.symbol == "W"][0]
    W_z = atoms.positions[W_index][2]
    z_dist = abs(Mo_z - W_z)

    return {"gap": gap, "z": z_dist, "indices": indices, "params": params}


def relax_z_dist(atoms: Atoms, fmax: float) -> Atoms:
    calc = GPAW(
        mode="lcao",
        basis="dzp",
        xc="PBE",
        kpts={"size": (12, 12, 1)},
        txt="gpaw.txt",
        extensions=[D3(xc="PBE")],
        convergence={"forces": 5e-4, "density": 1e-5},
    )
    atoms.calc = calc

    opt = BFGS(atoms, logfile="chalc_relax.log", trajectory="chalc_relax.traj")
    opt.run(fmax=fmax)
    return atoms


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
