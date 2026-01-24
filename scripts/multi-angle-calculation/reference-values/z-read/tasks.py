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
    return {"gap": gap, "indices": indices, "params": params}


def relax_chalcogenides(atoms: Atoms, fmax: float) -> Atoms:
    indices = [
        atom.index for atom in atoms if (atom.symbol == "W" or atom.symbol == "Mo")
    ]
    atoms.set_constraint(FixAtoms(indices=indices))

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
    z = tb.var()
    s1 = tb.var()
    s2 = tb.var()

    @tb.task
    def create_struct(self):
        return tb.node(
            "create_bilayer",
            z_dist=self.z,
            a_shift=self.s1,
            b_shift=self.s2,
            constrain=False,
        )

    @tb.task
    def relax_chalcs(self):
        return tb.node("relax_chalcogenides", atoms=self.create_struct, fmax=0.005)

    @tb.task
    def gap_calc(self):
        return tb.node(
            "gap_calculation",
            atoms=self.relax_chalcs,
            indices=self.indices,
            params=[self.z, self.s1, self.s2],
        )


@tb.dynamical_workflow_generator_task
def wfs(inputs):
    for i, z in enumerate(inputs["z"]):
        for j, s1 in enumerate(inputs["shift 1"]):
            for k, s2 in enumerate(inputs["shift 2"]):
                wf = gap_Wf(indices=[i, j, k], z=z, s1=s1, s2=s2)
                name = f"{z:.2f}_{s1:.2f}_{s2:.2f}"
                name = f"{z:.2f}_{s1:.2f}_{s2:.2f}"
                yield name, wf
