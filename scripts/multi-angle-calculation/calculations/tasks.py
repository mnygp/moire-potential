from pathlib import Path
import taskblaster as tb
from ase.io import read
from ase import Atoms
from numpy.typing import NDArray
import numpy as np

from functions.geometry import strain, interlayer_distance

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
        f"Could not find a directory named '{root}' in {current_path}"
    )

def get_dirs() -> list[str]:
    dir = get_root_path('moire-potential', 'structures/more-structures')
    # dir = '../../../'
    dir_arr = [str(p.resolve()) for p in Path(dir).iterdir() if p.is_dir()]
    return dir_arr

def read_atoms(p: str) -> Atoms:
    atoms = read(p + '/MatterSim_relaxed.json')
    return atoms

def strains(atoms: Atoms) -> dict[str, NDArray]:
    MoS2_strain = strain(atoms, 'Mo')
    WSe2_strain = strain(atoms, 'W')
    print(f'Max strain {max(np.max(MoS2_strain[2]), np.max(WSe2_strain[2]))}')
    print(f'Min strain {min(np.min(MoS2_strain[2]), np.min(WSe2_strain[2]))}')
    return {'MoS2': MoS2_strain, 'WSe2': WSe2_strain}

# def shifts

@tb.dynamical_workflow_generator_task
def generate_wfs(paths):
    for p in paths:
        name = p.split('/')[-1]
        wf = Sub_wf(path=p)
        yield name, wf


@tb.workflow
class Sub_wf:
    path = tb.var()

    @tb.task
    def get_atoms(self):
        return tb.node('read_atoms', p=self.path)

    @tb.task
    def get_strains(self):
        return tb.node('strains', atoms=self.get_atoms)

    @tb.task
    def z_dist(self):
        return tb.node('interlayer_distance', atoms=self.get_atoms)