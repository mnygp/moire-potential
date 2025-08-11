from ase.io import read, write

from mattersim.applications.relax import Relaxer
from mattersim.forcefield.potential import MatterSimCalculator

structure = read('initial.json')

structure.calc = MatterSimCalculator()

relaxer = Relaxer(optimizer="BFGS")

relaxed_structure = relaxer.relax(structure, steps=2000)

write("mattersim_relaxed.traj", relaxed_structure[1])
