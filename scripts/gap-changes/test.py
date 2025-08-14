from ase.io import read
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac
from pathlib import Path
import numpy as np
from ase.build import mx2
from ase.parallel import parprint
import matplotlib.pyplot as plt

atoms = mx2('MoS2', vacuum=10.0)

atoms.calc = GPAW(mode=PW(500),
                 xc='PBE',
                 kpts={'size': (8, 8, 1)},
                 occupations=FermiDirac(0.01),
                 txt='gpaw_output.gpw')

atoms.get_potential_energy()

calc = atoms.calc

V = calc.get_electrostatic_potential()
parprint(type(V), V.shape)

V_avg = V.mean(axis=(0, 1))  # average over x and y
vacuum_level = V_avg.max()
parprint(f'Vacuum level: {vacuum_level:.3f} eV')


# 2. Get Fermi level and eigenvalues
ef = calc.get_fermi_level()  # in eV
eigs = calc.get_eigenvalues(spin=0)  # first spin channel

parprint(f'Fermi level: {ef:.3f} eV')

# For spin-polarized case, include both spin channels
if calc.get_number_of_spins() == 2:
    eigs_spin1 = calc.get_eigenvalues(spin=1)
    eigs = np.concatenate([eigs, eigs_spin1])

# 3. HOMO is the highest eigenvalue <= Fermi level
homo = eigs[eigs <= ef].max()
# 4. LUMO is the lowest eigenvalue >= Fermi level
lumo = eigs[eigs >= ef].min()

# 5. Shift to vacuum level reference
homo_rel = homo - vacuum_level
lumo_rel = lumo - vacuum_level

parprint(f'HOMO: {homo:.3f} and LUMO: {lumo:.3f}')

parprint(f'Relative HOMO: {homo_rel:.3f} and relative LUMO: {lumo_rel:.3f}')
