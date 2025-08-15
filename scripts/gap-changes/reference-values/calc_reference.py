from ase.build import mx2
from ase.parallel import parprint
from ase.calculators.dftd3 import DFTD3
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from gpaw import GPAW, PW, FermiDirac
from functions.bandstructure import get_vacuum_and_band_edges
from functions.structure import create_bilayer

average_lattice = 3.2515
MoS2_lattice = 3.184
WSe2_lattice = 3.319

nkpts = 26

# Calc bilayer
bilayer = create_bilayer(z_dist=10)
c = FixAtoms(indices=[atom.index for atom in bilayer
                      if (atom.symbol == 'W' or atom.symbol == 'Mo')])
bilayer.set_constraint(c)
bilayer.calc = DFTD3(dft=GPAW(mode=PW(500),
                              xc='PBE',
                              kpts={'size': (nkpts, nkpts, 1)},
                              occupations=FermiDirac(0.01),
                              txt=None))
opt = BFGS(bilayer, trajectory='opt_bilayer.traj')
opt.run(fmax=0.02)
bilayer.calc.get_potential_energy()
bilayer.calc.dft.write('bilayer.gpw', mode='all')

bilayer_dict = get_vacuum_and_band_edges('bilayer.gpw')
parprint("---------------------Bilayer done----------------------")

# Calc MoS2
MoS2 = mx2('MoS2', a=average_lattice, vacuum=10.0)
MoS2.calc = DFTD3(dft=GPAW(mode=PW(500),
                           xc='PBE',
                           kpts={'size': (nkpts, nkpts, 1)},
                           occupations=FermiDirac(0.01),
                           txt=None))
opt = BFGS(MoS2, trajectory='opt_MoS2.traj')
opt.run(fmax=0.02)
MoS2.calc.get_potential_energy()
MoS2.calc.dft.write('MoS2.gpw', mode='all')

MoS2_dict = get_vacuum_and_band_edges('MoS2.gpw')
parprint("---------------------MoS2 done----------------------")

# Calc WSe2
WSe2 = mx2('WSe2', a=average_lattice, vacuum=10.0)
WSe2.calc = DFTD3(dft=GPAW(mode=PW(500),
                           xc='PBE',
                           kpts={'size': (nkpts, nkpts, 1)},
                           occupations=FermiDirac(0.01),
                           txt=None))
opt = BFGS(WSe2, trajectory='opt_WSe2.traj')
opt.run(fmax=0.02)
WSe2.calc.get_potential_energy()
WSe2.calc.dft.write('WSe2.gpw', mode='all')

WSe2_dict = get_vacuum_and_band_edges('WSe2.gpw')
parprint("---------------------WSe2 done----------------------")


parprint(f'MoS2 band gap: {MoS2_dict["bandgap"]:.2f}')
parprint(f'WSe2 band gap: {WSe2_dict["bandgap"]:.2f}')
parprint(f'Combined bandgap: {(MoS2_dict["lumo"] - WSe2_dict["homo"])}')
parprint("--------------------------------")
parprint(f'Bilayer bandgap: {bilayer_dict["bandgap"]}')
