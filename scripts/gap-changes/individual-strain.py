from ase.build import mx2
from ase.parallel import parprint
from ase.calculators.dftd3 import DFTD3
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac
from functions.bandstructure import get_vacuum_and_band_edges
import numpy as np
import csv

average_lattice = 3.2515
MoS2_lattice = 3.184
WSe2_lattice = 3.319

nkpts = 26

strain = np.linspace(0.995, 1.005, 21, endpoint=True)
MoS2_homo = []
MoS2_lumo = []
WSe2_homo = []
WSe2_lumo = []

# Calculate the HOMO and LUMO at different lattice constants
for i in strain:
    parprint(f'Begining strain {(i-1)*100:.2f}')
    MoS2 = mx2('MoS2', a=MoS2_lattice*i, vacuum=10.0)
    MoS2.calc = DFTD3(dft=GPAW(mode=PW(500),
                               xc='PBE',
                               kpts={'size': (nkpts, nkpts, 1)},
                               occupations=FermiDirac(0.01),
                               txt=None))
    opt = BFGS(MoS2, trajectory=f'traj_files/opt_MoS2_{(i-1)*100:.2f}.traj')
    opt.run(fmax=0.02)
    MoS2.calc.get_potential_energy()
    MoS2.calc.dft.write(f'gpw_files/MoS2_{(i-1)*100:.2f}.gpw', mode='all')

    MoS2_dict = get_vacuum_and_band_edges(f'gpw_files/MoS2_{(i-1)*100:.2f}.gpw')
    MoS2_homo.append(MoS2_dict['homo'])
    MoS2_lumo.append(MoS2_dict['lumo'])
    parprint('MoS2 done')

    WSe2 = mx2('WSe2', a=WSe2_lattice*i, vacuum=10.0)
    WSe2.calc = DFTD3(dft=GPAW(mode=PW(500),
                               xc='PBE',
                               kpts={'size': (nkpts, nkpts, 1)},
                               occupations=FermiDirac(0.01),
                               txt=None))
    opt = BFGS(WSe2, trajectory=f'traj_files/opt_WSe2_{(i-1)*100:.2f}.traj')
    opt.run(fmax=0.02)
    WSe2.calc.get_potential_energy()
    WSe2.calc.dft.write(f'gpw_files/WSe2_{(i-1)*100:.2f}.gpw', mode='all')

    WSe2_dict = get_vacuum_and_band_edges(f'gpw_files/WSe2_{(i-1)*100:.2f}.gpw')
    WSe2_homo.append(WSe2_dict['homo'])
    WSe2_lumo.append(WSe2_dict['lumo'])
    parprint('WSe2 done')
    parprint(f'Strain {(i-1)*100:.2f} done')
    parprint('---------------------------------------------')


with open('band_edges.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    # Header row
    writer.writerow(['strain', 'MoS2_homo', 'MoS2_lumo',
                     'WSe2_homo', 'WSe2_lumo'])
    # Data rows
    for s, m_h, m_l, w_h, w_l in zip(strain, MoS2_homo, MoS2_lumo,
                                     WSe2_homo, WSe2_lumo):
        writer.writerow([s, m_h, m_l, w_h, w_l])
