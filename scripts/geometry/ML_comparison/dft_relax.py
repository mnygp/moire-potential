from ase.io import read, write
from ase.optimize import BFGS
from ase.build import mx2
from gpaw import GPAW, Mixer
from gpaw.new.extensions import D3


calc_params = {'random': True,
               'mode': {'name': 'pw',
                        'ecut': 500,
                        'dedecut': 'estimate',
                        'dtype': "complex64"},
               'xc': 'PBE',
               'parallel': {'gpu': True},
               'kpts': {'size': (1, 1, 1),
                        'gamma': True},
               'eigensolver': {'name': 'ppcg',
                               'niter': 4,
                               'include_cg': False},
                                # False means slower convergence
                                # but less memory
               'convergence': {'eigenstates': 4e-6,
                               # 'forces': 5e-4,
                               'density': 5e-5}}


atoms = read('structure_mattersim.json')

calc = GPAW(**calc_params, extensions=[D3(xc='PBE')])

atoms.calc = calc

opt = BFGS(atoms, trajectory='relax_DFT_after_mattersim.traj', logfile='DFT_after_mattersim.log')
opt.run(fmax=0.01)

write('structure_DFT_after_mattersim.json', atoms)
