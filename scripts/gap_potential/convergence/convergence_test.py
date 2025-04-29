from ase import Atoms
from ase.io import read
from ase.parallel import parprint

from gpaw import GPAW, PW

from functions.util import add_distance

ml_atoms = read('atoms_240_ml.xyz')
extra_dist_atoms = read('atoms_240_extra_dist.xyz')


def PW_conv(atoms: Atoms) -> tuple[list[float], list[float]]:
    pw_arr = []
    pw_energy_arr = []
    pw_cut = 300.0
    diff = 200
    while diff > 0.01:
        calc = GPAW(mode=PW(pw_cut),
                    xc='PBE',
                    kpts={'size': (12, 12, 1)},
                    txt='gpaw.txt')

        atoms.calc = calc
        energy = atoms.get_potential_energy()
        pw_arr.append(pw_cut)
        pw_energy_arr.append(energy)
        if len(pw_arr) > 1:
            diff = abs(pw_energy_arr[-1] - pw_energy_arr[-2])/6
        parprint(f'PW: {pw_cut}, Energy: {energy}, Diff: {diff}')
        pw_cut += 50

        if pw_cut > 1000:
            parprint('PW cut-off too high, stopping calculation')
            break
    return pw_arr, pw_energy_arr


def k_conv(atoms: Atoms, pw_cut: float) -> tuple[list[int], list[float]]:
    k_arr = []
    k_energy_arr = []
    kpts = 2
    diff = 200.0
    while diff > 0.01:
        calc = GPAW(mode=PW(pw_cut),
                    xc='PBE',
                    kpts={'size': (kpts, kpts, 1)},
                    txt='gpaw.txt')

        atoms.calc = calc
        energy = atoms.get_potential_energy()
        k_arr.append(kpts)
        k_energy_arr.append(energy)
        if len(k_arr) > 1:
            diff = abs(k_energy_arr[-1] - k_energy_arr[-2])/6
        parprint(f'K: {kpts}, Energy: {energy}, Diff: {diff}')
        kpts += 2

        if kpts > 20:
            parprint('K points too high, stopping calculation')
            break
    return k_arr, k_energy_arr


def dist_conv(atoms: Atoms, pw_cut: float,
              kpts: int) -> tuple[list[float], list[float]]:
    dist_arr = []
    dist_energy_arr = []
    dist = 0.5
    diff = 200.0
    while diff > 0.01:
        atoms = add_distance(atoms, 0.5)
        calc = GPAW(mode=PW(pw_cut),
                    xc='PBE',
                    kpts={'size': (kpts, kpts, 1)},
                    txt='gpaw.txt')

        atoms.calc = calc
        energy = atoms.get_potential_energy()
        dist_arr.append(dist)
        dist_energy_arr.append(energy)
        if len(dist_arr) > 1:
            diff = abs(dist_energy_arr[-1] - dist_energy_arr[-2])/6
        parprint(f'Distance: {dist}, Energy: {energy}, Diff: {diff}')
        dist += 0.5
        if dist > 10:
            parprint('Distance too high, stopping calculation')
            break
    return dist_arr, dist_energy_arr


parprint('Starting convergence test')
ml_pw_arr, ml_pw_energy = PW_conv(ml_atoms)
parprint('Planewave convergence')
parprint(ml_pw_arr, ml_pw_energy)
ml_k_arr, ml_k_energy = k_conv(ml_atoms, ml_pw_arr[-1])
parprint('K points convergence')
parprint(ml_k_arr, ml_k_energy)

parprint('')
parprint('')

parprint('Starting extra distance convergence test')
extra_pw_arr, extra_pw_energy = PW_conv(extra_dist_atoms)
parprint('Planewave convergence')
parprint(extra_pw_arr, extra_pw_energy)
extra_k_arr, extra_k_energy = k_conv(extra_dist_atoms, extra_pw_arr[-1])
parprint('K points convergence')
parprint(extra_k_arr, extra_k_energy)
parprint('')
parprint('Starting distance convergence test')
ml_dist, ml_dist_energy = dist_conv(ml_atoms, ml_pw_arr[-1], ml_k_arr[-1])
parprint('Distance convergence')
parprint(ml_dist, ml_dist_energy)
