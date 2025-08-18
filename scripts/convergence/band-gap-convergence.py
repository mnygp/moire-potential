from functions.structure import create_bilayer
from functions.bandstructure import calc_gap
from ase.parallel import parprint

kpts = list(range(10, 37, 2))

for z in [6.3, 6.5, 6.8]:
    parprint(f'Z distance: {z} Å')
    for shift in [0, 0.1, 0.2]:
        parprint("Shift:", shift)

        for k in kpts:
            bilayer = create_bilayer(z_dist=z, constrain=True,
                                     a_shift=shift, b_shift=shift)

            gap = calc_gap(bilayer, kpts=k)
            parprint(f'k-points: {k}, Band gap: {gap:.5f} eV')

        parprint(" ")

    parprint("--------------------------------------------------")
    parprint(" ")
