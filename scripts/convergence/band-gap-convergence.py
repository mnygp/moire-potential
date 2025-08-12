from functions.structure import create_bilayer
from functions.bandstructure import calc_gap


for z in [6.3, 6.5, 6.8]:
    print(f'Z distance: {z} Å')
    for shift in [0, 0.1, 0.2]:
        print("Shift:", shift)

        kpts = list(range(10, 37, 2))
        gap_arr = []

        for k in kpts:
            bilayer = create_bilayer(z_dist=6.6, constrain=True,
                                     a_shift=shift, b_shift=shift)

            gap = calc_gap(bilayer, kpts=k)

            gap_arr.append(gap)
            print(f'k-points: {k}, Band gap: {gap:.3f} eV')
        print(" ")
    print("--------------------------------------------------")
    print(" ")
