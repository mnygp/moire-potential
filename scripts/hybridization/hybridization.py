from ase.parallel import parprint
from ase.optimize import BFGS
from ase.calculators.dftd3 import DFTD3
from ase.constraints import FixAtoms
from gpaw import GPAW, PW, FermiDirac
import numpy as np
from functions.structure import create_bilayer
from functions.bandstructure import get_vacuum_and_band_edges


ref_values = np.zeros((3, 4, 3))  # lattice, shift, (homo, lumo)
nkpts = 26

for k, lattice, name in zip(
    [0, 1, 2], [3.184, 3.2515, 3.319], ["MoS2", "Average", "WSe2"]
):
    parprint("----------------------------------")
    parprint(f"Starting {name} lattice length")

    data = np.zeros((4, 12, 3))  # shift , z, (homo, lumo))

    for i, shift in enumerate([0.0, 0.1, 0.2, 0.3]):
        parprint(f"Starting {shift} shift")
        # Calculate a reference value with no hybridization where the
        # layers are 10Å apart
        ref_bilayer = create_bilayer(
            10, lattice_length=lattice, a_shift=shift, b_shift=shift
        )

        c = FixAtoms(
            indices=[
                atom.index
                for atom in ref_bilayer
                if (atom.symbol == "W" or atom.symbol == "Mo")
            ]
        )
        ref_bilayer.set_constraint(c)
        ref_bilayer.calc = DFTD3(
            dft=GPAW(
                mode=PW(500),
                xc="PBE",
                kpts={"size": (nkpts, nkpts, 1)},
                occupations=FermiDirac(0.01),
                txt=None,
            )
        )
        opt = BFGS(
            ref_bilayer, trajectory=f"traj_files/opt_ref_bilayer_{name}_{shift}.traj"
        )
        opt.run(fmax=0.02)
        ref_bilayer.calc.get_potential_energy()
        ref_bilayer.calc.dft.write(
            f"gpw_files/ref_bilayer_{name}_{shift}.gpw", mode="all"
        )

        ref_bilayer_dict = get_vacuum_and_band_edges(
            f"gpw_files/ref_bilayer_{name}_{shift}.gpw"
        )

        ref_values[k, i] = [shift, ref_bilayer_dict["homo"], ref_bilayer_dict["lumo"]]
        parprint(f"Calculated reference values for {name}")

        # Calculate the actual values to quantify the hybridization
        for j, z in enumerate(np.linspace(6.3, 6.9, 12)):
            parprint(f"Starting {z:.2f}Å z distance")
            bilayer = create_bilayer(
                z, lattice_length=lattice, a_shift=shift, b_shift=shift
            )

            c = FixAtoms(
                indices=[
                    atom.index
                    for atom in bilayer
                    if (atom.symbol == "W" or atom.symbol == "Mo")
                ]
            )
            bilayer.set_constraint(c)
            bilayer.calc = DFTD3(
                dft=GPAW(
                    mode=PW(500),
                    xc="PBE",
                    kpts={"size": (nkpts, nkpts, 1)},
                    occupations=FermiDirac(0.01),
                    txt=None,
                )
            )
            opt = BFGS(
                bilayer,
                trajectory=f"traj_files/opt_ref_bilayer_{name}_{shift}_{z:.2f}.traj",
            )
            opt.run(fmax=0.02)
            bilayer.calc.get_potential_energy()
            bilayer.calc.dft.write(
                f"gpw_files/bilayer_{name}_{shift}_{z:.2f}.gpw", mode="all"
            )

            bilayer_dict = get_vacuum_and_band_edges(
                f"gpw_files/bilayer_{name}_{shift}_{z:.2f}.gpw"
            )
            data[i, j] = [z, bilayer_dict["homo"], bilayer_dict["lumo"]]

    np.save(f"{name}_values.npy", data)  # strain, (z, (homo, lumo))

np.save("ref_values.npy", ref_values)
