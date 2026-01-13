from ase.build import mx2
from ase.parallel import parprint
from ase.calculators.dftd3 import DFTD3
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac  # pyright: ignore
from functions.bandstructure import get_vacuum_and_band_edges
import numpy as np
import csv

average_lattice = 3.2515
MoS2_lattice = 3.184
WSe2_lattice = 3.319

# How much to strain either layer to reach equilibrium lattice constant
equilibrium_strain = (WSe2_lattice - average_lattice) / average_lattice

nkpts = 50

strain = np.linspace(1 - 0.035, 1 + 0.035, endpoint=True)
parprint(
    f"Calculating strain from {(1 - equilibrium_strain) * 100:.2f}%"
    f" to {(1 + equilibrium_strain) * 100:.2f}%"
)
MoS2_homo = []
MoS2_lumo = []
WSe2_homo = []
WSe2_lumo = []

# Calculate the HOMO and LUMO at different lattice constants
for i in strain:
    parprint(f"Begining strain {(i - 1) * 100:.2f}")
    MoS2 = mx2("MoS2", a=MoS2_lattice * i, vacuum=10.0)
    MoS2.calc = DFTD3(
        dft=GPAW(
            mode=PW(500),
            xc="PBE",
            kpts={"size": (nkpts, nkpts, 1)},
            occupations=FermiDirac(0.01),
            txt=None,
        )
    )
    file_Mo = f"MoS2_{(i - 1) * 100:.2f}"
    opt = BFGS(MoS2, trajectory=f"traj_files/opt_{file_Mo}.traj")
    opt.run(fmax=0.01)
    MoS2.calc.get_potential_energy()
    MoS2.calc.dft.write(
        f"gpw_files/{file_Mo}.gpw",  # pyright: ignore
        mode="all",
    )

    MoS2_dict = get_vacuum_and_band_edges(f"gpw_files/{file_Mo}.gpw", soc=True)
    MoS2_homo.append(MoS2_dict["homo"] - MoS2_dict["vacuum_level"])
    MoS2_lumo.append(MoS2_dict["lumo"] - MoS2_dict["vacuum_level"])
    parprint("MoS2 done")

    WSe2 = mx2("WSe2", a=WSe2_lattice * i, vacuum=10.0)
    WSe2.calc = DFTD3(
        dft=GPAW(
            mode=PW(500),
            xc="PBE",
            kpts={"size": (nkpts, nkpts, 1)},
            occupations=FermiDirac(0.01),
            txt=None,
        )
    )
    file_W = f"WSe2_{(i - 1) * 100:.2f}"
    opt = BFGS(WSe2, trajectory=f"traj_files/opt_{file_W}.traj")
    opt.run(fmax=0.01)
    WSe2.calc.get_potential_energy()
    WSe2.calc.dft.write(
        f"gpw_files/{file_W}.gpw",  # pyright: ignore
        mode="all",
    )

    WSe2_dict = get_vacuum_and_band_edges(f"gpw_files/{file_W}.gpw", soc=True)
    WSe2_homo.append(WSe2_dict["homo"] - WSe2_dict["vacuum_level"])
    WSe2_lumo.append(WSe2_dict["lumo"] - WSe2_dict["vacuum_level"])
    parprint("WSe2 done")
    parprint(f"Strain {(i - 1) * 100:.2f} done")
    parprint("---------------------------------------------")


with open("band_edges_large_soc.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    # Header row
    writer.writerow(["strain", "MoS2_homo", "MoS2_lumo", "WSe2_homo", "WSe2_lumo"])
    # Data rows
    for s, m_h, m_l, w_h, w_l in zip(
        strain, MoS2_homo, MoS2_lumo, WSe2_homo, WSe2_lumo
    ):
        writer.writerow([s, m_h, m_l, w_h, w_l])
