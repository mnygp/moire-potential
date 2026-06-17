from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from gpaw import PW, FermiDirac
from gpaw.new.ase_interface import GPAW
from gpaw.spinorbit import soc_eigenstates

from functions.util import generate_scissor_shifts


def calc_gap(
    atom_path: Path | Atoms,
    functional: str = "PBE",
    kpts: int = 18,
    pw_cut: float = 500,
    mode: str = "pw",
    soc: bool = False,
    eigensolver: dict | None = None,
) -> tuple[float, float, float]:
    if isinstance(atom_path, Path):
        atoms = read(atom_path)
    elif isinstance(atom_path, Atoms):
        atoms = atom_path
    else:
        raise TypeError("atom_path must be a Path or Atoms object")

    # TODO: Refactor this fucking mess
    # Parse dict for calc parameters
    if eigensolver is not None:
        calc = GPAW(
            mode=PW(pw_cut),  # Basis set
            xc=functional,  # Functional
            kpts={"size": (kpts, kpts, 1)},  # k-points
            occupations=FermiDirac(0.01),
            eigensolver=eigensolver,
            txt=None,
        )
    else:
        calc = GPAW(
            mode=PW(pw_cut),  # Basis set
            xc=functional,  # Functional
            kpts={"size": (kpts, kpts, 1)},  # k-points
            occupations=FermiDirac(0.01),
            txt=None,
        )

    if mode == "lcao":
        if eigensolver is not None:
            calc = GPAW(
                mode="lcao",
                basis="dzp",
                xc=functional,
                kpts={"size": (kpts, kpts, 1)},
                occupations=FermiDirac(0.01),
                eigensolver=eigensolver,
                txt=None,
            )
        else:
            calc = GPAW(
                mode="lcao",
                basis="dzp",
                xc=functional,
                kpts={"size": (kpts, kpts, 1)},
                occupations=FermiDirac(0.01),
                txt=None,
            )

    atoms.calc = calc
    atoms.get_potential_energy()

    if soc:
        soc_eig = soc_eigenstates(calc)
        eigs = soc_eig.eigenvalues()
        occ = soc_eig.occupation_numbers()

        energies = eigs.ravel()
        occs = occ.ravel()

        occ_thresh = 0.5
        occupied_energies = energies[occs > occ_thresh]
        unoccupied_energies = energies[occs <= occ_thresh]

        homo = occupied_energies.max()
        lumo = unoccupied_energies.min()
    else:
        homo, lumo = calc.get_homo_lumo()

    return lumo - homo, lumo, homo


def get_vacuum_and_band_edges(gpw_file: str, soc=False):
    calc = GPAW(gpw_file, txt=None)
    V = calc.get_electrostatic_potential()
    V_avg = V.mean(axis=(0, 1))  # average over x and y
    vacuum_level = V_avg.max()
    ef = calc.get_fermi_level()  # in eV

    if soc:
        soc = soc_eigenstates(calc)
        eigs = soc.eigenvalues()
        occ = soc.occupation_numbers()

        energies = eigs.ravel()  # flatten eigenvalues into 1D array
        occs = occ.ravel()  # flatten occupation numbers

        occ_thresh = 0.5
        occupied_energies = energies[occs > occ_thresh]
        unoccupied_energies = energies[occs <= occ_thresh]

        homo = occupied_energies.max()
        lumo = unoccupied_energies.min()
    else:
        eigs = calc.get_eigenvalues(spin=0)  # first spin channel

        if calc.get_number_of_spins() == 2:
            eigs_spin1 = calc.get_eigenvalues(spin=1)
            eigs = np.concatenate([eigs, eigs_spin1])

        homo = eigs[eigs <= ef].max()
        lumo = eigs[eigs >= ef].min()

    # 5. Shift to vacuum level reference
    homo_rel = homo - vacuum_level
    lumo_rel = lumo - vacuum_level

    return {
        "vacuum_level": vacuum_level,
        "homo": homo_rel,
        "lumo": lumo_rel,
        "bandgap": lumo - homo,
    }


def gap_and_kpts(atom_path, functional, kpts, occ_thresh=0.5, scissors=False):
    if isinstance(atom_path, Path):
        atoms = read(atom_path)
    elif isinstance(atom_path, Atoms):
        atoms = atom_path
    else:
        raise TypeError("atom_path must be a Path or Atoms object")

    if scissors:
        shift_arr = generate_scissor_shifts(atom_path)
        calc = GPAW(
            mode="lcao",
            basis="dzp",
            kpts=dict(size=(kpts, kpts, 1), gamma=True),
            eigensolver={"name": "scissors", "shifts": shift_arr},
            txt="gpaw.txt",
        )
    else:
        calc = GPAW(
            mode=PW(500),  # Basis set
            xc=functional,  # Functional
            kpts={"size": (kpts, kpts, 1)},  # k-points
            occupations=FermiDirac(0.01),
            txt="gpaw.txt",
        )

    atoms.calc = calc
    atoms.get_potential_energy()

    bz_kpts = calc.get_bz_k_points()

    soc_eig = soc_eigenstates(calc)
    eigs = soc_eig.eigenvalues()
    occ = soc_eig.occupation_numbers()

    global_homo = -np.inf
    global_lumo = np.inf
    homo_kpt = None
    lumo_kpt = None

    # Loop over k-points to find global HOMO/LUMO
    for i, (kpt_eigs, kpt_occ, kpt) in enumerate(zip(eigs, occ, bz_kpts)):
        # Occupied/unoccupied masks
        occ_mask = kpt_occ > occ_thresh
        unocc_mask = kpt_occ <= occ_thresh

        if np.any(occ_mask):
            kpt_homo = np.max(kpt_eigs[occ_mask])
            if kpt_homo > global_homo:
                global_homo = kpt_homo
                homo_kpt = kpt

        if np.any(unocc_mask):
            kpt_lumo = np.min(kpt_eigs[unocc_mask])
            if kpt_lumo < global_lumo:
                global_lumo = kpt_lumo
                lumo_kpt = kpt

    return {
        "gap": global_lumo - global_homo,
        "lumo": global_lumo,
        "homo": global_homo,
        "lumo_kpt": lumo_kpt,
        "homo_kpt": homo_kpt,
    }


def scissors_gpw_file(atom_path, kpts_dens: int, gpw_file: str):
    if isinstance(atom_path, Path):
        atoms = read(atom_path)
    elif isinstance(atom_path, Atoms):
        atoms = atom_path
    else:
        raise TypeError("atom_path must be a Path or Atoms object")

    shift_arr = generate_scissor_shifts(atom_path)
    calc = GPAW(
        mode="lcao",
        basis="szp(dzp)",
        kpts=dict(density=kpts_dens, gamma=True),
        eigensolver={"name": "scissors", "shifts": shift_arr},
        txt="gpaw.txt",
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.calc.write(f"{gpw_file}.gpw", mode="all")
    return Path(f"{gpw_file}.gpw")


def LDOS(gpw_file: str | Path, symbol: str, ldos_cut_off: float = 1e-2, width=0.05):
    calc = GPAW(gpw_file)
    atoms = calc.get_atoms()

    dos = calc.dos(soc=True)

    homo_arr = []
    lumo_arr = []
    x_arr = []
    y_arr = []

    symbol_index = [
        i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == symbol
    ]

    energies = np.linspace(-3, 3, 3000)
    for i in symbol_index:
        pdos_total = np.zeros_like(energies)
        for l in range(3):
            pdos_total += dos.raw_pdos(energies, a=i, l=l, width=width)

        occ_state = energies < 0
        unocc_state = energies > 0
        non_zero_dos = pdos_total > ldos_cut_off

        occ_e = energies[occ_state & non_zero_dos]
        unocc_e = energies[unocc_state & non_zero_dos]
        homo = float(occ_e.max()) if occ_e.size else np.nan
        lumo = float(unocc_e.min()) if unocc_e.size else np.nan

        lumo_arr.append(lumo)
        homo_arr.append(homo)
        x_arr.append(float(atoms.positions[i, 0]))
        y_arr.append(float(atoms.positions[i, 1]))

    return {"x": x_arr, "y": y_arr, "homo": homo_arr, "lumo": lumo_arr}


def plot_moire_band_structure(
    data: dict[str, dict[str, list[float | None]] | float],
    grid_resolution: int = 200,
    cmap: str = "RdYlBu_r",
    figsize: tuple[float, float] = (18, 10),
):
    import matplotlib.pyplot as plt
    from scipy.interpolate import LinearNDInterpolator

    w: dict[str, list[float | None]] = data["W"]
    mo: dict[str, list[float | None]] = data["Mo"]

    w_x: np.ndarray = np.array(w["x"])
    w_y: np.ndarray = np.array(w["y"])
    w_homo: np.ndarray = np.array(w["homo"], dtype=float)
    w_lumo: np.ndarray = np.array(w["lumo"], dtype=float)

    mo_x: np.ndarray = np.array(mo["x"])
    mo_y: np.ndarray = np.array(mo["y"])
    mo_homo: np.ndarray = np.array(mo["homo"], dtype=float)
    mo_lumo: np.ndarray = np.array(mo["lumo"], dtype=float)

    # --- Build interpolators for each quantity ---
    w_homo_interp: LinearNDInterpolator = LinearNDInterpolator(
        np.column_stack([w_x, w_y]),
        w_homo,
    )
    w_lumo_interp: LinearNDInterpolator = LinearNDInterpolator(
        np.column_stack([w_x, w_y]),
        w_lumo,
    )
    mo_homo_interp: LinearNDInterpolator = LinearNDInterpolator(
        np.column_stack([mo_x, mo_y]),
        mo_homo,
    )
    mo_lumo_interp: LinearNDInterpolator = LinearNDInterpolator(
        np.column_stack([mo_x, mo_y]),
        mo_lumo,
    )

    # --- Regular grid covering the full extent of all atoms ---
    all_x: np.ndarray = np.concatenate([w_x, mo_x])
    all_y: np.ndarray = np.concatenate([w_y, mo_y])
    margin: float = 0.5
    xi: np.ndarray = np.linspace(
        all_x.min() - margin, all_x.max() + margin, grid_resolution
    )
    yi: np.ndarray = np.linspace(
        all_y.min() - margin, all_y.max() + margin, grid_resolution
    )
    Xi, Yi = np.meshgrid(xi, yi)

    # --- Interpolate onto grid ---
    W_HOMO_grid: np.ndarray = w_homo_interp(Xi, Yi)
    W_LUMO_grid: np.ndarray = w_lumo_interp(Xi, Yi)
    Mo_HOMO_grid: np.ndarray = mo_homo_interp(Xi, Yi)
    Mo_LUMO_grid: np.ndarray = mo_lumo_interp(Xi, Yi)

    # --- Interlayer gap: Mo LUMO - W HOMO at W atom positions ---
    mo_lumo_at_w: np.ndarray = mo_lumo_interp(w_x, w_y)
    interlayer_gap: np.ndarray = mo_lumo_at_w - w_homo

    gap_interp: LinearNDInterpolator = LinearNDInterpolator(
        np.column_stack([w_x, w_y]),
        interlayer_gap,
    )
    gap_grid: np.ndarray = gap_interp(Xi, Yi)

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    panels: list[tuple[np.ndarray, str, np.ndarray, np.ndarray]] = [
        (W_HOMO_grid, "W HOMO (eV)", w_x, w_y),
        (W_LUMO_grid, "W LUMO (eV)", w_x, w_y),
        (Mo_HOMO_grid, "Mo HOMO (eV)", mo_x, mo_y),
        (Mo_LUMO_grid, "Mo LUMO (eV)", mo_x, mo_y),
        (gap_grid, "Interlayer gap (eV)", w_x, w_y),
    ]

    for ax, (grid, title, sx, sy) in zip(axes.flat, panels):
        im = ax.pcolormesh(Xi, Yi, grid, cmap=cmap, shading="auto")
        ax.scatter(sx, sy, c="k", s=5, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax)

    # Hide the unused 6th panel
    axes.flat[-1].axis("off")

    fig.suptitle(
        f"Moiré band structure  |  Fermi level: {data['fermi_level']:.3f} eV (vs vacuum)",
        fontsize=14,
    )
    fig.tight_layout()

    return fig
