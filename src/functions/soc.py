from gpaw.utilities import pack_density
from ase.units import Ha
import numpy as np
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.spinorbit import soc as soc_terms


def get_soc_hamiltonian(calc):
    H, S = get_lcao_hamiltonian(calc)
    density = calc.dft.density
    D_asii = density.D_asii
    V_avii = []
    for a, D_sii in D_asii.items():
        D_sp = np.array([pack_density(D_ii.real) for D_ii in D_sii])
        xc = calc.dft.pot_calc.xc.xc
        setup = calc.dft.pot_calc.setups[a]
        soc = soc_terms(setup, xc, D_sp) * Ha
        V_avii.append(soc)

    M = len(H[0][0])
    H_kNN = []  # N = 2*M
    for wfs, H_MM in zip(calc.dft.ibzwfs, H[0]):
        H_sMsM = np.zeros((2 * M, 2 * M), dtype=complex)
        H_sMsM[:M, :M] = H_MM
        H_sMsM[M:, M:] = H_MM
        for a, V_vii in enumerate(V_avii):
            P_Mi = wfs.P_aMi[a]
            V_vMM = np.einsum("Mi, vij, Tj -> vMT", P_Mi.conj(), V_vii.conj(), P_Mi)

            x_MM, y_MM, z_MM = V_vMM
            H_sMsM[:M, :M] += z_MM
            H_sMsM[:M, M:] += x_MM + 1j * y_MM
            H_sMsM[M:, :M] += x_MM - 1j * y_MM
            H_sMsM[M:, M:] -= z_MM
    H_kNN.append(H_sMsM)

    return H_kNN
