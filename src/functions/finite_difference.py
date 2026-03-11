import numpy as np
from scipy.constants import e, hbar, m_e
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

CONVERSION_FACTOR = hbar**2 / (m_e * e) * 1e20  # hbar²/(m_e*Å²)

coefficients = {
    2: [1, -2, 1],
    4: [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12],
    6: [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
    8: [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
}


def laplacian(N, dr, order=2, kx=0, ky=0):
    lap = lil_matrix((N * N, N * N), dtype=complex)

    coeffs = coefficients[order]
    N_c = len(coeffs)
    shifts = range(-(N_c // 2), N_c // 2 + 1)

    for i in range(N):
        for j in range(N):
            for coeff, shift in zip(coeffs, shifts):
                i_s = i + shift
                j_s = j + shift
                new_j = j_s % N
                new_i = i_s % N

                lap[j + i * N, new_j + i * N] += coeff * np.exp(
                    1j * ky * shift * dr
                )  # y direction
                lap[j + i * N, j + new_i * N] += coeff * np.exp(
                    1j * kx * shift * dr
                )  # x direction
    else:
        return lap / dr**2


def hex_laplacian(N, dr, order=2, kx=0, ky=0):
    lap = lil_matrix((N * N, N * N), dtype=complex)

    coeffs = coefficients[order]
    N_c = len(coeffs)
    shifts = range(-(N_c // 2), N_c // 2 + 1)

    # Primitive lattice vectors for hexagonal grid
    # a1 = dr * (1, 0)
    # a2 = dr * (1/2, sqrt(3)/2)
    # The three finite-difference directions and their real-space displacement vectors:
    #   x-dir:   Δr = shift * a1               = shift * dr * (1, 0)
    #   y-dir:   Δr = shift * a2               = shift * dr * (1/2, sqrt(3)/2)
    #   xy-dir:  Δr = shift * (a1 - a2)        = shift * dr * (1/2, -sqrt(3)/2)
    #
    # Bloch phase = exp(i * k⃗ · Δr⃗)

    sqrt3 = np.sqrt(3)

    for i in range(N):
        for j in range(N):
            for coeff, shift in zip(coeffs, shifts):
                new_j = (j + shift) % N
                new_i = (i + shift) % N

                # x-direction: displacement = shift * dr * (1, 0)
                phase_x = np.exp(1j * kx * shift * dr)
                lap[j + i * N, new_j + i * N] += coeff * phase_x

                # y-direction: displacement = shift * dr * (1/2, sqrt(3)/2)
                phase_y = np.exp(1j * (kx * 0.5 + ky * sqrt3 / 2) * shift * dr)
                lap[j + i * N, j + new_i * N] += coeff * phase_y

                # xy-direction: displacement = shift * dr * (1/2, -sqrt(3)/2)
                # This is a1 - a2, so both i and j shift together
                phase_xy = np.exp(1j * (kx * 0.5 - ky * sqrt3 / 2) * shift * dr)
                lap[j + i * N, new_j + new_i * N] += coeff * phase_xy

    return 2 / 3 * lap / dr**2


def diag_hamiltonian(
    V, m, dr, hexagonal, order, eigvals=10, kx=0, ky=0, conv_factor=None
):
    assert V.shape[0] == V.shape[1], "This code only works on regular square grids"

    if conv_factor is None:
        conv_factor = CONVERSION_FACTOR

    if hexagonal:
        L = hex_laplacian(V.shape[0], dr, order=order, kx=kx, ky=ky)
    else:
        L = laplacian(V.shape[0], dr, order=order, kx=kx, ky=ky)
    H = -CONVERSION_FACTOR / (2 * m) * L

    # Add potential on the diagonal
    H.setdiag(H.diagonal() + V.flatten())
    eigvals, eigvecs = eigsh(H, k=eigvals, which="SA")

    return eigvals, eigvecs
