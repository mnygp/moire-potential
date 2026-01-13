from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

CONVERSION_FACTOR = 3.80998211  # hbar²/(m_e*Å²)

coefficients = {
    2: [1, -2, 1],
    4: [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12],
    6: [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
    8: [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
}


def laplacian(N, dr, order=2, conv_factor=CONVERSION_FACTOR):
    lap = lil_matrix((N * N, N * N))

    coeffs = coefficients[order]
    N_c = len(coeffs)
    shifts = range(-(N_c // 2), N_c // 2 + 1)

    for i in range(N):
        for j in range(N):
            for coeff, shift in zip(coeffs, shifts):
                new_j = (j + shift) % N
                new_i = (i + shift) % N

                lap[j + i * N, new_j + i * N] += coeff  # y direction
                lap[j + i * N, j + new_i * N] += coeff  # x direction
    else:
        return lap / dr**2


def hex_laplacian(N, dr, conv_factor, order=2):
    lap = lil_matrix((N * N, N * N))

    coeffs = coefficients[order]
    N_c = len(coeffs)
    shifts = range(-(N_c // 2), N_c // 2 + 1)

    for i in range(N):
        for j in range(N):
            for coeff, shift in zip(coeffs, shifts):
                new_j = (j + shift) % N
                new_i = (i + shift) % N

                lap[j + i * N, new_j + i * N] += coeff  # y direction
                lap[j + i * N, j + new_i * N] += coeff  # x direction
                lap[j + i * N, new_j + new_i * N] += coeff  # xy direction
        return 2 / 3 * lap / dr**2


def diag_hamiltonian(V_flat, m, dr, hexagonal, order, conv_factor=None):
    if conv_factor is None:
        conv_factor = CONVERSION_FACTOR
    if hexagonal:
        L = hex_laplacian(len(V_flat), dr, order=order, conv_factor=conv_factor)
    else:
        L = laplacian(len(V_flat), dr, order=order, conv_factor=conv_factor)
    H = -CONVERSION_FACTOR / (2 * m) * L

    # Add potential on the diagonal
    H.setdiag(H.diagonal() + V_flat)
    eigvals, eigvecs = eigsh(H, k=10, which="SM")

    return eigvals, eigvecs
