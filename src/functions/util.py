import numpy as np


def closest_index(position: np.ndarray,
                  particles: np.ndarray,
                  twoD: bool = True) -> int:
    if twoD:
        r_diff = particles[:, :2] - position[:2]
    else:
        r_diff = particles - position

    distances = np.sqrt(np.sum(r_diff**2, axis=1))

    return np.argsort(distances)[0]


def repeate_cells(x: np.ndarray, y: np.ndarray, data: np.ndarray,
                  n_cells: range, vector1: np.ndarray,
                  vector2: np.ndarray) -> tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray]:

    # data_original = data.copy()
    x_original = x.copy()
    y_original = y.copy()
    data_original = data.copy()

    for i in n_cells:
        for j in n_cells:
            if i == 0 and j == 0:
                continue

            data = np.concatenate((data, data_original), axis=0)
            new_x = x_original + i*vector1[0] + j*vector2[0]
            new_y = y_original + i*vector1[1] + j*vector2[1]
            x = np.concatenate((x, new_x), axis=0)
            y = np.concatenate((y, new_y), axis=0)

    return x, y, data


def check_formula(chemicals: np.ndarray):
    chem = ["W", "Se", "Mo", "S"]
    number = [1, 2, 1, 2]
    chemicals = np.array(chemicals)
    for i, j in zip(number, chem):
        if i != np.sum(chemicals == j):
            return False

    return True

# TODO: Create function to move bilayers apart
#     # Take the average position of all atoms and then move
#     # all the atoms above further up and all the atoms below further down 
