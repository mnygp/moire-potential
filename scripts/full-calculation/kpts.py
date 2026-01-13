import numpy as np


data = np.genfromtxt(
    "ktps_fixed_cell_fixed_TM_scissors.csv", dtype=float, skip_header=1, delimiter=","
)

K_point = True

for row in data:
    if np.sign(row[2]) != np.sign(row[3]):
        K_point = False
        break

    if np.sign(row[5]) != np.sign(row[6]):
        K_point = False
        break

if K_point:
    print("All calculations are at the K point.")
else:
    print("Some calculations are NOT at the K point.")
