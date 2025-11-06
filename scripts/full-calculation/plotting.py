import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from ase.io import read
from functions.util import repeate_cells

path = '../../structures/MoS2-WSe2-MatterSim/1.05_3027/structure_ml.json'
atoms = read(path)

data = np.genfromtxt('output_no_soc_3027.csv', dtype=float,
                     skip_header=1, delimiter=',')


x = data[:, 0]
y = data[:, 1]
dist = data[:, 2]
gap = data[:, 3] - data[:, 4]
correction = data[:, 4]

x, y, gap = repeate_cells(x, y, gap, range(-1, 2),
                          atoms.cell[0, :2], atoms.cell[1, :2])


# dist_interp = LinearNDInterpolator(list(zip(x, y)), dist)
gap_interp = LinearNDInterpolator(list(zip(x, y)),
                                  gap - min(gap))
# correction_interp = LinearNDInterpolator(list(zip(x, y)), correction)

X, Y = np.meshgrid(np.linspace(min(x), max(x), 800),
                   np.linspace(min(y), max(y), 800))

# interpolated_dist = dist_interp(X, Y)
interpolated_gap = gap_interp(X, Y)
# interpolated_correction = correction_interp(X, Y)


plt.figure(figsize=(6, 5))

mesh = plt.pcolormesh(X, Y, interpolated_gap, shading="auto")
plt.colorbar(mesh, label="Gap")
plt.title("Non-corrected local band gap")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("non-corrected_gap.png", dpi=500)
plt.close()
