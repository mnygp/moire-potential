import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from functions import geometry
import matplotlib.pyplot as plt
from ase.io import read

average_lattice = 3.2515
MoS2_lattice = 3.184
WSe2_lattice = 3.319
average_cell = np.array([[1, 0], [0.5, np.sqrt(3) / 2]]) * average_lattice

# Small cell data for interpolation
small_data = np.genfromtxt('../mapping/results_old.csv',
                           skip_header=1,
                           delimiter=',')

i = small_data[:, 0]
j = small_data[:, 1]
z_dist = small_data[:, 4]

x_and_y = np.stack((i, j), axis=-1)
transform_coords = x_and_y @ average_cell

small_interpolator = CloughTocher2DInterpolator(transform_coords, z_dist)


def diag(num_points, v1, v2):
    diag_vec = v1 + v2
    t = np.linspace(0, 1, num_points)

    pts = np.outer(t, diag_vec)
    return pts


resolution = 200
diag_points = diag(resolution, average_cell[0, :], average_cell[1, :])
small_interp_data = small_interpolator(diag_points)

# Large cell data for plotting
atoms = read("../../structures/MoS2-WSe2-MatterSim/1.11_2946/structure_ml.json")

large_v1 = atoms.cell[0, :2]
large_v2 = atoms.cell[1, :2]

print(large_v1, large_v2)

inter_x, inter_y, inter_distance = geometry.interlayer_distance(atoms)

x = np.linspace(0, 1, resolution)
large_diag_points = np.outer(x, large_v1 - large_v2) + large_v2
"""
large_interpolator = LinearNDInterpolator(
    np.stack((inter_x, inter_y), axis=-1), inter_distance
)
"""
large_interpolator = CloughTocher2DInterpolator(
    np.stack((inter_x, inter_y), axis=-1), inter_distance
)

large_interp_data = large_interpolator(large_diag_points)

print(large_interp_data.shape)


plt.plot(np.linspace(0, 1, resolution), small_interp_data, label='DFT small cells')
plt.plot(np.linspace(0, 1, resolution), large_interp_data, label='MatterSim structure')
plt.xlabel('Norm of lattice shift along diagonal direction [Å]')
plt.ylabel('Interlayer Distance [Å]')
plt.title('Interlayer Distance vs Lattice Shift along diagonal')
plt.legend()
plt.tight_layout()
plt.savefig('interlayer_distance_vs_shift_diagonal.png', dpi=500)
