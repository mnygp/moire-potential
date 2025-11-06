import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator


average_lattice = 3.2515
MoS2_lattice = 3.184
WSe2_lattice = 3.319
average_cell = np.array([[1, 0], [0.5, np.sqrt(3) / 2]]) * average_lattice


data = np.genfromtxt('results_high_res.csv', skip_header=1, delimiter=',')

i = data[:, 0]
j = data[:, 1]
v1_norm = data[:, 2]
v2_norm = data[:, 3]
z_dist = data[:, 4]


x_and_y = np.stack((i, j), axis=-1)
transform_coords = x_and_y @ average_cell  # .T

print(transform_coords[:2, :2])


def scatter_plot(data, title, label, filename):
    fig, ax = plt.subplots()
    scatter = ax.scatter(transform_coords[:, 0],
                         transform_coords[:, 1],
                         c=data,
                         cmap='viridis')
    fig.colorbar(scatter, ax=ax, label=label)
    ax.set_xlabel("X Position [Å]")
    ax.set_ylabel("Y Position [Å]")
    ax.axis('equal')
    ax.set_title(title)
    fig.savefig(f"{filename}.png", dpi=500)
    plt.close(fig)


average_norm = (v1_norm + v2_norm) / 2
a_norm_devi = (average_norm - average_lattice) / average_lattice * 100
norm_diff = v1_norm - v2_norm

scatter_plot(average_norm, "Average Lattice Constant",
             "Average Lattice Constant (Å)",
             "average_lattice_constant")

scatter_plot(a_norm_devi, "Lattice Constant Deviation",
             "Lattice Constant Deviation (%)",
             "lattice_constant_deviation")

scatter_plot(norm_diff, "Lattice Constant asymmetry",
             "Lattice Constant Difference (Å)",
             "lattice_constant_difference")

scatter_plot(z_dist, "Interlayer Distance",
             "Z Distance (Å)",
             "z_distance")

# interpolator = LinearNDInterpolator(transform_coords, z_dist)
interpolator = CloughTocher2DInterpolator(transform_coords, z_dist)

nx = 400   # resolution in X (adjust as needed)
ny = 400   # resolution in Y

x_min, x_max = transform_coords[:, 0].min(), transform_coords[:, 0].max()
y_min, y_max = transform_coords[:, 1].min(), transform_coords[:, 1].max()

# create mesh grid
grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, nx),
    np.linspace(y_min, y_max, ny)
)

# interpolate
grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
grid_vals = interpolator(grid_points).reshape(grid_x.shape)

# plot
plt.figure()  # figsize=(6, 5))
im = plt.imshow(
    grid_vals,
    extent=(x_min, x_max, y_min, y_max),
    origin="lower",
    aspect="equal",       # very important for lattices!
)
plt.colorbar(im, label="Interlayer Distance [Å]")
plt.xlabel("X [Å]")
plt.ylabel("Y [Å]")
plt.title("Interlayer Distance Map")
plt.tight_layout()
plt.savefig("interlayer_distance_map.png", dpi=500)
