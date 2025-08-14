import numpy as np
import ast
import matplotlib.pyplot as plt


average_lattice = 3.2515
MoS2_lattice = 3.184
WSe2_lattice = 3.319
average_cell = np.array([[1, 0], [-0.5, np.sqrt(3) / 2]]) * average_lattice


data = np.genfromtxt('results.csv', skip_header=1, delimiter=',', dtype=str)

# Convert columns to correct types
i = data[:, 0].astype(float)
j = data[:, 1].astype(float)
v1_norm = data[:, 2].astype(float)
v2_norm = data[:, 3].astype(float)

# Parse string list into float
z_dist = np.array([ast.literal_eval(x)[0] for x in data[:, 4]])


x_and_y = np.stack((i, j), axis=-1)
transform_coords = x_and_y @ average_cell # .T

print(transform_coords[:2, :2])


def scatter_plot(data, title, label, filename):
    fig, ax = plt.subplots()
    scatter = ax.scatter(transform_coords[:, 0],
                         transform_coords[:, 1],
                         c=data,
                         cmap='cool')
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

print(f'Average norm: {np.mean(average_norm)} +/- {np.std(average_norm)}')
MoS2_devi = (average_norm - MoS2_lattice) * 100 / MoS2_lattice
WSe2_devi = (average_norm - WSe2_lattice) * 100 / WSe2_lattice
print(f'Average deviation from MoS2 lattice {np.mean(MoS2_devi):.3f}')
print(f'Average deviation from WSe2 lattice {np.mean(WSe2_devi):.3f}')
