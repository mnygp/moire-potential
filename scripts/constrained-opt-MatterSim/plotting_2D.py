import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

data = np.loadtxt('tree/write_csv_task/results.csv',
                  delimiter=',', skiprows=1)

atoms = read('../../structures/MoS2-WSe2-MatterSim/1.05_3027/structure_ml.json')

cell = atoms.get_cell()[:2, :2]

# 0: x,
# 1: y,
# 2: center x,
# 3: center y,
# 4: gap pre relax,
# 5: gap post no strain,
# 6: dist no strain,
# 7: gap post with strain,
# 8: dist with strain,
# 9: dist with ml


x = data[:, 2]
y = data[:, 3]
pre_gap = data[:, 4]
gap_no_strain = data[:, 5]
gap_with_strain = data[:, 7]

dist_ml = data[:, 9]
dist_cell_opt = data[:, 8]
dist_no_cell_opt = data[:, 6]

x_and_y = np.stack((x, y), axis=-1)
transform_coords = x_and_y @ cell.T


def scatter_plot(data, title, label, filename):
    fig, ax = plt.subplots()
    # scatter = ax.scatter(transform_coords[:, 1],
    #                     transform_coords[:, 0],
    scatter = ax.scatter(x_and_y[:, 0],
                         x_and_y[:, 1],
                         c=data,
                         cmap='cool')
    fig.colorbar(scatter, ax=ax, label=label)
    ax.set_xlabel("X Position [Å]")
    ax.set_ylabel("Y Position [Å]")
    ax.axis('equal')
    ax.set_title(title)
    fig.savefig(f"plots/{filename}.png", dpi=500)
    plt.close(fig)


scatter_plot(pre_gap - min(pre_gap),
             "Local gap pre relaxations",
             "Gap [eV]", "gap_pre")
scatter_plot(gap_no_strain - min(gap_no_strain),
             "Local gap no cell optimization",
             "Gap [eV]", "gap_no_strain")
scatter_plot(gap_with_strain - min(gap_with_strain),
             "Local gap with cell optimization",
             "Gap [eV]", "gap_with_strain")
scatter_plot((gap_no_strain - min(gap_no_strain))
             - (gap_with_strain - min(gap_with_strain)),
             "Difference in local gap: no cell opt - cell opt",
             "Gap [eV]", "gap_difference_no_cell_opt")

scatter_plot(dist_no_cell_opt - dist_cell_opt,
             "Difference in Z-distance",
             "Z-distance [Å]", "z_distance_difference")
scatter_plot(dist_ml - dist_cell_opt,
             "Z-distance: ML minus cell optimized relaxation",
             "Z-distance [Å]", "z_distance_ml")
scatter_plot(dist_ml - dist_no_cell_opt,
             "Z-distance: ML minus not cell optimized relaxation",
             "Z-distance [Å]", "z_distance_ml_no_cell_opt")
scatter_plot(dist_cell_opt,
             "Z-distance: After cell optimized relaxation",
             "Z-distance [Å]", "z_distance_cell_opt")
