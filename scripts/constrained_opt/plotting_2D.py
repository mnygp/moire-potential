import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.build import mx2
from functions.heat_map import interlayer_distance

data = np.loadtxt('tree/write_csv_task/results.csv',
                  delimiter=',', skiprows=1)

atoms = read('../../structures/1.05_3027/structure_ml.json')

MoS2_len = 3.184
WSe2_len = 3.319
average_len = (MoS2_len + WSe2_len)/2

MoS2 = mx2('MoS2', a=average_len, vacuum=6.0)

cell = MoS2.get_cell()[:2, :2]
x_and_y = data[:, :2]

transform_coords = x_and_y @ cell.T

x = data[:, 0]
y = data[:, 1]
distance = abs(data[:, 2])
pre_gap = data[:, 3]
post_gap = data[:, 4]

fig, ax = plt.subplots()
contour = ax.scatter(transform_coords[:, 1],
                     transform_coords[:, 0],
                     c=post_gap,
                     cmap="cool")
fig.colorbar(contour, ax=ax, label="Gap [eV]")
ax.set_xlabel("X Position [Å]")
ax.set_ylabel("Y Position [Å]")
ax.axis('equal')
ax.set_title("Local gap post relaxations")
fig.savefig("gap.png", dpi=500)
plt.close(fig)

fig, ax = plt.subplots()
contour = ax.scatter(transform_coords[:, 1],
                     transform_coords[:, 0],
                     c=distance,
                     cmap="cool")
fig.colorbar(contour, ax=ax, label="Z-distance [Å]")
ax.set_xlabel("X Position [Å]")
ax.set_ylabel("Y Position [Å]")
ax.axis('equal')
ax.set_title("Distance between TMs")
fig.savefig("distance.png", dpi=500)
plt.close(fig)

atoms = read('../../structures/1.05_3027/structure_ml.json')

x_i, y_i, z_i = interlayer_distance(atoms)
fig4, ax4 = plt.subplots()
sc4 = ax4.scatter(x_i, y_i, c=z_i, cmap='cool')
fig4.colorbar(sc4, label='Interlayer Distance (Å)', ax=ax4)
ax4.set_xlabel('X Coordinate')
ax4.set_ylabel('Y Coordinate')
ax4.set_title('Interlayer Distance from ML relaxed structure')
fig4.tight_layout()
fig4.savefig('interlayer_distance.png', dpi=500)
plt.close(fig4)
