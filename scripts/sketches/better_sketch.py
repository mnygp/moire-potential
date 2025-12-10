import matplotlib.pyplot as plt
from ase.io import read
import numpy as np
from scipy.spatial import KDTree


p = '../../structures/MoS2-WSe2-MatterSim/1.11_2946/structure_ml.json'
atoms = read(p)

W_atoms = atoms[[atom.symbol == 'W' for atom in atoms]]
Mo_atoms = atoms[[atom.symbol == 'Mo' for atom in atoms]]

# #################### Z dist plot ##################################

W_filtered = W_atoms[[atom.position[1] < 2 for atom in W_atoms]]
Mo_filtered = Mo_atoms[[atom.position[1] < 2 for atom in Mo_atoms]]

W_pos_x = W_filtered.positions[:, 0]
W_pos_z = W_filtered.positions[:, 2]

Mo_pos_x = Mo_filtered.positions[:, 0]
Mo_pos_z = Mo_filtered.positions[:, 2]

plt.figure(figsize=(12, 2.5))
plt.axes(aspect='equal')
plt.plot(W_pos_x, W_pos_z, '-o', label='W atoms')
plt.plot(Mo_pos_x, Mo_pos_z, '-o', label='Mo atoms')
W_i = 4
Mo_i = 5
plt.vlines([W_pos_x[W_i]], Mo_pos_z[Mo_i], W_pos_z[W_i],
           color='black', linestyles='--', label='Interlayer distance')
plt.hlines([Mo_pos_z[Mo_i]], W_pos_x[W_i], Mo_pos_x[Mo_i],
           linestyles='--', alpha=0.6)
plt.title('Definition of Interlayer distance')
plt.xlabel('X-Position [Å]')
plt.ylabel('Z-Position [Å]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('better_vert_dist.png', dpi=500)
plt.close()

# ################### Distorte lattice plot ###########################
ref_i = 100
dist_x = W_atoms.positions[:, 0] - W_atoms.positions[ref_i, 0]
dist_y = W_atoms.positions[:, 1] - W_atoms.positions[ref_i, 1]
# plt.scatter(dist_x, dist_y, label='Distorted lattice')

a = 3.319

nx, ny = 10, 8
x_coords = []
y_coords = []
for j in range(ny):
    for i in range(nx):
        x = i * a + (j % 2) * (a / 2)
        y = j * (np.sqrt(3) / 2) * a
        x_coords.append(x)
        y_coords.append(y)

# Convert to numpy arrays
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

ref2_i = 34

ideal_x = x_coords-x_coords[ref2_i]
ideal_y = y_coords-y_coords[ref2_i]

theta_deg = 26  # rotation angle in degrees (example)
theta = np.radians(theta_deg)  # convert to radians

# Apply rotation
rot_x = ideal_x * np.cos(theta) - ideal_y * np.sin(theta)
rot_y = ideal_x * np.sin(theta) + ideal_y * np.cos(theta)


ideal_points = np.column_stack((rot_x, rot_y))
dist_points = np.column_stack((dist_x, dist_y))

tree = KDTree(ideal_points)
dists, indices = tree.query(dist_points)

# --- Compute actual distortion vectors ---
dx = dist_x - rot_x[indices]
dy = dist_y - rot_y[indices]

# --- Scale the distortions to make them visible ---
scale = 30  # try 10–100 depending on how big you want it
amplified_x = dist_x + scale * dx
amplified_y = dist_y + scale * dy



plt.scatter(rot_x, rot_y, label='Ideal lattice')
plt.scatter(amplified_x, amplified_y, label='Distorted lattice \n (Amplifiedx30)')
plt.scatter(0, 0, color='red', label='Reference atom')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Hexagonal Lattice Pattern')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-7.5, 7.5)
plt.ylim(-7.5, 7.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('better_lattice_distortion.png', dpi=500)
