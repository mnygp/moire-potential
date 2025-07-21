import matplotlib.pyplot as plt
import numpy as np

# Parameters
n_atoms = 2  # number of atoms in each direction from center
spacing = 1.0  # hexagonal grid spacing
jitter_strength = 0.3  # maximum distortion from ideal position

# Hexagonal basis vectors
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])

# Generate grid of reference positions (ideal hexagonal)
positions_ref = []
positions_shifted = []
for i in range(-n_atoms, n_atoms + 1):
    for j in range(-n_atoms, n_atoms + 1):
        pos = i * a1 + j * a2
        positions_ref.append(pos)
        # Apply small random shift (in-plane distortion)
        if not (i == 0 and j == 0):
            shift = pos + jitter_strength * (np.random.rand(2) - 0.5)
            positions_shifted.append(shift)

positions_ref = np.array(positions_ref)
positions_shifted = np.array(positions_shifted)

# Plot
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.set_aspect("equal")
ax.axis("off")

# Plot ideal positions
ax.scatter(positions_ref[:, 0], positions_ref[:, 1],
           s=80, color="gray", alpha=0.4, label="Ideal lattice")

# Plot shifted positions
ax.scatter(positions_shifted[:, 0], positions_shifted[:, 1],
           s=100, color="purple", alpha=0.8, label="Distorted atoms")

# Plot central atom
ax.scatter(0, 0, s=120, color="red", zorder=10, label="Reference atom")

ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("inplane_distortion_sketch.png", dpi=500)
plt.show()