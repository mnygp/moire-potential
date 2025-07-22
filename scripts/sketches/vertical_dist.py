import matplotlib.pyplot as plt
import numpy as np

# Number of atoms per layer
n_atoms = 10

# Evenly spaced x-positions
x_atoms = np.linspace(0, 10, n_atoms)
x_atoms_shifted = x_atoms + 0.5  # Mo layer shifted in x

# Layer curves using the same x-positions
z_atoms_top = 5 + 0.3 * np.sin(2 * np.pi * x_atoms / 10)
z_atoms_bottom = 2 + 0.3 * np.sin(2 * np.pi * x_atoms_shifted / 10)

# Dense curves for smooth lines
x_dense = np.linspace(0, 10, 200)
z_dense_top = 5 + 0.3 * np.sin(2 * np.pi * x_dense / 10)
z_dense_bottom = 2 + 0.3 * np.sin(2 * np.pi * (x_dense + 0.5) / 10)

# Plot
fig, ax = plt.subplots(figsize=(8, 4))

# Curved lines
ax.plot(x_dense, z_dense_top, color='purple',
        label='WSe₂ layer (W atoms)')
ax.plot(x_dense + 0.5, z_dense_bottom, color='orange',
        label='MoS₂ layer (Mo atoms)')

# Atom dots
ax.plot(x_atoms, z_atoms_top, 'o', color='purple')
ax.plot(x_atoms_shifted, z_atoms_bottom, 'o', color='orange')

# Indicator lines using existing atoms
i = 3  # index of chosen atom pair
x_w = x_atoms[i]
z_w = z_atoms_top[i]
x_mo = x_atoms_shifted[i]
z_mo = z_atoms_bottom[i]

# Vertical + horizontal connector
ax.plot([x_w, x_w], [z_w, z_mo], 'k--', label="Vertical distance")
ax.plot([x_w, x_mo], [z_mo, z_mo], 'k--', alpha=0.3)

# Formatting
ax.set_title('Illustrative Side View of MoS₂/WSe₂ Layers')
ax.set_xlabel('x (arb. units)')
ax.set_ylabel('z (arb. units)')
ax.set_aspect('equal')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("moire_side_sketch.png", dpi=500)
