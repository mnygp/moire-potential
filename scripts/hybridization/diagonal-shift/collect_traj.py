from ase.io import Trajectory, write
import numpy as np


traj_files = [f'traj_files/opt_3.32_{shift:.2f}.traj'
              for shift in np.linspace(0, 1, 30)]

last_images = []
for fname in traj_files:
    with Trajectory(fname) as traj:
        last_images.append(traj[-1])   # take last frame

# Write all last images into one trajectory file
write("collected_last_images_3.32.traj", last_images)
