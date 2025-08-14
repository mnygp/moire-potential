import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('band_edges.csv', skip_header=1, dtype=float, delimiter=',')

# 0: Strain
# 1: MoS2_homo
# 2: MoS2_lumo
# 3: WSe2_homo
# 4: WSe2_lumo


plt.plot(data[:, 0], data[:, 1], '-o', label='MoS2 Homo')
plt.plot(data[:, 0], data[:, 2], '-o', label='MoS2 Lumo')
plt.plot(data[:, 0], data[:, 3], '-o', label='WSe2 Homo')
plt.plot(data[:, 0], data[:, 4], '-o', label='WSe2 Lumo')
plt.legend()
plt.grid()
plt.savefig('Homo-lumo-strain.png', dpi=500)
