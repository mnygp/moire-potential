import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('tree/write_csv_task/gap_results.csv',
                     delimiter=',',
                     skip_header=1)

raw_gap = data[:, 1]
x = data[:, 2]
y = data[:, 3]

fig1, ax1 = plt.subplots()
sc1 = ax1.scatter(x, y, c=raw_gap, cmap='cool')
fig1.colorbar(sc1, label='Band Gap (eV)', ax=ax1)
ax1.set_xlabel('X Coordinate')
ax1.set_ylabel('Y Coordinate')
ax1.set_title('Local Band Gap from ML relaxed sturucture')
fig1.tight_layout()
fig1.savefig('local_gap.png', dpi=500)

data_dist = np.genfromtxt('tree/write_extra_space_csv_task' +
                          '/gap_extra_space_results.csv',
                          delimiter=',',
                          skip_header=1)

extra_dist_gap = data_dist[:, 1]

fig2, ax2 = plt.subplots()
sc2 = ax2.scatter(x, y, c=extra_dist_gap, cmap='cool')
fig2.colorbar(sc2, label='Band Gap (eV)', ax=ax2)
ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
ax2.set_title('Local Band Gap for large interlayer distance')
fig2.tight_layout()
fig2.savefig('extra_dist_gap.png', dpi=500)

fig3, ax3 = plt.subplots()
sc3 = ax3.scatter(x, y, c=(raw_gap - extra_dist_gap), cmap='cool')
fig3.colorbar(sc3, label='Band Gap (eV)', ax=ax3)
ax3.set_xlabel('X Coordinate')
ax3.set_ylabel('Y Coordinate')
ax3.set_title('Band gap shift from layer hybridization')
fig3.tight_layout()
fig3.savefig('Hybridization.png', dpi=500)
