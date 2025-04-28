import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('tree/write_csv_task/results.csv',
                     delimiter=',',
                     skip_header=1)

gap = data[:, 1] - min(data[:, 1])
x = data[:, 2]
y = data[:, 3]

plt.scatter(x, y, c=gap, cmap='cool')
plt.colorbar(label='Band Gap (eV)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Band Gap Distribution')
plt.tight_layout()
plt.savefig('local_gap.png', dpi=500)
