import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(
    4, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1]}
)

lattice = [3.2515]

for a, label in zip(lattice, ["3.25"]):
    data = np.loadtxt(f"gap_shift_{a:.2f}_lcao.csv", delimiter=",", skiprows=1)

    shift_arr = data[:, 0]
    z_dist_arr = data[:, 1]
    gap_arr = data[:, 2]
    gap_arr_soc = data[:, 3]

    ax[0].plot(shift_arr, gap_arr - min(gap_arr), "-o", markersize=4, label=label)
    ax[0].plot(
        shift_arr,
        gap_arr_soc - min(gap_arr_soc),
        "-o",
        markersize=4,
        label=label + " SOC",
    )

    ax[1].plot(shift_arr, gap_arr, "-o", markersize=4, label=label)
    ax[1].plot(shift_arr, gap_arr_soc, "-o", markersize=4, label=label + " SOC")

    ax[2].plot(shift_arr, z_dist_arr, "-o", markersize=4, label=label, color="C3")

    data = np.loadtxt(f"gap_shift_{a:.2f}.csv", delimiter=",", skiprows=1)
    shift_arr = data[:, 0]
    z_dist_arr = data[:, 1]
    gap_arr = data[:, 2]

    ax[3].plot(shift_arr, gap_arr - min(gap_arr), "-o", markersize=4, label=label)

ax[0].set_ylabel("BG [eV] normed")
ax[0].grid()
ax[2].set_xlabel("Shift")
ax[1].set_ylabel("BG [eV]")
ax[1].legend()
ax[1].grid()
ax[2].set_ylabel("Inter dist [Å]")
ax[2].grid()
ax[3].set_ylabel("BG [eV] normed (PW)")
ax[3].grid()
fig.tight_layout()
fig.savefig("gap_shift_plot_norm_lcao.png", dpi=500)


fig2, ax2 = plt.subplots(
    2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

data = np.loadtxt(
    f"gap_shift_{a:.2f}_lcao_high_fidelity.csv", delimiter=",", skiprows=1
)

shift_arr = data[:, 0]
z_dist_arr = data[:, 1]
gap_arr = data[:, 2]
gap_arr_soc = data[:, 3]
gap_scissors_arr = data[:, 4]
gap_scissors_arr_soc = data[:, 5]

ax2[0].plot(shift_arr, gap_arr - min(gap_arr), "-o", markersize=4, label="LCAO")
ax2[0].plot(
    shift_arr, gap_arr_soc - min(gap_arr_soc), "-o", markersize=4, label="LCAO SOC"
)
ax2[0].plot(
    shift_arr,
    gap_scissors_arr - min(gap_scissors_arr),
    "-o",
    markersize=4,
    label="LCAO Scissors",
)
ax2[0].plot(
    shift_arr,
    gap_scissors_arr_soc - min(gap_scissors_arr_soc),
    "-o",
    markersize=4,
    label="LCAO Scissors SOC",
)
ax2[0].set_ylabel("BG [eV] normed")
ax2[0].grid()
ax2[0].legend()


ax2[1].plot(shift_arr, z_dist_arr, "-o", markersize=4, label="Z dist", color="C6")
ax2[1].set_ylabel("Inter dist [Å]")
ax2[1].grid()
ax2[1].set_xlabel("Shift")
fig2.tight_layout()
fig2.savefig("gap_shift_plot_lcao_high_fid.png", dpi=500)
# fig2.close()

(fig3, ax3) = plt.subplots(1, 1, figsize=(6, 6))

data = np.loadtxt(
    f"gap_shift_{a:.2f}_lcao_constant_z_6.25.csv", delimiter=",", skiprows=1
)

shift_arr = data[:, 0]
z_dist_arr = data[:, 1]
gap_arr = data[:, 2]
gap_arr_soc = data[:, 3]
gap_scissors_arr = data[:, 4]
gap_scissors_arr_soc = data[:, 5]

ax3.plot(shift_arr, gap_arr - min(gap_arr), "-o", markersize=4, label="LCAO")
ax3.plot(
    shift_arr, gap_arr_soc - min(gap_arr_soc), "-o", markersize=4, label="LCAO SOC"
)
ax3.plot(
    shift_arr,
    gap_scissors_arr - min(gap_scissors_arr),
    "-o",
    markersize=4,
    label="LCAO Scissors",
)
ax3.plot(
    shift_arr,
    gap_scissors_arr_soc - min(gap_scissors_arr_soc),
    "-o",
    markersize=4,
    label="LCAO Scissors SOC",
)
ax3.set_ylabel("BG [eV] normed")
ax3.grid()
ax3.legend()
ax3.set_xlabel("Shift")
fig3.tight_layout()
fig3.savefig("gap_shift_plot_lcao_constant_z_6.25.png", dpi=500)
