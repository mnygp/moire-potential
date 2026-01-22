import matplotlib.pyplot as plt
import numpy as np

a = 3.25

for name, title in zip(
    [f"gap_shift_{a:.2f}_lcao_before_relax", f"gap_shift_{a:.2f}_lcao_after_relax"],
    ["Band gap before relaxation", "band gap after relaxation"],
):
    fig, ax = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    data = np.loadtxt(name + ".csv", delimiter=",", skiprows=1)

    shift_arr = data[:, 0]
    z_dist_arr = abs(data[:, 1])
    gap_arr = data[:, 2]
    gap_arr_soc = data[:, 3]
    gap_scissors_arr = data[:, 4]
    gap_scissors_arr_soc = data[:, 5]

    ax[0].plot(shift_arr, gap_arr - min(gap_arr), "-o", markersize=4, label="LCAO")
    ax[0].plot(
        shift_arr, gap_arr_soc - min(gap_arr_soc), "-o", markersize=4, label="LCAO SOC"
    )
    ax[0].plot(
        shift_arr,
        gap_scissors_arr - min(gap_scissors_arr),
        "-o",
        markersize=4,
        label="LCAO Scissors",
    )
    ax[0].plot(
        shift_arr,
        gap_scissors_arr_soc - min(gap_scissors_arr_soc),
        "-o",
        markersize=4,
        label="LCAO Scissors SOC",
    )
    ax[0].set_ylabel("BG [eV] normed")
    ax[0].grid()
    ax[0].set_title(title)
    ax[0].legend()

    ax[1].plot(shift_arr, z_dist_arr, "-o", markersize=4, label="Z dist", color="C6")
    ax[1].set_ylabel("Inter dist [Å]")
    ax[1].grid()
    ax[1].set_xlabel("Shift")
    fig.tight_layout()
    fig.savefig(name + ".png", dpi=500)
    # fig2.close()
