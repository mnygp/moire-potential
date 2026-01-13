import csv
import numpy as np
import matplotlib.pyplot as plt

z_arr = np.linspace(6, 7, 15)
strain_arr = np.linspace(0.99, 1.01, 14)

for shift in [0.0, 0.4]:
    filename = f"gap_sensitivity_shift_{shift:.2f}.csv"

    gaps = []

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = float(row["z_dist"])
            strain = float(row["strain"])

            gap_str = row["gap"].strip("()")  # remove parentheses
            first_val = gap_str.split(",")[0]  # take first number as string
            gap = float(first_val)
            gaps.append(gap)

    gap_grid = np.array(gaps).reshape(len(z_arr), len(strain_arr))

    # --- Plot ---
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        gap_grid,
        origin="lower",
        extent=[
            min((strain_arr - 1) * 100),
            max((strain_arr - 1) * 100),
            min(z_arr),
            max(z_arr),
        ],
        aspect="auto",
        cmap="viridis",
        vmin=0.2,
        vmax=1.1,
    )

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Band gap (eV)")

    # Labels
    plt.xlabel("Strain from average lattice constant [%]")
    plt.ylabel("z distance [Å]")
    plt.title(f"Band gap vs. z distance and strain (shift = {shift:.2f})")
    plt.tight_layout()
    plt.savefig(f"gap_sensitivity_shift_{shift:.2f}.png", dpi=500)
