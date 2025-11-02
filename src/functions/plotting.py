import matplotlib.pyplot as plt
import numpy as np


def contour_plot_fill(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    filename: str,
    title: str,
    color_map: str,
    label: str,
    strings=None,
    levels=100,
):
    fig, ax = plt.subplots()
    contour = ax.tricontourf(X, Y, Z, levels=levels, cmap=color_map)

    fig.colorbar(contour, ax=ax, label=label)
    ax.set_xlabel("X Position [Å]")
    ax.set_ylabel("Y Position [Å]")
    ax.axis("equal")
    ax.set_title(title)

    if strings is not None:
        text = "\n".join(strings)
        ax.text(
            0.03,
            0.17,
            text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
        )

    fig.savefig("plots/" + filename + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def contour_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    filename: str,
    title: str,
    color_map: str,
    label: str,
    strings=None,
    levels=7,
):
    level_values = np.linspace(min(Z) + 0.01, max(Z) - 0.01, levels)

    fig, ax = plt.subplots()
    contour = ax.tricontour(X, Y, Z, level_values, cmap=color_map)

    fig.colorbar(contour, ax=ax, label=label)
    ax.set_xlabel("X Position [Å]")
    ax.set_ylabel("Y Position [Å]")
    ax.axis("equal")
    ax.set_title(title)

    if strings is not None:
        text = "\n".join(strings)
        ax.text(
            0.03,
            0.17,
            text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
        )

    fig.savefig("plots/" + filename + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
