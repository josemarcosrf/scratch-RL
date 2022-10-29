import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FIG_SIZE = (15, 10)


def plot_heatmap(
    m,
    title: str = None,
    row_labels: str = None,
    col_labels: str = None,
    save_fpath: str = None,
    ax=None,
) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    # heatmap = ax.pcolor(m, cmap=plt.cm.Blues)

    ax.set_yticks(np.arange(m.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(m.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()

    if row_labels:
        ax.set_yticklabels(row_labels, minor=False)
    if col_labels:
        ax.set_xticklabels(col_labels, minor=False)

    ax.set_title(title)
    ax.grid()
    # ax.colorbar(heatmap)

    if save_fpath:
        plt.savefig(save_fpath)
    else:
        # plt.show()
        return ax


def plot_line(x, title: str = None, save_fpath: str = None, ax=None) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    ax.plot(x)
    ax.set_title(title)
    # ax.set_xticks(range(len(x)))

    if save_fpath:
        plt.savefig(save_fpath)
    else:
        # plt.show()
        return ax


def plot_vector_field(
    u: np.ndarray, v: np.ndarray, title: str = None, save_fpath: str = None, ax=None
):
    """Given 2D vectors with components U, V (x, y) plots a vector field
    and displays or saves as a png file.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    h, w = u.shape[:2]  # numpy and pyplot have swapped axis
    x, y = np.meshgrid(np.arange(0, w, 1.0), np.arange(0, h, 1.0))

    ax.set_xticks(x[0])
    ax.set_yticks([p[0] for p in y])  # noqa

    # Plotting Vector Field with QUIVER
    ax.quiver(x + 0.5, y + 0.5, u, v, color="b")

    # Setting x, y boundary limits
    ax.xlim(0, w)
    ax.ylim(h, 0)  # flipped y-axis

    ax.set_title(title)
    ax.grid()

    if save_fpath:
        plt.savefig(save_fpath)
    else:
        # plt.show()
        return ax
