import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FIG_SIZE = (20, 10)


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

    ax.pcolor(m, cmap=plt.cm.Blues)
    # ax.colorbar(heatmap)

    ax.set_yticks(np.arange(m.shape[0]), minor=False)
    ax.set_xticks(np.arange(m.shape[1]), minor=False)
    ax.invert_yaxis()

    if row_labels:
        ax.set_yticklabels(row_labels, minor=False)
    if col_labels:
        ax.set_xticklabels(col_labels, minor=False)

    ax.set_title(title)
    ax.grid()

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


def plot_stats(stats, env_shape):
    fig, axis = plt.subplots(1, 3, figsize=(20, 10))
    plot_line(stats["ep_rewards"], title="Episode rewards", ax=axis[0])
    plot_line(stats["ep_length"], title="Episode length", ax=axis[1])

    if len(env_shape) == 2:
        state_visits = np.sum(stats["visits"], axis=-1).reshape(env_shape)
        plot_heatmap(state_visits, title="state visits", ax=axis[2])

    plt.show()
