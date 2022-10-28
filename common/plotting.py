import numpy as np
import matplotlib.pyplot as plt


def plot_vector_field(u: np.ndarray, v: np.ndarray, save_fpath:str = None):
    """Given 2D vectors with components U, V (x, y) plots a vector field
    and displays or saves as a png file.
    """
    h, w = u.shape[:2]  # numpy and pyplot have swapped
    x, y = np.meshgrid(np.arange(0, w, 1.0), np.arange(0, h, 1.0))

    ax = plt.axes()
    ax.set_xticks(x[0])
    ax.set_yticks([l[0] for l in y])

    # Plotting Vector Field with QUIVER
    plt.quiver(x + 0.5, y + 0.5, u, v, color="b")
    plt.title("Q table")

    # Setting x, y boundary limits
    plt.xlim(0, w)
    plt.ylim(h, 0)  # flipped y-axis

    # Show plot with grid
    plt.grid()

    if save_fpath:
        plt.savefig(save_fpath)
    else:
        plt.show()
