import matplotlib.pyplot as plt

from helpers.plotting import *


def many_subplots():
    fig, axis = plt.subplots(1, 2, figsize=(10, 20))

    m = np.random.uniform(0, 1, size=(10, 10))
    plot_heatmap(m, ax=axis[0])

    x = list(range(100))
    plot_line(x, ax=axis[1])

    plt.show()


def many_plots():
    m = np.random.uniform(0, 1, size=(10, 10))
    plot_heatmap(m)

    x = list(range(100))
    plot_line(x)

    plt.show()
