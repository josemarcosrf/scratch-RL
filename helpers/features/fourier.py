import numpy as np


class FourierBasis:
    def __init__(self, order: int, c: np.array) -> None:
        """Fourier Basis lienar transform.

        Args:
            order (int): The number of Fourier basis
            c (np.array): vector of shape (N+1, D) to deterine the functions frequencies along each axis
        """
        self.N = order
        self.c = c
        # Defines a function for each feature
        self.basis = [
            lambda s, i=i: np.cos(np.pi * np.dot(self.c[i], s))
            for i in range(self.N + 1)
        ]

    def encode(self, state: np.array) -> np.array:
        return np.array([b_i(state) for b_i in self.basis])


if __name__ == "__main__":
    """Creates a Fourier Basis transform instance and generates the 6 basis functions
    showed in figure 9.4, page 207 of Sutton & Barto's
    'Reinforcement Learning: An Introduction' book
    """
    import matplotlib.pyplot as plt

    N = 5
    c = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 5],
            [2, 5],
            [5, 2],
        ]
    )
    fb = FourierBasis(N, c)

    x, y = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))

    fig = plt.figure()

    for b_idx in range(0, N + 1):
        ax = fig.add_subplot(2, 3, b_idx + 1, projection="3d")

        n = x.shape[0]
        z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                state = (x[i, j], y[i, j])
                z[i, j] = fb.basis[b_idx](state)

        ax.plot_surface(x, y, z, cmap="viridis", edgecolor="green")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Fourier bases with c_{b_idx}={fb.c[b_idx]}")

    plt.show()
