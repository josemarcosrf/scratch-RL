import numpy as np

from typing import Any

from algorithms import State
from helpers.features.fourier import FourierBasis

class FourierLinearValueFunction:
    def __init__(self, interval_len: int, n_params: int = 5):
        # Generate a random set of axis frequency vectors.
        c = np.arange(0, n_params + 1)
        self.weights = np.zeros(n_params + 1)
        self.features = FourierBasis(interval_len, n_params, c)

    def __call__(self, s:State, a:int ) -> Any:
        return np.dot(self.weights, self.features.encode(s))

    def update(self, s:State, delta: float):
        derivate_val = self.__call__(s)
        self.weights += delta * derivate_val


class SemiGradientTD:
    def __init__(self, env):
        self.env = env
