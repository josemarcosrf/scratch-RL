from typing import Tuple

import numpy as np


def linear_to_multi_index(idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(d.item() for d in np.unravel_index([idx], shape))


def multi_to_linear_index(idx_tuple: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    return np.ravel_multi_index(idx_tuple, shape)
