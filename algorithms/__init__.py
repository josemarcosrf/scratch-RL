from functools import wraps
from typing import Any
from typing import Callable
from typing import Tuple
from typing import Union


State = Union[int, Tuple[Any, ...]]


def fix_state(state) -> State:
    if isinstance(state, tuple):
        return tuple(map(int, state))

    return int(state)


def state_as_int(func: Callable) -> Callable:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @wraps(func)
    def mapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, tuple):
                new_args.append(fix_state(arg))
            else:
                new_args.append(arg)

        new_kwargs = {}
        for k, arg in kwargs.items():
            if isinstance(arg, tuple):
                new_kwargs[k] = fix_state(arg)
            else:
                new_kwargs[k] = arg

        return func(*new_args, **new_kwargs)

    return mapper
