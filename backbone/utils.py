import tensorflow as tf
from itertools import repeat
from collections.abc import Iterable


def _n_tuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse