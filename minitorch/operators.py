"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    if abs(x - y) < 1e-2:
        return 1.0
    else:
        return 0.0


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x: float) -> float:
    return math.log(x + EPS)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d * inv(x)


def inv(x: float) -> float:
    return 1 / x


def inv_back(x: float, d: float) -> float:
    return d * -1 / (x ** 2)


def relu_back(x: float, d: float) -> float:
    if x > 0:
        return d * 1.0
    else:
        return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    def apply(ls: Iterable[float]) -> Iterable[float]:
        res = []
        for num in ls:
            res.append(fn(num))
        return res
    
    return apply

def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        res = []
        for num1, num2 in zip(ls1, ls2):
            res.append(fn(num1, num2))
        return res
    
    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    def apply(ls: Iterable[float]) -> float:
        if len(ls) == 0:
            return 0
        elif len(ls) == 1:
            return ls[0]
        return fn(ls[-1], apply(ls[:len(ls) - 1]))
    
    return apply


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    return reduce(mul, 0)(ls)
