import math
import torch
from typing import Union, Callable, List, Tuple

def exists(x: Union[None, object]) -> bool:
    """
    Checks if a value is not None.

    Args:
        x: The value to check.

    Returns:
        bool: True if `x` is not None, otherwise False.
    """
    return x is not None

def noop(*args, **kwargs) -> None:
    """
    A no-op function, does nothing.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        None
    """
    pass

def is_odd(n: int) -> bool:
    """
    Checks if a number is odd.

    Args:
        n: The number to check.

    Returns:
        bool: True if `n` is odd, otherwise False.
    """
    return (n % 2) == 1

def default(val: Union[None, object], d: Union[object, Callable[[], object]]) -> object:
    """
    Returns the value `val` if it is not None, otherwise returns the default value `d`.
    If `d` is callable, it will be called to get the default value.

    Args:
        val: The value to check.
        d: The default value or callable to return if `val` is None.

    Returns:
        The value `val` or the default value `d`.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    """
    Cycles through the data loader indefinitely.

    Args:
        dl: The data loader to cycle.

    Yields:
        The next batch of data from the data loader.
    """
    while True:
        for data in dl:
            yield data

def num_to_groups(num: int, divisor: int) -> List[int]:
    """
    Splits a number into groups of a specified divisor.

    Args:
        num: The total number to split.
        divisor: The number to divide `num` into groups by.

    Returns:
        List[int]: A list of group sizes.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape: Tuple[int, ...], prob: float, device: torch.device) -> torch.Tensor:
    """
    Generates a mask tensor with a specified probability.

    Args:
        shape: The shape of the tensor to generate.
        prob: The probability of each element being True (in the range [0, 1]).
        device: The device to place the tensor on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor with the same shape as `shape`, containing True with 
                      probability `prob` and False otherwise.
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

def is_list_str(x: Union[List[object], Tuple[object, ...]]) -> bool:
    """
    Checks if an object is a list or tuple of strings.

    Args:
        x: The object to check.

    Returns:
        bool: True if `x` is a list or tuple and all elements are strings, otherwise False.
    """
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])
