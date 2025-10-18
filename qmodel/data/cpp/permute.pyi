# .pyi file
import numpy as np
from typing import overload

class Permute64:
    """
    A 64-bit permutation object.
    It permute the given index to another index.
    This can be used as a shuffler for the data loader.

    This is faster by calling with numpy integer array rather than per element.

    :param length: The length of the permutation.
        Non-negative and no more than 2**63-1.
    :param seed_val: The seed value for the permutation.
        Non-negative.

    >>> from qmodel.data.permute import Permute64
    >>> import numpy as np
    >>> p = Permute64(10, 42)
    >>> [p.permute(i) for i in range(16)]
    [0, 9, 1, 7, 5, 3, 2, 8, 4, 6, 0, 9, 1, 7, 5, 3]
    >>> p.permute(np.arange(16) - 10)  # also works for negative index
    array([0, 9, 1, 7, 5, 3, 2, 8, 4, 6, 0, 9, 1, 7, 5, 3])
    >>> p2 = Permute64(2**63-1, 42)

    """

    def __init__(self, length: int, seed: int) -> None:
        ...

    @overload
    def permute(self, idx: int) -> int:
        """
        Permute the given index.

        :param idx: The index to permute. Non-negative.
            It will modulo the length of the permutation.
        :return: The permuted index.
        """
        ...

    @overload
    def permute(self, idx: np.ndarray) -> np.ndarray:
        """
        Permute the given index.

        :param idx: The index to permute. Non-negative.
            It will modulo the length of the permutation.
        :return: The permuted index.
        """
        ...
