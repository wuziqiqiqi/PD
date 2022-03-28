from __future__ import annotations
from typing import List, Dict
import attr
import numpy as np

# Constant used for indicating a missing value in the NumPy array.
# Indicates a background index.
_MISSING = -1

__all__ = ("TransMatrix",)


@attr.define(eq=True, order=False)
class TransMatrix:
    """Data class wrapper for the translation matrix."""

    trans_matrix: List[Dict[int, int]] = attr.field()

    def key_array(self) -> np.ndarray:
        """The keys of the trans matrix in NumPy format."""
        return self.as_array()[:, :, 0]

    def value_array(self) -> np.ndarray:
        """The values of the trans matrix in NumPy format."""
        return self.as_array()[:, :, 1]

    def as_array(self) -> np.ndarray:
        """Convert the trans matrix into its NumPy array format.
        "Missing" values are designated the value of -1.

        This representation is more convenient for storage.
        """
        arr = np.zeros((len(self), self.n_indices, 2), dtype=np.dtype(int))
        for ii, dct in enumerate(self.trans_matrix):
            if len(dct) == 0:
                # Populate with -1's, indiate missing, i.e. background atoms
                arr[ii, :, :] = _MISSING
                continue
            arr[ii, :, 0] = list(dct.keys())
            arr[ii, :, 1] = list(dct.values())
        return arr

    @property
    def n_indices(self) -> int:
        """Number of index pairs for indices which are not background."""
        # Some entries may have 0 length, if they are background atoms
        # So we have to check the lengths. The validation already checked
        # we only have at most 2 possible values.
        try:
            return max(len(dct) for dct in self.trans_matrix)
        except ValueError:
            # Trans matrix has length 0
            return 0

    @property
    def n_sites(self) -> int:
        """The number of sites which were used to build the trans matrix."""
        return len(self)

    @trans_matrix.validator
    def _validate_trans_matrix(self, attribute, value):
        # pylint: disable=unused-argument, no-self-use
        if not isinstance(value, list):
            raise TypeError(f"Expected the trans matrix as a list, got {value!r}")
        lens = set(len(dct) for dct in value)
        if len(lens) not in {0, 1, 2}:
            # length 0: No entries in the TM
            # length 1: No background atoms, every entry is filled
            # length 2: Background atoms has a length 0, so we should have 0's
            #  and then whatever number of pairs
            raise ValueError(f"Got too many different lengths, expected 0, 1 or 2, got {len(lens)}")

    @classmethod
    def from_array(cls, array: np.ndarray) -> TransMatrix:
        """Construct the TransMatrix from it's NumPy array representation,
        as exported from the "as_array" method.
        """
        # Sanity checks
        # We sometimes load this object from a h5py dataset, which is just a wrapper around the
        # numpy array, so we want to avoid making a copy into an entirely new NumPy array,
        # since there's no need.
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Expected a NumPy array, got {array!r}")
        if array.ndim != 3:
            raise ValueError(f"Wrong number of dimensions, expected 3, got {array.ndim}")
        if array.shape[-1] != 2:
            raise ValueError(f"Last dimension should be 2, got {array.shape[-1]}")

        def _convert_array(arr: np.ndarray) -> dict:
            """Helper function to convert the array."""
            if (arr == _MISSING).any():
                # This is a background atom, every entry should be "MISSING"
                assert (arr == _MISSING).all()
                return {}
            # "zip" the last dimension together as key-value pairs into a dict
            return dict(arr)

        tm = list(map(_convert_array, array))
        return cls(tm)

    def __len__(self) -> int:
        return len(self.trans_matrix)

    def __getitem__(self, i):
        return self.trans_matrix[i]
