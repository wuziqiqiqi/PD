"""This module defines the "Figure" class, which is a collection of FourVector objects."""
from typing import Iterable, Any, Tuple
import attr
from .four_vector import FourVector

__all__ = ('Figure',)


def _convert_figure(x: Any) -> Any:
    """Perform a possible type conversion on the input, prior to validation.
    We allow any iterable, which can be converted into a tuple.
    No duplication checks are made."""
    # Fast-track the tuple case, since this is the correct type of the class
    # and it is also an Iterable.
    if not isinstance(x, tuple) and isinstance(x, Iterable):
        # Try to convert into a tuple
        x = tuple(x)
    return x


@attr.s(frozen=True, order=False)
class Figure:
    """Class which defines a Figure, i.e. a collection of FourVector objects
    which defines a single cluster. Each entry in the components must be a FourVector,
    and is checked upon construction. The order of the FourVector objects is preserved,
    and is not checked for duplicate entries.

    It is possible to pass in the FourVectors in any Iterable, which will be then
    converted into a tuple, e.g.

    >>> from clease.datastructures import Figure, FourVector
    >>> fv1 = FourVector(0, 0, 0, 0)
    >>> fv2 = FourVector(1, 1, 1, 1)
    >>> Figure([fv1, fv2])
    >>> Figure((fv1, fv2))
    """

    components: Tuple[FourVector] = attr.ib(converter=_convert_figure,
                                            validator=attr.validators.instance_of(tuple))

    @components.validator
    def _validate_all_four_vectors(self, attribute, value):
        """Perform a check that all elements in the components sequence are FourVector objects"""
        # pylint: disable=unused-argument, no-self-use
        # The signature of this function is dictated by attrs.
        for ii, v in enumerate(value):
            if not isinstance(v, FourVector):
                raise TypeError(f'All values must FourVector type, got {value} '
                                f'of type {type(v)} in index {ii}.')
