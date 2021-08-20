from typing import NamedTuple

__all__ = ('SystemChange',)


class SystemChange(NamedTuple):
    index: int
    old_symb: str
    new_symb: str
    name: str
