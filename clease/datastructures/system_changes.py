from typing import Sequence, Tuple
import typing
import ase
from ase.data import atomic_numbers
import attr
from clease.jsonio import AttrSavable, jsonable

__all__ = ("SystemChange", "SystemChanges")


def _apply_symbol_to_index(atoms: ase.Atoms, index: int, symbol: str) -> None:
    """Faster method for changing a single symbol at a given index."""
    number = atomic_numbers[symbol]
    atoms.numbers[index] = number


def _apply_position_change(atoms: ase.Atoms, index: int, new_position: Tuple[float, float, float]) -> None:
    """Apply a change to the position of an atom at a given index."""
    atoms.positions[index] = new_position


@jsonable("system_change")
@attr.define(eq=True, slots=True)
class SystemChange(AttrSavable):
    """Representation of a change to the system."""

    index: int = attr.field()
    old_symb: str = attr.field(default=None)
    new_symb: str = attr.field(default=None)
    name: str = attr.field(eq=False, default="")
    old_position: Tuple[float, float, float] = attr.field(default=None)
    new_position: Tuple[float, float, float] = attr.field(default=None)
    
    def apply_change(self, atoms: ase.Atoms) -> None:
        """Apply the change to the atoms object."""
        if self.new_symb is not None:
            _apply_symbol_to_index(atoms, self.index, self.new_symb)
        if self.new_position is not None:
            _apply_position_change(atoms, self.index, self.new_position)
            atoms.wrap()

    def undo_change(self, atoms: ase.Atoms) -> None:
        """Undo the change to the atoms object."""
        if self.old_symb is not None:
            _apply_symbol_to_index(atoms, self.index, self.old_symb)
        if self.old_position is not None:
            _apply_position_change(atoms, self.index, self.old_position)

    def astuple(self) -> Tuple[int, str, str, Tuple[float, float, float], Tuple[float, float, float], str]:
        """Return the tuple representation of the SystemChange."""
        return attr.astuple(self, recurse=False)


# Type alias for multiple SystemChange objects
SystemChanges = Sequence[SystemChange]


def get_inverted_changes(
    system_changes: SystemChanges,
) -> typing.Iterator[SystemChange]:
    """Invert system changes by doing old_symbs -> new_symbs and old_positions -> new_positions."""
    yield from (
        SystemChange(
            index=change.index,
            old_symb=change.new_symb,
            new_symb=change.old_symb,
            old_position=change.new_position,
            new_position=change.old_position,
            name=change.name,
        )
        for change in system_changes
    )