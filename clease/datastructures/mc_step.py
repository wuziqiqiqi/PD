import attr
from clease.jsonio import AttrSavable, jsonable
from .system_changes import SystemChanges

__all__ = ("MCStep",)


def _compare_moves(move1: SystemChanges, move2: SystemChanges) -> bool:
    """Helper function to compare equality between system changes."""
    if len(move1) != len(move2):
        return False
    for change1, change2 in zip(move1, move2):
        if change1 != change2:
            return False
    return True


@jsonable("mc_step")
@attr.define(eq=True, slots=True)
class MCStep(AttrSavable):
    """Container with information about a single MC step.
    No validation checks are made in this class for performance reasons.
    """

    step: int = attr.field()
    energy: float = attr.field()
    move_accepted: bool = attr.field()
    # last_move may be a tuple or list for example, so ensure we can compare them
    last_move: SystemChanges = attr.field(
        eq=attr.cmp_using(eq=_compare_moves, require_same_type=False)
    )

    @property
    def move_rejected(self) -> bool:
        return not self.move_accepted
