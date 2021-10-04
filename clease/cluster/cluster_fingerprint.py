from typing import Sequence
from functools import total_ordering
import json
import attr

__all__ = ('ClusterFingerprint',)


@total_ordering
@attr.s(eq=False, order=False)
class ClusterFingerprint:

    fp: Sequence[float] = attr.ib()
    tol: float = attr.ib(default=1e-9)

    def __lt__(self, other):
        if len(self.fp) < len(other.fp):
            return True
        if len(self.fp) > len(other.fp):
            return False

        for x, y in zip(self.fp, other.fp):
            diff = x - y
            if diff < -self.tol:
                return True
            if diff > self.tol:
                return False
        return False

    def __eq__(self, other):
        if len(self.fp) != len(other.fp):
            return False

        for x, y in zip(self.fp, other.fp):
            if abs(x - y) > self.tol:
                return False
        return True

    def __getitem__(self, i):
        return self.fp[i]

    def __len__(self):
        return len(self.fp)

    def __str__(self):
        return json.dumps(self.todict())

    def todict(self):
        return {'tol': self.tol, 'fp': self.fp}
