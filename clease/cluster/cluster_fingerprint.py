import json

__all__ = ('ClusterFingerprint',)


class ClusterFingerprint(object):

    def __init__(self, fp, tol=1E-9):
        self.fp = fp
        self.tol = tol

    def __lt__(self, other):
        if len(self.fp) < len(other.fp):
            return True
        elif len(self.fp) > len(other.fp):
            return False

        for x, y in zip(self.fp, other.fp):
            diff = x - y
            if diff < -self.tol:
                return True
            elif diff > self.tol:
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

    def fromJSON(self, data):
        self.tol = data['tol']
        self.fp = list(data['fp'])

    @staticmethod
    def load(data):
        fp = ClusterFingerprint(None)
        fp.fromJSON(data)
        return fp
