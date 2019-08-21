from clease.montecarlo.observers import MCObserver
from ase.io.trajectory import TrajectoryWriter


class Snapshot(MCObserver):
    """Store a snapshot in a trajectory file.

    Parameters:

    fname: str
        Name of the trajectory file (without .traj extension)

    atoms: Atoms
        Instance of the atoms objected modofied by the MC object
    """

    def __init__(self, fname="snapshot", atoms=None):
        super(Snapshot, self).__init__()
        self.name = "Snapshot"
        full_fname = fname + '.traj'
        if atoms is None:
            raise ValueError("No atoms object given!")
        self.atoms = atoms
        self.traj = TrajectoryWriter(full_fname, mode="w")
        self.fname = full_fname

    def __call__(self, system_changes):
        """Write a snapshot to a .traj file.

        Parameters:

        system_changes: list
            System changes. See doc-string of
            `clease.montecarlo.observers.MCObserver`
        """
        self.traj.write(self.atoms)

    def reset(self):
        self.traj = TrajectoryWriter(self.fname, mode="w")
