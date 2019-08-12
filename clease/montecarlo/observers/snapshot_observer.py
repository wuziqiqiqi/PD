from clease.montecarlo.observers import MCObserver
from ase.io.trajectory import TrajectoryWriter


class Snapshot(MCObserver):
    """
    Store a snapshot in a trajectory file

    Parameters

    trajfile: str
        Filename of the trajectory file
    atoms: Atoms
        Instance of the atoms objected modofied by the MC object
    """

    def __init__(self, trajfile="default.traj", atoms=None):
        super(Snapshot, self).__init__()
        self.name = "Snapshot"
        if not trajfile.endswith(".traj"):
            msg = "This object stores all images in a trajectory file. "
            msg += "File extension should be .traj"
            raise ValueError(msg)
        if atoms is None:
            raise ValueError("No atoms object given!")
        self.atoms = atoms
        self.traj = TrajectoryWriter(trajfile, mode="w")
        self.fname = trajfile

    def __call__(self, system_changes):
        self.traj.write(self.atoms)

    def reset(self):
        self.traj = TrajectoryWriter(self.fname, mode="w")
