class MCConstraint(object):
    """
    Class for that prevents the MC sampler to run certain moves
    """

    def __init__(self):
        self.name = "GenericConstraint"

    def __call__(self, system_changes):
        """Return true if the trial move is valid.

        Parameters
        system_changes:
            List of tuples with information about the changes introduced. See
            doc string of MCObserver
        """
        return True
