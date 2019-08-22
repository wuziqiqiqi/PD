class MCConstraint(object):
    """
    Class for that prevents the MC sampler to run certain moves
    """

    def __init__(self):
        self.name = "GenericConstraint"

    def __call__(self, system_changes):
        """Return `True` if the trial move is valid.

        Parameters:

        system_changes: list
            System changes. See doc-string of
            `clease.montecarlo.observers.MCObserver`
        """
        return True
