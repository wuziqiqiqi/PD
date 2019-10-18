class MCObserver(object):
    """Base class for all MC observers."""

    def __init__(self):
        self.name = "GenericObserver"

    def __call__(self, system_changes):
        """
        Gets information about the system changes and can perform some action

        Parameters:

        system_changes: list
            List of system changes. For example, if the occupation of the
            atomic index 23 is changed from Mg to Al,
            system_change = [(23, Mg, Al)].
            If an Mg atom occupying the atomic index 26 is swapped with an Al
            atom occupying the atomic index 12,
            system_change = [(26, Mg, Al), (12, Al, Mg)]
        """
        pass

    def reset(self):
        """Reset all values of the MC observer"""
        pass

    def get_averages(self):
        """Return averages in the form of a dictionary."""
        return {}

    def calculate_from_scratch(self, atoms):
        """
        Method for calculating the tracked value from scratch
        (i.e. without using fast update methods)
        """
        pass
