class MCObserver(object):
    """Base class for all MC observers."""

    def __init__(self):
        self.name = "GenericObserver"

    def __call__(self, system_changes):
        """
        Gets information about the system changes and can perform some action

        Parameters

        system_changes: list
            List of system changes if indx 23 changed
            from Mg to Al this argument would be
            [(23, Mg, Al)]
            If site 26 with an Mg atom is swapped with site 12 with an Al atom
            this would be
            [(26, Mg, Al), (12, Al, Mg)]
        """
        pass

    def reset(self):
        """Reset all values of the MC observer"""
        pass

    def get_averages(self):
        """Return averages in the form of a dictionary."""
        return {}
