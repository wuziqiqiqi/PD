from clease.montecarlo.constraints import MCConstraint


class FixedElement(MCConstraint):
    """
    Class for forcing an element of a certiain type to stay fixed.

    Parameters:

    element: str
        Name of the element that is supposed to stay fixed
    """

    def __init__(self, element):
        self.element = element

    def __call__(self, system_changes):
        for change in system_changes:
            if change[1] == self.element or change[2] == self.element:
                return False
        return True
