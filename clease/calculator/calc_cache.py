import numpy as np
from ase.calculators.calculator import Calculator, PropertyNotImplementedError


class CleaseCacheCalculator(Calculator):
    """This is a simple clease calculator cache object,
    similar to the ASE singlepoint calculator, but less restrictive"""

    name = "clease_cache"

    def __init__(self, **results):
        super().__init__()
        self.results = {}

        self.results.update(**results)

    def __str__(self) -> str:
        tokens = []
        for key, val in sorted(self.results.items()):
            if np.isscalar(val):
                txt = f"{key}={val}"
            else:
                txt = f"{key}=..."
            tokens.append(txt)
        return "{}({})".format(self.__class__.__name__, ", ".join(tokens))

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.results:
            raise PropertyNotImplementedError("The property '{name}' is not available.")

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def check_state(self, *args, **kwargs):
        """Will never fail the check_state check"""
        # pylint: disable=unused-argument
        return []
