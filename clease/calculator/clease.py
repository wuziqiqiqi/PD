"""Calculator for Cluster Expansion."""
import sys
import contextlib
from typing import Dict, Optional, TextIO, Union, List, Sequence, Any
import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from clease_cxx import PyCEUpdater, has_parallel
from clease.datastructures import SystemChange, SystemChanges
from clease.corr_func import CorrFunction
from clease.settings import ClusterExpansionSettings


class MovedIgnoredAtomError(Exception):
    """Raised when ignored atoms is moved."""


class UnitializedCEError(Exception):
    """The C++ CE Updater has not yet been initialized"""


class KeepChanges:
    def __init__(self):
        self.keep_changes = True


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class Clease:
    """Class for calculating energy using CLEASE.

    :param settings: `ClusterExpansionSettings` object

    :param eci: Dictionary containing cluster names and their ECI values

    :param init_cf: (Optional) One can supply the correlation function values
        of the Atoms object to skip the initial assessment step. The dictionary
        should contain cluster names (same as the ones provided in `eci`) and
        their correlation function values.

    :logfile: One can pass the file object or string of the file name to get a
        log file. Do not specify or set it to *None* to avoid generating a log
        file. Use '-' for stdout.
    """

    name = "CLEASE"
    implemented_properties = ["energy"]

    def __init__(
        self,
        settings: ClusterExpansionSettings,
        eci: Dict[str, float],
        init_cf: Optional[Dict[str, float]] = None,
        logfile: Union[TextIO, str, None] = None,
    ) -> None:

        if not isinstance(settings, ClusterExpansionSettings):
            msg = "settings must be CEBulk or CECrystal object."
            raise TypeError(msg)
        # C++ updater initialized when atoms are set
        self.updater = None
        self.results = {}
        self.settings = settings
        self.corrFunc = CorrFunction(settings)
        self.eci = eci
        # store cluster names
        self.cf_names = list(eci.keys())

        # calculate init_cf or convert init_cf to array
        if init_cf is None or isinstance(init_cf, dict):
            self.init_cf = init_cf
        else:
            msg = "'init_cf' must be a dictionary containing cluster names "
            msg += "and their correlation function values."
            raise TypeError(msg)

        if self.init_cf is not None and len(self.eci) != len(self.init_cf):
            msg = "length of provided ECIs and correlation functions do not "
            msg += "match"
            raise ValueError(msg)

        # logfile
        if isinstance(logfile, str):
            if logfile == "-":
                logfile = sys.stdout
            else:
                # pylint: disable=consider-using-with
                logfile = open(logfile, "a")
        self.logfile = logfile

        # reference atoms for calculating the cf and energy for new atoms
        self.atoms = None
        self.symmetry_group = None
        self.is_backround_index = None

    def set_atoms(self, atoms: Atoms) -> None:
        self.atoms = atoms

        if self.init_cf is None:
            self.init_cf = self.corrFunc.get_cf_by_names(self.atoms, self.cf_names)

        if len(self.settings.atoms) != len(atoms):
            msg = "Passed Atoms object and settings.atoms should have same "
            msg += "number of atoms."
            raise ValueError(msg)

        if not np.allclose(atoms.positions, self.settings.atoms.positions):
            msg = "Positions of all atoms in the passed Atoms object and "
            msg += "settings.atoms should be the same. "
            raise ValueError(msg)

        self.symmetry_group = np.zeros(len(atoms), dtype=int)
        for symm, indices in enumerate(self.settings.index_by_sublattice):
            self.symmetry_group[indices] = symm
        self.is_backround_index = np.zeros(len(atoms), dtype=np.uint8)
        self.is_backround_index[self.settings.background_indices] = 1

        self.updater = PyCEUpdater(
            self.atoms,
            self.settings,
            self.init_cf,
            self.eci,
            self.settings.cluster_list,
        )

    def get_energy_given_change(self, system_changes: SystemChanges, keep_changes=False) -> float:
        """
        Calculate the energy when the change is known. No checking will be
        performed.

        :param system_changes: List of system changes. For example, if the
            occupation of the atomic index 23 is changed from Mg to Al,
            system_change = [(23, Mg, Al)]. If an Mg atom occupying the atomic
            index 26 is swapped with an Al atom occupying the atomic index 12,
            system_change = [(26, Mg, Al), (12, Al, Mg)]
        :param keep_changes: Should the calculator revert to the state prior to
            the system changes, or should the calculator stay in the new state
            with the given system changes. Default: False
        """
        with self.with_system_changes(system_changes) as keeper:
            keeper.keep_changes = keep_changes
            return self.energy

    def reset(self) -> None:
        self.results = {}

    def calculate_cf_from_scratch(self) -> Dict[str, float]:
        """Calculate correlation functions from scratch."""
        self.require_updater()
        return self.updater.calculate_cf_from_scratch(self.atoms, self.cf_names)

    def calculate_energy_from_scratch(self) -> float:
        """Calculate the correlation functions, and hence also the energy,
        from scratch.
        """
        self.calculate_cf_from_scratch()
        self.energy = self.updater.get_energy()
        return self.energy

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: SystemChanges = None,
    ) -> float:
        """Calculate the energy of the passed Atoms object.

        If accept=True, the most recently used atoms object is used as a
        reference structure to calculate the energy of the passed atoms.
        Returns energy.

        :param atoms: ASE Atoms object

        :param properties: List of what needs to be calculated.
            It can only be ['energy'] for the basic Clease calculator.

        :param system_changes: List of system changes. For example, if the
            occupation of the atomic index 23 is changed from Mg to Al,
            system_change = [(23, Mg, Al)]. If an Mg atom occupying the atomic
            index 26 is swapped with an Al atom occupying the atomic index 12,
            system_change = [(26, Mg, Al), (12, Al, Mg)].
            Currently not supported.
        """
        if atoms is not None:
            if self.atoms is None:
                # We haven't yet initialized, so initialize with the passed atoms object.
                self.set_atoms(atoms)
            else:
                # Use the symbols of the passed atoms object
                self.atoms.symbols[:] = atoms.symbols
        self.require_updater()

        _check_properties(properties, self.implemented_properties)

        if system_changes is not None:
            raise ValueError("system_changes in calculate is currently not supported.")

        self.update_energy()
        return self.energy

    def get_property(self, name: str, atoms: Atoms = None, allow_calculation: bool = True):
        """Get a property from the calculator.

        Exists due to compatibility with ASE, should not be used directly.
        """
        _check_properties([name], self.implemented_properties)

        if not allow_calculation:
            return self.energy
        return self.get_potential_energy(atoms=atoms)

    def get_potential_energy(self, atoms: Atoms = None) -> float:
        """Calculate the energy from scratch with an atoms object"""
        # self.set_atoms(atoms)
        return self.calculate(atoms=atoms)

    def clear_history(self) -> None:
        """Direct access to the clear history method"""
        self.updater.clear_history()

    def update_energy(self) -> None:
        """Update correlation function and get new energy."""
        self.update_cf()
        self.energy = self.updater.get_energy()

    def calculation_required(self, atoms: Atoms, properties: Sequence[str] = None) -> bool:
        """Check whether a calculation is required for a given atoms object.
        The ``properties`` argument only exists for compatibility reasons, and has no effect.
        Primarily for ASE compatibility.
        """
        _check_properties(properties, self.implemented_properties)
        if self.updater is None:
            return True
        if "energy" not in self.results:
            return True
        changed_indices = self.get_changed_sites(atoms)
        return bool(changed_indices)

    def check_state(self, atoms: Atoms) -> List[str]:
        """Method for checking if energy needs calculation.
        Primarily for ASE compatibility.
        """
        res = []
        if self.calculation_required(atoms):
            res.append("energy")
        return res

    @property
    def energy(self):
        return self.results.get("energy", None)

    @energy.setter
    def energy(self, value):
        self.results["energy"] = value

    @property
    def indices_of_changed_atoms(self) -> List[int]:
        """Return the indices of atoms that have been changed."""
        changed = self.get_changed_sites(self.atoms)
        for index in changed:
            if self.is_backround_index[index] and not self.settings.include_background_atoms:
                msg = f"Atom with index {index} is a background atom."
                raise MovedIgnoredAtomError(msg)

        return changed

    def get_changed_sites(self, atoms: Atoms) -> List[int]:
        """Return the list of indices which differ from the internal ones."""
        self.require_updater()
        return self.updater.get_changed_sites(atoms)

    def get_cf(self) -> Dict[str, float]:
        """Return the correlation functions as a dict"""
        return self.updater.get_cf()

    def update_cf(self, system_changes: SystemChanges = None) -> None:
        """Update correlation function based on the reference value.

        :param system_changes: List of system changes. For example, if the
            occupation of the atomic index 23 is changed from Mg to Al,
            system_change = [(23, Mg, Al)]. If an Mg atom occupying the atomic
            index 26 is swapped with an Al atom occupying the atomic index 12,
            system_change = [(26, Mg, Al), (12, Al, Mg)]
            If system_changes is None, the correlation functions are recalculated.
            Default: None.
        """
        if system_changes is None:
            swapped_indices = self.indices_of_changed_atoms
            symbols = self.updater.get_symbols()
            system_changes = [
                SystemChange(
                    index=idx,
                    old_symb=symbols[idx],
                    new_symb=self.atoms[idx].symbol,
                    name="internal_symbol_change",
                )
                for idx in swapped_indices
            ]
        for change in system_changes:
            self.updater.update_cf(change)

    @property
    def cf(self) -> List[float]:
        temp_cf = self.updater.get_cf()
        return [temp_cf[x] for x in self.cf_names]

    def log(self) -> None:
        """Write energy to log file."""
        if self.logfile is None:
            return
        self.logfile.write(f"{self.energy}\n")
        self.logfile.flush()

    def update_eci(self, eci: Dict[str, float]) -> None:
        """Update the ECI values.

        :param eci: dictionary with new ECI values
        """
        self.eci = eci
        self.updater.set_eci(eci)
        self._on_eci_changed()

    def get_singlets(self) -> np.ndarray:
        return self.updater.get_singlets()

    def get_energy(self) -> float:
        self.energy = self.updater.get_energy()
        return self.energy

    def _on_eci_changed(self):
        """
        Callback that is called after ECIs are changed. It is provided such
        that calculators inheriting from this class (and implementing this
        method) can be notified when the ECIs are changed.
        """

    @contextlib.contextmanager
    def with_system_changes(self, system_changes: SystemChanges):
        """
        : param system_changes: List of system changes. For example, if the
            occupation of the atomic index 23 is changed from Mg to Al,
            system_change = [(23, Mg, Al)]. If an Mg atom occupying the atomic
            index 26 is swapped with an Al atom occupying the atomic index 12,
            system_change = [(26, Mg, Al), (12, Al, Mg)]
        """
        keeper = KeepChanges()  # We need an object we can mutate to flag for cleanup
        try:
            # Apply the updates
            self.apply_system_changes(system_changes)
            yield keeper
        finally:
            if keeper.keep_changes:
                # Keep changes
                self.keep_system_changes()
            else:
                # Revert changes
                self.undo_system_changes()

    def apply_system_changes(self, system_changes: SystemChanges) -> None:
        """Apply a set of changes to the calculator, and evaluate the energy"""
        # Apply the updates
        self.update_cf(system_changes)
        self.energy = self.updater.get_energy()

    def undo_system_changes(self) -> None:
        """Revert a set of changes. The changes passed in should be the original
        sequence of system changes which were applied. Restores the original results.

        Restore the Atoms object to its original configuration and energy.

        This method reverts the Atoms object to its oldest state stored in
        memory. The state is restored to either
        (1) an initial state when the calculator was attached, or
        (2) the state at which the `clear_history()` method was invoked last
            time.

        NOTE: The maximum capacity for the history buffer is 1000 steps
        """
        self.updater.undo_changes()
        self.energy = self.updater.get_energy()

    def keep_system_changes(self) -> None:
        """A set of system changes are to be kept. Perform necessary actions to prepare
        for a new evaluation."""
        # Call clear_history directly intentionally, rather than using self.clear_history()
        self.updater.clear_history()

    def get_num_threads(self) -> int:
        """Get the number of threads from the C++ updater."""
        self.require_updater()
        return self.updater.get_num_threads()

    def set_num_threads(self, value: int) -> None:
        """Number of threads to use when updating CFs. Requires CLEASE to
        be compiled with OpenMP if the value is different from 1.

        Args:
            value (int): Number of threads.
        """
        if not isinstance(value, int):
            raise TypeError(f"Number of threads must be int, got {value!r}")
        if value < 1:
            raise ValueError("Number of threads must be at least 1")
        self.require_updater()

        if value != 1 and not has_parallel():
            # Having a value of 1 is OK, since that's not threading.
            raise ValueError("CLEASE not compiled with OpenMP. Cannot add more threads.")
        self.updater.set_num_threads(value)

    def require_updater(self) -> None:
        if self.updater is None:
            raise UnitializedCEError("Updater hasn't been initialized yet.")

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return a dictionary with relevant parameters."""
        return {"eci": self.eci}

    def todict(self) -> dict:
        """Return the parameters, i.e. the ECI values.
        For ASE compatibility.
        """
        return self.parameters


def _check_properties(properties: Optional[List[str]], implemented_properties: List[str]) -> None:
    """Check whether the passed properties is supported. If it is None, nothing is checked.
    Raises PropertyNotImplementedError upon finding a bad property.
    """
    if properties is None:
        return
    for prop in properties:
        if prop not in implemented_properties:
            allowed = ", ".join(implemented_properties)
            raise PropertyNotImplementedError(
                f"Property '{prop}' is unsupported. Implemented properties: {allowed}"
            )
