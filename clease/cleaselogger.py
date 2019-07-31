from enum import IntEnum
from ase.utils import convert_string_to_fd


class LogVerbosity(IntEnum):
    # IntEnum for comparisons
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3


class CLEASELogger:
    """Class for handling printing in CLEASE"""

    def __init__(self, fd='-', verbosity=LogVerbosity.INFO):
        self.oldfd = None           # Dummy
        self.fd = fd                # File descriptor
        self.verbosity = verbosity  # Print messages with verbose <= verbosity

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = self.format_verbose(verbosity)

    @property
    def fd(self):
        return self._fd

    @fd.setter
    def fd(self, fd):
        if fd == self.oldfd:
            return
        self.oldfd = fd
        self._fd = convert_string_to_fd(fd)

    def format_verbose(self, verbose):
        """Ensure that verbose is LogVerbosity type"""
        if isinstance(verbose, LogVerbosity):
            return verbose
        elif isinstance(verbose, int):
            return LogVerbosity(verbose)
        else:
            raise ValueError('Bad verbosity level: {}'.format(verbose))

    def __call__(self, *args, verbose=LogVerbosity.INFO, **kwargs):
        verbose = self.format_verbose(verbose)
        if verbose <= self.verbosity:
            print(*args, file=self._fd, **kwargs)

    def __str__(self):
        return 'Verbosity threshold: {}, FD: {}'.format(str(self.verbosity),
                                                        self.fd)
