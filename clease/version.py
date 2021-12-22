from pathlib import Path

with Path(__file__).with_name('_version.txt').open('r') as f:
    __version__ = f.readline().strip()

# Representation of version_info as (x, y, z)
version_info = __version__.split('.')

__all__ = ('__version__', 'version_info')
