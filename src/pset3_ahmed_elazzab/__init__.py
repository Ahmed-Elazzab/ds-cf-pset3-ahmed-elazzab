"""Main package for pset3_Ahmed_Elazzab.

Code in this module takes care of your package versioning.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Read version from PKG metadata
    __version__ = version("pg-ds-cf-pset3-ahmed-elazzab")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fall-back version
