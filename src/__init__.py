"""Main package for pset3_depap.

Code in this module takes care of your package versioning - including a switch required
after introduction of importlib.metadata in Python3.8+ and importlib_metadata in older versions.
"""
try:
    from importlib.metadata import PackageNotFoundError, version
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import PackageNotFoundError, version

try:
    # Read version from PKG metadata
    __version__ = version("pg-ds-cf-pset3-depap")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fall-back version
