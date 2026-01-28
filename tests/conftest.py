import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

# Ensure the project root is on sys.path so `sakura` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock heavy dependencies that are unavailable in the test environment.
# These must be injected before any sakura module is imported.

# mpi4py
_mpi = MagicMock()
sys.modules.setdefault("mpi4py", _mpi)
sys.modules.setdefault("mpi4py.MPI", _mpi.MPI)

# bson
sys.modules.setdefault("bson", MagicMock())

# gnutools — provide a working RecNamespace so sakura.functional is testable
class _RecNamespace(SimpleNamespace):
    """Minimal stand-in for gnutools.utils.RecNamespace."""
    def __init__(self, d=None, **kwargs):
        if d is not None:
            kwargs.update(d)
        super().__init__(**kwargs)

_gnutools = MagicMock()
_gnutools_utils = MagicMock()
_gnutools_utils.RecNamespace = _RecNamespace
_gnutools_utils.RecDict = dict
sys.modules.setdefault("gnutools", _gnutools)
sys.modules.setdefault("gnutools.utils", _gnutools_utils)
sys.modules.setdefault("gnutools.fs", MagicMock())
