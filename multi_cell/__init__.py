"""Backwards-compatibility shim for the multi_cell → viva_munk rename.

The Python package was renamed from ``multi_cell`` to ``viva_munk`` (so
the import name matches the PyPI/repo name viva-munk). This shim keeps
``from multi_cell.X import Y`` working for downstream consumers while
they migrate, with a DeprecationWarning on first use.

Remove this module in a future release once all known consumers
(v2ecoli, pbg-template-derived workspaces) have moved to ``viva_munk``.

Mechanism: install ``viva_munk`` and every already-imported submodule
into ``sys.modules`` under the legacy ``multi_cell.*`` names, then
re-export the public surface here so ``from multi_cell import X``
resolves the same name as ``from viva_munk import X``.
"""
from __future__ import annotations

import sys
import warnings as _warnings

import viva_munk as _viva_munk

_warnings.warn(
    "The 'multi_cell' import root was renamed to 'viva_munk'. "
    "Update `from multi_cell.X import Y` to `from viva_munk.X import Y`. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias the top-level module so `import multi_cell` and identity checks
# like `sys.modules['multi_cell']` resolve to viva_munk's loaded module.
sys.modules[__name__] = _viva_munk

# Mirror every already-loaded viva_munk.* submodule under multi_cell.*
# so existing `from multi_cell.processes.multibody import PymunkProcess`
# imports resolve to the same module objects (no double-import, no
# duplicate registry entries).
for _full, _mod in list(sys.modules.items()):
    if _full == "viva_munk" or _full.startswith("viva_munk."):
        _alias = "multi_cell" + _full[len("viva_munk"):]
        sys.modules.setdefault(_alias, _mod)

# Re-export viva_munk's public attributes so attribute access on the
# shim module (when someone has held a reference to it from before the
# sys.modules swap) also works.
for _name in dir(_viva_munk):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_viva_munk, _name)

del _full, _mod, _alias, _name, _viva_munk, _warnings, sys
