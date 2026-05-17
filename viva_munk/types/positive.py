"""
Positive numeric types for fields and concentrations.

Adapted from spatio_flux/types/positive.py. The minimal subset needed to
support diffusing concentration fields and cell ↔ field exchange.

Classes are prefixed with ``VM`` to avoid collision with spatio_flux's
identically-named classes in the bigraph-schema type resolver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bigraph_schema.schema import Array, Float
from bigraph_schema.methods import apply, render


# ---------------------------------------------------------------------
# Type definitions (Node subclasses)
# ---------------------------------------------------------------------

@dataclass(kw_only=True)
class VMSetFloat(Float):
    """A float that is replaced by its update (no accumulation)."""


@dataclass(kw_only=True)
class VMPositiveFloat(Float):
    """A float that accumulates updates and is clamped to be non-negative."""


@dataclass(kw_only=True)
class VMConcentration(VMPositiveFloat):
    """Non-negative accumulator for an environmental concentration."""


@dataclass(kw_only=True)
class VMPositiveArray(Array):
    """An array whose updates are accumulated and clamped elementwise to be
    non-negative. Used for diffusing concentration fields."""


# Backwards-compat aliases — keep the old short names exposed at module scope
# so any code that does ``from viva_munk.types.positive import Concentration``
# still works. The CLASS object is the same; the disambiguation comes from the
# class __name__ which is now prefixed.
SetFloat = VMSetFloat
PositiveFloat = VMPositiveFloat
Concentration = VMConcentration
PositiveArray = VMPositiveArray


# ---------------------------------------------------------------------
# Render: dataclass schema -> registry name
# ---------------------------------------------------------------------

@render.dispatch
def render(schema: VMPositiveFloat, defaults: bool = False):
    return "viva_munk_positive_float"


@render.dispatch
def render(schema: VMConcentration, defaults: bool = False):
    return "viva_munk_concentration"


@render.dispatch
def render(schema: VMPositiveArray, defaults: bool = False):
    return "viva_munk_positive_array"


@render.dispatch
def render(schema: VMSetFloat, defaults: bool = False):
    return "viva_munk_set_float"


# ---------------------------------------------------------------------
# Apply: state update semantics
# ---------------------------------------------------------------------

@apply.dispatch
def apply(schema: VMSetFloat, state, update, path):
    return update, []


@apply.dispatch
def apply(schema: VMPositiveFloat, state, update, path):
    if update is None:
        return state, []
    if state is None:
        state = 0.0
    return max(0.0, state + update), []


@apply.dispatch
def apply(schema: VMPositiveArray, current, update, path):
    """
    Update modes:
      - dense numpy array: add elementwise, clamp at 0
      - sparse nested dict {i: {j: delta}}: add at indices, clamp at 0
      - scalar fallback: PositiveFloat semantics
    """
    if update is None:
        return current, []

    if not isinstance(current, np.ndarray):
        if isinstance(update, dict):
            raise ValueError("Cannot apply dict update to scalar current value.")
        return max(0.0, (current or 0.0) + update), []

    if isinstance(update, np.ndarray):
        return np.maximum(0.0, current + update), []

    # sparse update
    result = np.array(current, copy=True)

    def _apply_sparse(delta, idx=()):
        if isinstance(delta, dict):
            for k, v in delta.items():
                _apply_sparse(v, idx + (k,))
            return
        result[idx] = max(0.0, float(result[idx]) + float(delta))

    _apply_sparse(update)
    return result, []


# ---------------------------------------------------------------------
# Registry name -> schema instance mapping
# ---------------------------------------------------------------------

positive_types = {
    "viva_munk_positive_float": VMPositiveFloat(),
    "viva_munk_positive_array": VMPositiveArray(),
    "viva_munk_concentration": VMConcentration(),
    "viva_munk_set_float": VMSetFloat(),
}
