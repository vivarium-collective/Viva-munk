"""
Positive numeric types for fields and concentrations.

Adapted from spatio_flux/types/positive.py. The minimal subset needed to
support diffusing concentration fields and cell ↔ field exchange.
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
class SetFloat(Float):
    """A float that is replaced by its update (no accumulation)."""


@dataclass(kw_only=True)
class PositiveFloat(Float):
    """A float that accumulates updates and is clamped to be non-negative."""


@dataclass(kw_only=True)
class Concentration(PositiveFloat):
    """Non-negative accumulator for an environmental concentration."""


@dataclass(kw_only=True)
class PositiveArray(Array):
    """An array whose updates are accumulated and clamped elementwise to be
    non-negative. Used for diffusing concentration fields."""


# ---------------------------------------------------------------------
# Render: dataclass schema -> registry name
# ---------------------------------------------------------------------

@render.dispatch
def render(schema: PositiveFloat, defaults: bool = False):
    return "positive_float"


@render.dispatch
def render(schema: Concentration, defaults: bool = False):
    return "concentration"


@render.dispatch
def render(schema: PositiveArray, defaults: bool = False):
    return "positive_array"


@render.dispatch
def render(schema: SetFloat, defaults: bool = False):
    return "set_float"


# ---------------------------------------------------------------------
# Apply: state update semantics
# ---------------------------------------------------------------------

@apply.dispatch
def apply(schema: SetFloat, state, update, path):
    return update, []


@apply.dispatch
def apply(schema: PositiveFloat, state, update, path):
    if update is None:
        return state, []
    if state is None:
        state = 0.0
    return max(0.0, state + update), []


@apply.dispatch
def apply(schema: PositiveArray, current, update, path):
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
    "positive_float": PositiveFloat(),
    "positive_array": PositiveArray(),
    "concentration": Concentration(),
    "set_float": SetFloat(),
}
