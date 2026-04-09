"""
Custom PymunkAgent type with optimized dispatch.

The default dict schema for pymunk_agent has 10 fields, each requiring
its own apply/reconcile/realize dispatch via plum. With 250 cells x 480 ticks,
that's millions of method dispatches per simulation. This module defines a
single Node subclass that handles all 10 fields in one Python pass per
operation, eliminating per-field dispatch overhead.

Field semantics:
  - mass, radius, length, angle, inertia: Float (accumulate / delta)
  - location, velocity: Tuple[Float, Float] (accumulate / delta)
  - elasticity, friction: Float (accumulate, but rarely changes)
  - type: Enum[circle, segment] (set semantics)
"""
from dataclasses import dataclass

from plum import dispatch
from bigraph_schema.schema import Node


# Field categories for fast iteration
_FLOAT_FIELDS = (
    'mass', 'radius', 'length', 'angle', 'inertia',
    'elasticity', 'friction',
)
_TUPLE_FIELDS = ('location', 'velocity')
_STRING_FIELDS = ('type',)
_ALL_FIELDS = _FLOAT_FIELDS + _TUPLE_FIELDS + _STRING_FIELDS

_FLOAT_DEFAULTS = {
    'mass': 0.0,
    'radius': 0.0,
    'length': 0.0,
    'angle': 0.0,
    'inertia': 0.0,
    'elasticity': 0.0,
    'friction': 0.8,
}


@dataclass(kw_only=True)
class PymunkAgent(Node):
    """Opaque agent type with optimized field-level dispatch."""
    pass


def register_pymunk_agent_dispatches():
    """Register dispatched methods for PymunkAgent."""
    from bigraph_schema.methods.apply import apply
    from bigraph_schema.methods.reconcile import reconcile
    from bigraph_schema.methods.default import default
    from bigraph_schema.methods.realize import realize
    from bigraph_schema.methods.check import check

    @apply.dispatch
    def apply_pymunk_agent(schema: PymunkAgent, state, update, path):
        """Single-pass apply for all 10 fields. Floats accumulate, type sets, tuples accumulate elementwise."""
        if update is None:
            return state, []
        if state is None:
            # Initial value: copy update directly
            return dict(update), []

        result = state.copy()

        # Float fields: accumulate
        for k in _FLOAT_FIELDS:
            if k in update:
                u = update[k]
                if u is not None:
                    s = result.get(k)
                    result[k] = (s if s is not None else 0.0) + u

        # Tuple fields (location, velocity): elementwise accumulate
        for k in _TUPLE_FIELDS:
            if k in update:
                u = update[k]
                if u is not None:
                    s = result.get(k, (0.0, 0.0))
                    result[k] = (s[0] + u[0], s[1] + u[1])

        # String/enum fields (type): set semantics
        if 'type' in update and update['type'] is not None:
            result['type'] = update['type']

        return result, []

    @reconcile.dispatch
    def reconcile_pymunk_agent(schema: PymunkAgent, updates: list):
        """Combine multiple updates by summing floats/tuples and last-wins for type."""
        if not updates:
            return None
        # Fast path: single update
        non_none = [u for u in updates if u is not None]
        if not non_none:
            return None
        if len(non_none) == 1:
            return non_none[0]

        result = {}
        # Sum float fields across updates
        for k in _FLOAT_FIELDS:
            total = None
            for u in non_none:
                v = u.get(k) if isinstance(u, dict) else None
                if v is not None:
                    total = v if total is None else total + v
            if total is not None:
                result[k] = total

        # Sum tuple fields elementwise
        for k in _TUPLE_FIELDS:
            tx = ty = None
            for u in non_none:
                v = u.get(k) if isinstance(u, dict) else None
                if v is not None:
                    tx = v[0] if tx is None else tx + v[0]
                    ty = v[1] if ty is None else ty + v[1]
            if tx is not None:
                result[k] = (tx, ty)

        # Type: last non-None wins
        for u in non_none:
            v = u.get('type') if isinstance(u, dict) else None
            if v is not None:
                result['type'] = v

        return result if result else None

    @default.dispatch
    def default_pymunk_agent(schema: PymunkAgent):
        result = dict(_FLOAT_DEFAULTS)
        result['type'] = None
        result['location'] = (0.0, 0.0)
        result['velocity'] = (0.0, 0.0)
        return result

    @realize.dispatch
    def realize_pymunk_agent(core, schema: PymunkAgent, state, path=()):
        """Fill missing fields with defaults; pass through provided values."""
        if state is None:
            return schema, default_pymunk_agent(schema), []
        if not isinstance(state, dict):
            return schema, state, []
        # Merge defaults under provided state
        result = default_pymunk_agent(schema)
        result.update(state)
        return schema, result, []

    @check.dispatch
    def check_pymunk_agent(schema: PymunkAgent, state):
        return isinstance(state, dict)
