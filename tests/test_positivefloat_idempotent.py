"""viva-munk's positive types coexist with a sibling package's (e.g. spatio-flux).

viva-munk's positive numeric types are a fork of spatio_flux's positive.py, so a
core that has both packages may already carry 'positive_float' et al. when
``register_pymunk_types`` runs. Re-registering used to deep-merge through
``resolve()`` and raise when the existing entry was an incompatible (Python
class-valued) type — which silently aborted the rest of ``core_import`` (so the
workspace's custom types and process links never registered). Registration is
now idempotent: it defers to whatever a sibling already registered.
"""

from bigraph_schema.core import BASE_TYPES, Core

import viva_munk
from viva_munk.types.positive import VMPositiveFloat


def _fresh() -> Core:
    return Core(BASE_TYPES)


class _ForeignPositiveFloat:
    """Stand-in for a sibling package's class-valued positive_float
    (e.g. spatio_flux.types.positive.PositiveFloat)."""


def test_solo_registration_still_works():
    core = _fresh()
    viva_munk.register_pymunk_types(core)
    assert "positive_float" in core.registry
    assert "pymunk_agent" in core.registry


def test_reregistering_over_a_class_valued_entry_is_the_bug():
    # A sibling registered 'positive_float' as a bare class; a raw re-register
    # goes through update_type -> resolve and cannot merge a class with a schema.
    core = _fresh()
    core.registry["positive_float"] = _ForeignPositiveFloat
    try:
        core.register_type("positive_float", VMPositiveFloat())
    except Exception:
        return  # expected: this is exactly the crash the fix routes around
    # If a future bigraph-schema merges this cleanly the fix is still harmless.


def test_register_pymunk_types_is_idempotent_when_a_sibling_registered_first():
    core = _fresh()
    core.registry["positive_float"] = _ForeignPositiveFloat  # sibling got there first
    # Must NOT raise, and must not clobber the sibling's entry.
    viva_munk.register_pymunk_types(core)
    assert core.registry["positive_float"] is _ForeignPositiveFloat
    # ...while still registering everything viva-munk owns.
    for type_name in ("pymunk_agent", "positive_array", "concentration", "set_float"):
        assert type_name in core.registry, f"missing {type_name}"
