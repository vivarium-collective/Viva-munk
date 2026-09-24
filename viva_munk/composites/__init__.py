"""Composite-generator registrations for viva-munk's experiment documents.

Each function here is a thin `@composite_generator`-decorated wrapper around
the corresponding `*_document(config=None)` builder in
`viva_munk.experiments.documents`. The decorator side-effect (registering
into pbg-superpowers' `_REGISTRY`) is what makes these visible to the
vivarium-dashboard's composite browser.

Every viva-munk composite is built on the `pymunk_agent` type and viva-munk's
own processes (PymunkProcess, GrowDivide, …). Those live in viva-munk's core,
not in the workspace's `build_core`, so the dashboard needs them registered on
whatever core *realizes* the composite. We declare that via each generator's
``core_extensions`` (see ``process_bigraph.composite_generator.apply_core_extensions``):
without it, a run builds the composite in the workspace core, `pymunk_agent` is
unregistered, and Composite realization fails with
``cannot resolve type "map[pymunk_agent]"``.

Parameters intentionally left unexposed for now — call sites get the
builder's default config. Per-composite kwargs can be surfaced later by
filling in `parameters=` on the decorator.
"""
from viva_superpowers.composite_generator import composite_generator

from viva_munk.experiments.documents.attachment import attachment_document
from viva_munk.experiments.documents.bending_pressure import bending_pressure_document
from viva_munk.experiments.documents.biofilm import biofilm_document
from viva_munk.experiments.documents.chemotaxis import chemotaxis_document
from viva_munk.experiments.documents.daughter_machine import daughter_machine_document
from viva_munk.experiments.documents.glucose_growth import glucose_growth_document
from viva_munk.experiments.documents.inclusion_bodies import inclusion_bodies_document
from viva_munk.experiments.documents.mother_machine import mother_machine_document
from viva_munk.experiments.documents.quorum_sensing import quorum_sensing_document


# ``register_pymunk_types`` / ``register_processes`` live in ``viva_munk/__init__``,
# which imports THIS module (to fire the generator side-effects) *before* it
# defines them — so importing them at module top-level would be a circular
# import. Wrap them in lazy indirections that import at call time, when
# ``apply_core_extensions`` runs the extensions against the realize core.
def _register_pymunk_types(core):
    from viva_munk import register_pymunk_types
    return register_pymunk_types(core)


def _register_processes(core):
    from viva_munk import register_processes
    return register_processes(core)


# The types + processes every viva-munk composite depends on. Applied to the
# core that realizes the composite (dashboard run path + drill-in).
_CORE_EXTENSIONS = [_register_pymunk_types, _register_processes]


@composite_generator(
    name="attachment",
    description="Cells settling/attaching on a surface.",
    default_n_steps=200,
    core_extensions=_CORE_EXTENSIONS,
)
def attachment(core=None) -> dict:
    return attachment_document()


@composite_generator(
    name="bending_pressure",
    description="Bending-pressure rod-like cells.",
    default_n_steps=200,
    core_extensions=_CORE_EXTENSIONS,
)
def bending_pressure(core=None) -> dict:
    return bending_pressure_document()


@composite_generator(
    name="biofilm",
    description="Growing biofilm with EPS secretion + pressure dynamics.",
    default_n_steps=500,
    core_extensions=_CORE_EXTENSIONS,
)
def biofilm(core=None) -> dict:
    return biofilm_document()


@composite_generator(
    name="chemotaxis",
    description="Run/tumble chemotaxis up a static 2D ligand gradient.",
    default_n_steps=200,
    core_extensions=_CORE_EXTENSIONS,
)
def chemotaxis(core=None) -> dict:
    return chemotaxis_document()


@composite_generator(
    name="daughter_machine",
    description="Daughter-machine geometry: tracking daughter cells.",
    default_n_steps=300,
    core_extensions=_CORE_EXTENSIONS,
)
def daughter_machine(core=None) -> dict:
    return daughter_machine_document()


@composite_generator(
    name="glucose_growth",
    description="Glucose-driven growth and division.",
    default_n_steps=300,
    core_extensions=_CORE_EXTENSIONS,
)
def glucose_growth(core=None) -> dict:
    return glucose_growth_document()


@composite_generator(
    name="inclusion_bodies",
    description="Inclusion-body accumulation across a growing colony.",
    default_n_steps=500,
    core_extensions=_CORE_EXTENSIONS,
)
def inclusion_bodies(core=None) -> dict:
    return inclusion_bodies_document()


@composite_generator(
    name="mother_machine",
    description="Mother-machine geometry: single-channel cell trapping.",
    default_n_steps=300,
    core_extensions=_CORE_EXTENSIONS,
)
def mother_machine(core=None) -> dict:
    return mother_machine_document()


@composite_generator(
    name="quorum_sensing",
    description="Quorum-sensing autoinducer feedback in a growing colony.",
    default_n_steps=300,
    core_extensions=_CORE_EXTENSIONS,
)
def quorum_sensing(core=None) -> dict:
    return quorum_sensing_document()
