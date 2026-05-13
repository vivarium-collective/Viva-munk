"""Composite-generator registrations for viva-munk's experiment documents.

Each function here is a thin `@composite_generator`-decorated wrapper around
the corresponding `*_document(config=None)` builder in
`multi_cell.experiments.documents`. The decorator side-effect (registering
into pbg-superpowers' `_REGISTRY`) is what makes these visible to the
vivarium-dashboard's composite browser.

Parameters intentionally left unexposed for now — call sites get the
builder's default config. Per-composite kwargs can be surfaced later by
filling in `parameters=` on the decorator.
"""
from pbg_superpowers.composite_generator import composite_generator

from multi_cell.experiments.documents.attachment import attachment_document
from multi_cell.experiments.documents.bending_pressure import bending_pressure_document
from multi_cell.experiments.documents.biofilm import biofilm_document
from multi_cell.experiments.documents.chemotaxis import chemotaxis_document
from multi_cell.experiments.documents.daughter_machine import daughter_machine_document
from multi_cell.experiments.documents.glucose_growth import glucose_growth_document
from multi_cell.experiments.documents.inclusion_bodies import inclusion_bodies_document
from multi_cell.experiments.documents.mother_machine import mother_machine_document
from multi_cell.experiments.documents.quorum_sensing import quorum_sensing_document


@composite_generator(name="attachment", description="Cells settling/attaching on a surface.")
def attachment(core=None) -> dict:
    return attachment_document()


@composite_generator(name="bending_pressure", description="Bending-pressure rod-like cells.")
def bending_pressure(core=None) -> dict:
    return bending_pressure_document()


@composite_generator(name="biofilm", description="Growing biofilm with EPS secretion + pressure dynamics.")
def biofilm(core=None) -> dict:
    return biofilm_document()


@composite_generator(name="chemotaxis", description="Run/tumble chemotaxis up a static 2D ligand gradient.")
def chemotaxis(core=None) -> dict:
    return chemotaxis_document()


@composite_generator(name="daughter_machine", description="Daughter-machine geometry: tracking daughter cells.")
def daughter_machine(core=None) -> dict:
    return daughter_machine_document()


@composite_generator(name="glucose_growth", description="Glucose-driven growth and division.")
def glucose_growth(core=None) -> dict:
    return glucose_growth_document()


@composite_generator(name="inclusion_bodies", description="Inclusion-body accumulation across a growing colony.")
def inclusion_bodies(core=None) -> dict:
    return inclusion_bodies_document()


@composite_generator(name="mother_machine", description="Mother-machine geometry: single-channel cell trapping.")
def mother_machine(core=None) -> dict:
    return mother_machine_document()


@composite_generator(name="quorum_sensing", description="Quorum-sensing autoinducer feedback in a growing colony.")
def quorum_sensing(core=None) -> dict:
    return quorum_sensing_document()
