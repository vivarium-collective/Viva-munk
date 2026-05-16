"""Experiment document builders.

Each module here exports a single ``*_document(config=None) -> dict`` function
that returns a process_bigraph composite document. The functions are
re-exported here for convenient registry access.
"""
from viva_munk.experiments.documents.daughter_machine import daughter_machine_document
from viva_munk.experiments.documents.attachment import attachment_document
from viva_munk.experiments.documents.glucose_growth import glucose_growth_document
from viva_munk.experiments.documents.bending_pressure import bending_pressure_document
from viva_munk.experiments.documents.mother_machine import mother_machine_document
from viva_munk.experiments.documents.biofilm import biofilm_document
from viva_munk.experiments.documents.chemotaxis import chemotaxis_document
from viva_munk.experiments.documents.inclusion_bodies import inclusion_bodies_document
from viva_munk.experiments.documents.quorum_sensing import quorum_sensing_document

__all__ = [
    'daughter_machine_document',
    'attachment_document',
    'glucose_growth_document',
    'bending_pressure_document',
    'mother_machine_document',
    'biofilm_document',
    'chemotaxis_document',
    'inclusion_bodies_document',
    'quorum_sensing_document',
]
