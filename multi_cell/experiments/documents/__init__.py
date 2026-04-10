"""Experiment document builders.

Each module here exports a single ``*_document(config=None) -> dict`` function
that returns a process_bigraph composite document. The functions are
re-exported here for convenient registry access.
"""
from multi_cell.experiments.documents.daughter_machine import daughter_machine_document
from multi_cell.experiments.documents.attachment import attachment_document
from multi_cell.experiments.documents.glucose_growth import glucose_growth_document
from multi_cell.experiments.documents.bending_pressure import bending_pressure_document
from multi_cell.experiments.documents.mother_machine import mother_machine_document
from multi_cell.experiments.documents.biofilm import biofilm_document

__all__ = [
    'daughter_machine_document',
    'attachment_document',
    'glucose_growth_document',
    'bending_pressure_document',
    'mother_machine_document',
    'biofilm_document',
]
