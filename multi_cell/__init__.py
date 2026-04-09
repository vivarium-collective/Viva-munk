"""
Registration-related functions
"""

from process_bigraph import allocate_core, register_types as pb_register_types
from bigraph_viz import register_types as viz_register_types
from bigraph_schema.schema import Tuple, Float, Enum
from multi_cell.processes.multibody import PymunkProcess
from multi_cell.processes.grow_divide import GrowDivide
from multi_cell.processes.remove_crossing import RemoveCrossing
from multi_cell.processes.secrete_eps import SecreteEPS


def register_pymunk_types(core):
    pymunk_agent = {
        'type': Enum(_values=['circle', 'segment']),
        'mass': Float(),
        'radius': Float(),
        'length': Float(_default=0.0),
        'angle': Float(_default=0.0),
        'inertia': Float(_default=float('inf')),
        'location': Tuple(_values=[Float(), Float()]),
        'velocity': Tuple(_values=[Float(), Float()]),
        'elasticity': Float(_default=0.0),
        'friction': Float(_default=0.8),
    }

    core.register_type('pymunk_agent', pymunk_agent)


def register_processes(core):
    core.register_link('PymunkProcess', PymunkProcess)
    core.register_link('GrowDivide', GrowDivide)
    core.register_link('RemoveCrossing', RemoveCrossing)
    core.register_link('SecreteEPS', SecreteEPS)


def core_import(core=None, config=None):
    if not core:
        core = allocate_core()
    pb_register_types(core)
    viz_register_types(core)
    register_pymunk_types(core)
    register_processes(core)
    return core
