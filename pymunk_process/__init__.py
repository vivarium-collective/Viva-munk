"""
Registration-related functions
"""

from process_bigraph import ProcessTypes
from pymunk_process.processes.multibody import PymunkProcess
from pymunk_process.processes.grow_divide import GrowDivide


def register_types(core):
    core.register('point2d', '(length|length)')
    core.register('boundary', {'location': 'point2d',
                               'angle': 'float',
                               'length': 'length',
                               'width': 'length',
                               'mass': 'mass',
                               'velocity': 'length/time'})

    # One schema that works for both circles and segments
    pymunk_agent = {
        # if your TS supports enum, use 'enum[circle, segment]'; otherwise keep string
        'type': 'enum[circle,segment]',   # 'circle' or 'segment'
        'mass': 'float',
        'radius': 'float',  # used by both circle and segment (segment = capsule radius)

        # OPTIONAL/segment-only fields: make them present with safe defaults
        'length': {'_type': 'float', '_apply': 'set', '_default': 0.0},
        'angle':  {'_type': 'float', '_apply': 'set', '_default': 0.0},

        # Common mechanical/visual fields
        'inertia':   {'_type': 'float', '_default': float('inf'), '_apply': 'set'},
        'location':  (
            {'_type': 'float', '_apply': 'set'},
            {'_type': 'float', '_apply': 'set'}
        ),
        'velocity':  (
            {'_type': 'float', '_apply': 'set'},
            {'_type': 'float', '_apply': 'set'}
        ),
        'elasticity': {'_type': 'float', '_apply': 'set', '_default': 0.0},
        'friction':   {'_type': 'float', '_apply': 'set', '_default': 0.8},
    }

    core.register('pymunk_agent', pymunk_agent)


def register_processes(core):
    core.process_registry.register('PymunkProcess', PymunkProcess)
    core.process_registry.register('GrowDivide', GrowDivide)


def core_import(core=None, config=None):
    if not core:
        core = ProcessTypes()
    register_types(core)
    register_processes(core)
    return core
