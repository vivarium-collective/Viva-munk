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
        'type': 'string',   # 'circle' or 'segment'
        'mass': 'float',
        'radius': 'float',  # used by both circle and segment (segment = capsule radius)

        # OPTIONAL/segment-only fields: make them present with safe defaults
        'length': {'_type': 'float', '_default': 0.0},
        'angle':  {'_type': 'float', '_default': 0.0},

        # Common mechanical/visual fields
        'inertia':   {'_type': 'float', '_default': float('inf')},
        'location':  ('float', 'float'),
        'velocity':  ('float', 'float'),
        'elasticity': {'_type': 'float', '_default': 0.0},
        'friction':   {'_type': 'float', '_default': 0.8},
    }

    core.register('pymunk_agent', pymunk_agent)







    circle_agent_type = {
        'type': 'string',  # TODO this should be 'enum[circle, segment]'
        'mass': 'float',
        'radius': 'float',
        'inertia': {'_type': 'float',
                    '_default': float('inf')
                    },
        'location': ('float', 'float'),
        'velocity': ('float', 'float'),
        'elasticity': 'float'
    }
    segment_agent_type = {
        '_inherit': 'circle_agent',
        'length': 'float',
        'angle': 'float',
    }

    core.register('circle_agent', circle_agent_type)
    core.register('segment_agent', segment_agent_type)


def register_processes(core):
    core.process_registry.register('pymunk_process', PymunkProcess)
    core.process_registry.register('grow_divide', GrowDivide)


def get_pymunk_core(config=None):
    core = ProcessTypes()
    register_types(core)
    register_processes(core)
    return core
