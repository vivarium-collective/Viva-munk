"""
growth and division
"""
from process_bigraph import Process, default

def get_grow_divide_schema(core, config=None):
    config = config or core.default(GrowDivide.config_schema)
    agents_key = config.get('agents_key', 'agents')
    return {
        agents_key: {
            '_type': 'map',
            '_value': {
                'grow_divide': {
                    '_type': 'process',
                    'address': default('string', 'local:GrowDivide'),
                    'config': default('quote', config),
                    '_inputs': {
                        'agent_id':'string',
                        'mass': 'float',
                    },
                    '_outputs':  {
                        'mass': 'float',
                        'agents': 'map'
                    },
                    'inputs': default(
                        'tree[wires]', {
                            'agent_id': ['id'],
                            'mass': ['mass'],
                        }),
                    'outputs': default(
                        'tree[wires]', {
                            'mass': ['mass'],
                            'agents': [agents_key],
                        })
                }
            }
        }
    }

class GrowDivide(Process):
    config_schema = {
        'rate': 'float',
        'threshold': {'_type': 'float', '_default': 100.0},
    }

    # def initialize(self, config):
    #     breakpoint()

    def inputs(self):
        return {
            'agent_id': 'string',
            'mass': 'float'
        }

    def outputs(self):
        return {
            'mass': 'float',
            'agents': 'map'
        }

    def update(self, state, interval):
        # this calculates a delta
        divide_update = {}
        if state['mass'] >= self.config['threshold']:
            pass
            # TODO divide
            # position = self.get_boundary_position()
            new_particle = {}
            #
            # new_particle['id'] = pid
            # divide_update['_add'][pid] = new_particle
            #
            #
            # divide_update = {
            #
            # }

        return {
            'mass': state['mass'] * self.config['rate'] * interval,
            'agents': divide_update
        }
