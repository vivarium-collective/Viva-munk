"""
RemoveCrossing — a Step that removes agents crossing a configurable y-threshold.

Used in mother-machine experiments to model the flow channel:
cells that grow past the channel tops are swept away.
"""
from process_bigraph import Step


class RemoveCrossing(Step):
    config_schema = {
        'crossing_y': {'_type': 'float', '_default': 500.0},
        'agents_key': {'_type': 'string', '_default': 'cells'},
    }

    def inputs(self):
        return {
            'agents': 'map[pymunk_agent]',
        }

    def outputs(self):
        return {
            'agents': 'map[pymunk_agent]',
        }

    def update(self, state):
        agents = state.get('agents', {}) or {}
        crossing_y = self.config['crossing_y']

        remove = []
        for aid, agent in agents.items():
            loc = agent.get('location', (0.0, 0.0))
            if loc and loc[1] > crossing_y:
                remove.append(aid)

        if not remove:
            return {'agents': {}}

        return {'agents': {'_remove': remove}}


def make_remove_crossing_process(crossing_y=500.0, agents_key='cells'):
    """Create a RemoveCrossing step spec."""
    return {
        '_type': 'step',
        'address': 'local:RemoveCrossing',
        'config': {
            'crossing_y': crossing_y,
            'agents_key': agents_key,
        },
        'inputs': {
            'agents': [agents_key],
        },
        'outputs': {
            'agents': [agents_key],
        },
    }
