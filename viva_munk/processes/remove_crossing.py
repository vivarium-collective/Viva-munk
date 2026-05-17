"""
RemoveCrossing — a Step that removes agents crossing a configurable boundary.

Supports per-axis thresholds: cells crossing above/below/left/right of any
configured threshold are removed. Used to model flow channels and other
absorbing boundaries.
"""
from process_bigraph import Step


class RemoveCrossing(Step):
    config_schema = {
        # Axis-aligned threshold limits. Set any to None to disable that side.
        'x_max': {'_type': 'maybe[float]', '_default': None},
        'x_min': {'_type': 'maybe[float]', '_default': None},
        'y_max': {'_type': 'maybe[float]', '_default': None},
        'y_min': {'_type': 'maybe[float]', '_default': None},
        # Backwards compatibility
        'crossing_y': {'_type': 'maybe[float]', '_default': None},
        'agents_key': {'_type': 'string', '_default': 'cells'},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        # Resolve effective thresholds (legacy crossing_y -> y_max)
        self.x_max = self.config.get('x_max')
        self.x_min = self.config.get('x_min')
        self.y_max = self.config.get('y_max')
        self.y_min = self.config.get('y_min')
        if self.config.get('crossing_y') is not None and self.y_max is None:
            self.y_max = self.config['crossing_y']

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
        x_max, x_min = self.x_max, self.x_min
        y_max, y_min = self.y_max, self.y_min

        remove = []
        for aid, agent in agents.items():
            loc = agent.get('location')
            if loc is None:
                continue
            x, y = loc[0], loc[1]
            if x_max is not None and x > x_max:
                remove.append(aid)
            elif x_min is not None and x < x_min:
                remove.append(aid)
            elif y_max is not None and y > y_max:
                remove.append(aid)
            elif y_min is not None and y < y_min:
                remove.append(aid)

        if not remove:
            return {'agents': {}}

        return {'agents': {'_remove': remove}}


def make_remove_crossing_process(
    crossing_y=None, x_max=None, x_min=None, y_max=None, y_min=None,
    agents_key='cells',
):
    """Create a RemoveCrossing step spec.

    Pass any combination of x_max, x_min, y_max, y_min thresholds.
    `crossing_y` is kept for backwards compatibility (maps to y_max).
    """
    config = {'agents_key': agents_key}
    if x_max is not None:
        config['x_max'] = x_max
    if x_min is not None:
        config['x_min'] = x_min
    if y_max is not None:
        config['y_max'] = y_max
    if y_min is not None:
        config['y_min'] = y_min
    if crossing_y is not None:
        config['crossing_y'] = crossing_y
    return {
        '_type': 'step',
        'address': 'local:RemoveCrossing',
        'config': config,
        'inputs': {
            'agents': [agents_key],
        },
        'outputs': {
            'agents': [agents_key],
        },
    }
