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
                        'agents': 'map[pymunk_agent]',
                    },
                    '_outputs':  {
                        'agents': 'map[pymunk_agent]'
                    },
                    'inputs': default(
                        'tree[wires]', {
                            'agent_id': ['id'],
                            'agents': ['..', '..', agents_key],
                        }),
                    'outputs': default(
                        'tree[wires]', {
                            'agents': ['..', '..', agents_key],
                        })
                }
            }
        }
    }

class GrowDivide(Process):
    config_schema = {
        'rate': {'_type': 'float', '_default': 0.01},
        'threshold': {'_type': 'float', '_default': 100.0},
    }

    def inputs(self):
        return {
            'agent_id': 'string',
            'agents': 'map[pymunk_agent]',
        }

    def outputs(self):
        return {
            'agents': 'map[pymunk_agent]',
        }

    def update(self, state, interval):
        agent_id = state['agent_id']
        agent = state['agents'][agent_id]

        # mass growth
        m = float(agent.get('mass', 0.0))
        if not (m > 0.0):
            return {'agents': {}}  # nothing to do / malformed
        dm = m * float(self.config['rate']) * float(interval)
        m_new = m + dm

        # infer shape
        t = agent.get('type')
        if t not in ('circle', 'segment'):
            # heuristic: has length? -> segment ; else circle
            t = 'segment' if ('length' in agent and float(agent.get('length', 0.0)) > 0.0) else 'circle'

        update = {}

        if t == 'circle':
            # Keep density constant: m = ρ * π r^2  => r ∝ sqrt(m)
            r = float(agent.get('radius', 0.0))
            if r > 0.0:
                scale = (m_new / m) ** 0.5
                r_new = r * scale
                dr = r_new - r
                update = {
                    agent_id: {
                        'mass': dm,
                        'radius': dr,
                        # no change to location/angle here
                    }
                }
            else:
                # fallback: just mass if radius missing/nonpositive
                update = {agent_id: {'mass': dm}}

        if t == 'segment':
            # Capsule with fixed radius, grow length: m = ρ * (2 r) * L  => L ∝ m (if r fixed)
            L = float(agent.get('length', 0.0))
            r = float(agent.get('radius', 0.0))
            if L > 0.0 and r > 0.0:
                scale = (m_new / m)
                L_new = L * scale
                dL = L_new - L
                update = {
                    agent_id: {
                        'mass': dm,
                        'length': L_new,
                        # keep radius fixed; angle/location unchanged
                    }
                }
            else:
                # fallback: just mass if geometry missing
                update = {agent_id: {'mass': dm}}

        # Threshold hook: leave division for later
        if m_new >= float(self.config.get('threshold', 100.0)):
            # TODO: implement division (create two daughters, remove parent)
            # For now, just keep growing; or you could clamp here if desired.
            pass

        return {'agents': update}