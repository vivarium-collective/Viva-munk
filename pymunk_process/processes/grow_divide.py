"""
growth and division
"""
import math
from process_bigraph import Process, default
from pymunk_process.processes.multibody import build_microbe, daughter_locations


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

        # --- Division ---
        if m_new >= float(self.config['threshold']):
            half_mass = 0.5 * m_new
            loc1, loc2 = daughter_locations(agent)
            inherit_keys = [
                'elasticity', 'friction', 'angle', 'radius', 'length', 'velocity', 'type'
            ]
            base = {k: agent[k] for k in inherit_keys if k in agent}

            # small velocity nudges to separate daughters
            vx, vy = agent.get('velocity', (0.0, 0.0))
            angle = float(agent.get('angle', 0.0))
            eps_perp = 0.1  # circle: nudge perpendicular to "angle"
            eps_axis = 0.1  # segment: nudge along the rod axis

            if t == 'circle':
                # constant density => r_d = r * sqrt((m_new/2)/m)
                r = float(agent.get('radius', 1.0)) or 1.0
                r_d = r * (half_mass / m) ** 0.5

                # perpendicular nudge
                dvx = eps_perp * math.cos(angle + math.pi / 2.0)
                dvy = eps_perp * math.sin(angle + math.pi / 2.0)

                d1 = dict(base)
                d1.update({
                    'type': 'circle',
                    'mass': half_mass,
                    'radius': r_d,
                    'location': (float(loc1[0]), float(loc1[1])),
                    'velocity': (float(vx + dvx), float(vy + dvy)),
                })

                d2 = dict(base)
                d2.update({
                    'type': 'circle',
                    'mass': half_mass,
                    'radius': r_d,
                    'location': (float(loc2[0]), float(loc2[1])),
                    'velocity': (float(vx - dvx), float(vy - dvy)),
                })
            elif t == 'segment':
                L = float(agent['length'])
                r = float(agent['radius'])
                # constant density, fixed radius => L_d = L * ((m_new/2)/m)
                L_d = L * (half_mass / m)

                # axis nudge
                dvx = eps_axis * math.cos(angle)
                dvy = eps_axis * math.sin(angle)

                d1 = dict(base)
                d1.update({
                    'type': 'segment',
                    'mass': half_mass,
                    'length': L_d,
                    'radius': r,
                    'angle': angle,
                    'location': (float(loc1[0]), float(loc1[1])),
                    'velocity': (float(vx + dvx), float(vy + dvy)),
                })

                d2 = dict(base)
                d2.update({
                    'type': 'segment',
                    'mass': half_mass,
                    'length': L_d,
                    'radius': r,
                    'angle': angle,
                    'location': (float(loc2[0]), float(loc2[1])),
                    'velocity': (float(vx - dvx), float(vy - dvy)),
                })

                # IDs for daughters: deterministic suffixes are handy for debugging
            d1_id = f"{agent_id}_0"
            d2_id = f"{agent_id}_1"
            d1['id'] = d1_id
            d2['id'] = d2_id

            if not '_add 'in update:
                update['_add'] = {}
            update['_add'][d1_id] = d1
            update['_add'][d2_id] = d2
            if not '_remove' in update:
                update['_remove'] = []
            update['_remove'].append(agent_id)

        return {
            'agents': update
        }