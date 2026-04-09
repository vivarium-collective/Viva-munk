"""
growth and division
"""
import math, random
from process_bigraph import Process
from multi_cell.processes.multibody import build_microbe, daughter_locations

def make_grow_divide_process(config=None, agents_key='cells', interval=30.0):
    """Create a grow_divide process spec to embed in an agent's state."""
    config = config or {}
    config.setdefault('agents_key', agents_key)
    return {
        '_type': 'process',
        'address': 'local:GrowDivide',
        'config': config,
        'interval': interval,
        'inputs': {
            'agent_id': ['id'],
            'agents': ['..', '..', agents_key],
        },
        'outputs': {
            'agents': ['..', '..', agents_key],
        }
    }


def add_grow_divide_to_agents(initial_state, agents_key='cells', config=None):
    """Add grow_divide process to each agent in the initial state."""
    agents = initial_state.get(agents_key, {})
    for agent_id, agent in agents.items():
        agent['grow_divide'] = make_grow_divide_process(
            config=dict(config) if config else None,
            agents_key=agents_key,
        )
    return initial_state

class GrowDivide(Process):
    config_schema = {
        # global defaults (used if agent doesn't carry its own config)
        'agents_key': {'_type': 'string', '_default': 'agents'},
        'rate': {'_type': 'float', '_default': 0.01},
        'threshold': {'_type': 'float', '_default': 100.0},

        # --- mutation controls for daughters ---
        # If False, daughters inherit mother exactly.
        'mutate': {'_type': 'boolean',  '_default': False},

        # 'mult' => multiplicative (val *= exp(N(0, sigma)))
        # 'add'  => additive       (val += N(0, sigma))
        'mutation_mode': {'_type': 'string', '_default': 'mult'},  # 'enum[mult,add]'

        # Spread for rate and threshold mutations
        # - If mode == 'mult': interpreted as sigma in log-space (i.e., exp(N(0,sigma)))
        # - If mode == 'add' : interpreted as sigma in linear space
        'mutation_sigma_rate':      {'_type': 'float', '_default': 0.10},
        'mutation_sigma_threshold': {'_type': 'float', '_default': 0.10},

        # Optional bounds (applied post-mutation)
        'rate_min':      {'_type': 'float', '_default': 1e-6},
        'rate_max':      {'_type': 'float', '_default': 10.0},
        'threshold_min': {'_type': 'float', '_default': 1e-3},
        'threshold_max': {'_type': 'float', '_default': 1e9},
    }

    # ---------------- internal helpers ----------------

    def _get_agent_gd_params(self, agent):
        """Get (rate, threshold) from agent.grow_divide.config if present, else process defaults."""
        gd = agent.get('grow_divide', {})
        cfg = gd.get('config', {})
        rate = float(cfg.get('rate', self.config['rate']))
        thr  = float(cfg.get('threshold', self.config['threshold']))
        return rate, thr

    def _mutate_value(self, val, sigma, mode, vmin, vmax, rng):
        if sigma <= 0.0:
            return max(vmin, min(vmax, val))
        if mode == 'mult':
            # multiplicative log-normal style
            mutated = val * math.exp(rng.gauss(0.0, sigma))
        else:  # 'add'
            mutated = val + rng.gauss(0.0, sigma)
        # clamp
        return max(vmin, min(vmax, mutated))

    def _mutate_daughter_gd_config(self, mother_rate, mother_thr, rng):
        if not self.config['mutate']:
            rate = mother_rate
            thr  = mother_thr
        else:
            rate = self._mutate_value(
                mother_rate,
                float(self.config['mutation_sigma_rate']),
                self.config['mutation_mode'],
                float(self.config['rate_min']),
                float(self.config['rate_max']),
                rng
            )
            thr = self._mutate_value(
                mother_thr,
                float(self.config['mutation_sigma_threshold']),
                self.config['mutation_mode'],
                float(self.config['threshold_min']),
                float(self.config['threshold_max']),
                rng
            )
        return {
            'agents_key': self.config['agents_key'],
            'rate': rate,
            'threshold': thr,
        }

    # ---------------- IO ----------------

    def inputs(self):
        return {
            'agent_id': 'string',
            'agents': 'map[pymunk_agent]',
        }

    def outputs(self):
        return {
            'agents': 'map[pymunk_agent]',
        }

    # ---------------- main update ----------------

    def update(self, state, interval):
        rng = random  # use module RNG; swap for instance RNG if needed

        agent_id = state['agent_id']
        agent = state['agents'].get(agent_id, {})
        if not agent:
            return {'agents': {}}

        # per-agent (or default) kinetics
        rate, thresh = self._get_agent_gd_params(agent)

        # mass growth
        m = float(agent.get('mass', 0.0))
        if not (m > 0.0):
            return {'agents': {}}  # nothing to do / malformed

        dt = float(interval)
        dm = m * float(rate) * dt
        m_new = m + dm

        # infer shape
        t = agent.get('type')
        if t not in ('circle', 'segment'):
            t = 'segment' if ('length' in agent and float(agent.get('length', 0.0)) > 0.0) else 'circle'

        # Return deltas for accumulate fields (mass, radius, length).
        # Must include 'type' because Enum has set semantics and would become None if omitted.
        if t == 'circle':
            r = float(agent.get('radius', 0.0))
            if r > 0.0:
                r_new = r * (m_new / m) ** 0.5
                update = {agent_id: {'type': t, 'mass': dm, 'radius': r_new - r}}
            else:
                update = {agent_id: {'type': t, 'mass': dm}}

        else:  # segment
            L = float(agent.get('length', 0.0))
            r = float(agent.get('radius', 0.0))
            if L > 0.0 and r > 0.0:
                L_new = L * (m_new / m)
                update = {agent_id: {'type': t, 'mass': dm, 'length': L_new - L}}
            else:
                update = {agent_id: {'type': t, 'mass': dm}}

        # --- Division ---
        if m_new >= float(thresh):
            half_mass = 0.5 * m_new

            inherit_keys = [
                'elasticity', 'friction', 'angle', 'radius', 'length', 'velocity', 'type'
            ]
            base = {k: agent[k] for k in inherit_keys if k in agent}

            vx, vy = agent.get('velocity', (0.0, 0.0))
            angle = float(agent.get('angle', 0.0))

            if t == 'circle':
                r = float(agent.get('radius', 1.0)) or 1.0
                r_d = r * (half_mass / m) ** 0.5
                loc1, loc2 = daughter_locations(
                    agent, gap=r_d * 0.1, daughter_radius=r_d)
                # Small perpendicular nudge
                eps = r_d * 0.01
                dvx = eps * math.cos(angle + math.pi/2)
                dvy = eps * math.sin(angle + math.pi/2)

                d1 = dict(base, **{
                    'type': 'circle',
                    'mass': half_mass,
                    'radius': r_d,
                    'location': (float(loc1[0]), float(loc1[1])),
                    'velocity': (float(vx + dvx), float(vy + dvy)),
                })
                d2 = dict(base, **{
                    'type': 'circle',
                    'mass': half_mass,
                    'radius': r_d,
                    'location': (float(loc2[0]), float(loc2[1])),
                    'velocity': (float(vx - dvx), float(vy - dvy)),
                })

            else:  # segment
                L = float(agent['length'])
                r = float(agent['radius'])
                L_d = L * (half_mass / m)
                loc1, loc2 = daughter_locations(
                    agent, gap=r * 0.1, daughter_length=L_d, daughter_radius=r)
                # Tiny axial nudge
                eps = r * 0.01
                dvx = eps * math.cos(angle)
                dvy = eps * math.sin(angle)

                d1 = dict(base, **{
                    'type': 'segment',
                    'mass': half_mass,
                    'length': L_d,
                    'radius': r,
                    'angle': angle,
                    'location': (float(loc1[0]), float(loc1[1])),
                    'velocity': (float(vx + dvx), float(vy + dvy)),
                })
                d2 = dict(base, **{
                    'type': 'segment',
                    'mass': half_mass,
                    'length': L_d,
                    'radius': r,
                    'angle': angle,
                    'location': (float(loc2[0]), float(loc2[1])),
                    'velocity': (float(vx - dvx), float(vy - dvy)),
                })

            # Per-daughter grow_divide configs (mutated)
            mother_rate, mother_thr = rate, thresh
            gd_cfg_1 = self._mutate_daughter_gd_config(mother_rate, mother_thr, rng)
            gd_cfg_2 = self._mutate_daughter_gd_config(mother_rate, mother_thr, rng)

            # Ensure clean process spec (drop any instance handles if present)
            d1_gd = dict(agent.get('grow_divide', {}))
            d1_gd.pop('instance', None)
            d1_gd['config'] = gd_cfg_1
            d1['grow_divide'] = d1_gd

            d2_gd = dict(agent.get('grow_divide', {}))
            d2_gd.pop('instance', None)
            d2_gd['config'] = gd_cfg_2
            d2['grow_divide'] = d2_gd

            # IDs for daughters
            d1_id = f"{agent_id}_0"
            d2_id = f"{agent_id}_1"
            d1['id'] = d1_id
            d2['id'] = d2_id

            update.setdefault('_add', {})
            update['_add'][d1_id] = d1
            update['_add'][d2_id] = d2

            update.setdefault('_remove', [])
            update['_remove'].append(agent_id)

        return {'agents': update}
