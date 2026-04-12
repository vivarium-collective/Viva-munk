"""
growth and division — two Process classes:

GrowDivide       — simple mass-based exponential growth, divides at a mass
                   threshold. Fast, suitable for daughter_machine and similar
                   open-chamber experiments where framework overhead matters.

AdderGrowDivide  — length-based exponential growth with the "adder" division
                   rule (Taheri-Araghi et al. 2015). Each cell samples its own
                   growth rate α and added-length target Δ, and divides when
                   L >= L_birth + Δ. More biologically realistic but creates
                   denser populations and more process-instance churn.
"""
import math, random
from process_bigraph import Process
from multi_cell.processes.multibody import build_microbe, daughter_locations


# =====================================================================
# Helpers shared by both classes
# =====================================================================

def _build_segment_daughters(
    agent, agent_id, angle, r, L_d1, L_d2, m_d1, m_d2,
    gd_spec_1, gd_spec_2,
):
    """Build daughter dicts for segment cells (shared by both processes).

    Returns the (d1, d2, d1_id, d2_id) tuple ready to stuff into an
    ``_add`` update dict.
    """
    inherit_keys = ['elasticity', 'friction', 'angle', 'radius', 'velocity', 'type']
    base = {k: agent[k] for k in inherit_keys if k in agent}
    half_adhesins = float(agent.get('adhesins', 0.0) or 0.0) / 2.0
    vx, vy = agent.get('velocity', (0.0, 0.0))

    is_attached = float(agent.get('attached', 0.0) or 0.0) >= 0.5
    mx, my = agent.get('location', (0.0, 0.0))
    if is_attached:
        loc1 = (float(mx), float(my))
        perp_x = -math.sin(angle)
        perp_y = math.cos(angle)
        if perp_y < 0:
            perp_x, perp_y = -perp_x, -perp_y
        push = max(L_d1, L_d2) + 2 * r + 0.2
        loc2 = (float(mx + perp_x * push), float(my + perp_y * push))
    else:
        loc1, loc2 = daughter_locations(
            agent, gap=r * 0.1,
            daughter_length=max(L_d1, L_d2), daughter_radius=r,
        )

    eps = r * 0.01
    dvx = eps * math.cos(angle)
    dvy = eps * math.sin(angle)

    # Bending-cell ghost fix (commit d053a22): seed a fresh straight
    # polyline only when the mother was a bending cell.
    mother_has_polyline = bool(agent.get('polyline'))
    d1_extra, d2_extra = {}, {}
    if mother_has_polyline:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        hx1, hy1 = cos_a * (L_d1 / 2), sin_a * (L_d1 / 2)
        hx2, hy2 = cos_a * (L_d2 / 2), sin_a * (L_d2 / 2)
        d1_extra['polyline'] = [
            (float(loc1[0] - hx1), float(loc1[1] - hy1)),
            (float(loc1[0] + hx1), float(loc1[1] + hy1)),
        ]
        d2_extra['polyline'] = [
            (float(loc2[0] - hx2), float(loc2[1] - hy2)),
            (float(loc2[0] + hx2), float(loc2[1] + hy2)),
        ]

    d1 = dict(base, **{
        'type': 'segment', 'mass': m_d1, 'length': L_d1, 'radius': r,
        'angle': angle,
        'location': (float(loc1[0]), float(loc1[1])),
        'velocity': (float(vx + dvx), float(vy + dvy)),
        'adhesins': half_adhesins,
        **d1_extra,
    })
    d2 = dict(base, **{
        'type': 'segment', 'mass': m_d2, 'length': L_d2, 'radius': r,
        'angle': angle,
        'location': (float(loc2[0]), float(loc2[1])),
        'velocity': (float(vx - dvx), float(vy - dvy)),
        'adhesins': half_adhesins,
        **d2_extra,
    })

    d1['grow_divide'] = gd_spec_1
    d2['grow_divide'] = gd_spec_2

    # Asymmetric inclusion-body segregation: the entire aggregate mass
    # goes to daughter 0 ("old-pole" lineage); daughter 1 is rejuvenated.
    ib_mother = float(agent.get('inclusion_body', 0.0) or 0.0)
    d1['inclusion_body'] = ib_mother
    d2['inclusion_body'] = 0.0
    if 'inclusion_body_proc' in agent:
        ib_proc = dict(agent['inclusion_body_proc'])
        ib_proc.pop('instance', None)
        ib_proc.setdefault('_type', 'process')
        d1['inclusion_body_proc'] = dict(ib_proc)
        d2['inclusion_body_proc'] = dict(ib_proc)

    d1_id = f"{agent_id}_0"
    d2_id = f"{agent_id}_1"
    d1['id'] = d1_id
    d2['id'] = d2_id

    return d1, d2, d1_id, d2_id


# =====================================================================
# Process-spec factories
# =====================================================================

def make_grow_divide_process(config=None, agents_key='cells', interval=30.0):
    """Create a GrowDivide process spec to embed in an agent's state."""
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
    """Add GrowDivide process to each agent in the initial state."""
    agents = initial_state.get(agents_key, {})
    for agent_id, agent in agents.items():
        agent['grow_divide'] = make_grow_divide_process(
            config=dict(config) if config else None,
            agents_key=agents_key,
        )
    return initial_state


def make_adder_grow_divide_process(config=None, agents_key='cells', interval=30.0):
    """Create an AdderGrowDivide process spec to embed in an agent's state."""
    config = config or {}
    config.setdefault('agents_key', agents_key)
    return {
        '_type': 'process',
        'address': 'local:AdderGrowDivide',
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


def add_adder_grow_divide_to_agents(initial_state, agents_key='cells', config=None):
    """Add AdderGrowDivide process to each agent in the initial state."""
    agents = initial_state.get(agents_key, {})
    for agent_id, agent in agents.items():
        agent['grow_divide'] = make_adder_grow_divide_process(
            config=dict(config) if config else None,
            agents_key=agents_key,
        )
    return initial_state


# =====================================================================
# GrowDivide — simple mass-based growth
# =====================================================================

class GrowDivide(Process):
    config_schema = {
        'agents_key': {'_type': 'string', '_default': 'agents'},
        # Default baseline growth: ln(2)/2400 s ≈ 40 min doubling (E. coli
        # rich-medium/minimal midpoint). Modulations (pressure, nutrient,
        # IB burden) only slow cells down from this baseline.
        'rate': {'_type': 'float', '_default': 0.000289},
        'threshold': {'_type': 'float', '_default': 100.0},
        'mutate': {'_type': 'boolean', '_default': False},
        'mutation_mode': {'_type': 'string', '_default': 'mult'},
        'mutation_sigma_rate': {'_type': 'float', '_default': 0.10},
        'mutation_sigma_threshold': {'_type': 'float', '_default': 0.10},
        'rate_min': {'_type': 'float', '_default': 1e-6},
        'rate_max': {'_type': 'float', '_default': 10.0},
        'threshold_min': {'_type': 'float', '_default': 1e-3},
        'threshold_max': {'_type': 'float', '_default': 1e9},
        'pressure_k': {'_type': 'float', '_default': 5.0},
        'nutrient_key': {'_type': 'string', '_default': ''},
        'nutrient_km': {'_type': 'float', '_default': 0.5},
        'nutrient_yield': {'_type': 'float', '_default': 1.0},
    }

    def _get_agent_gd_params(self, agent):
        gd = agent.get('grow_divide', {})
        cfg = gd.get('config', {})
        rate = float(cfg.get('rate', self.config['rate']))
        thr = float(cfg.get('threshold', self.config['threshold']))
        return rate, thr

    def _mutate_value(self, val, sigma, mode, vmin, vmax, rng):
        if sigma <= 0.0:
            return max(vmin, min(vmax, val))
        if mode == 'mult':
            mutated = val * math.exp(rng.gauss(0.0, sigma))
        else:
            mutated = val + rng.gauss(0.0, sigma)
        return max(vmin, min(vmax, mutated))

    def _mutate_daughter_gd_config(self, mother_rate, mother_thr, rng):
        if not self.config['mutate']:
            rate, thr = mother_rate, mother_thr
        else:
            rate = self._mutate_value(
                mother_rate, float(self.config['mutation_sigma_rate']),
                self.config['mutation_mode'],
                float(self.config['rate_min']), float(self.config['rate_max']), rng,
            )
            thr = self._mutate_value(
                mother_thr, float(self.config['mutation_sigma_threshold']),
                self.config['mutation_mode'],
                float(self.config['threshold_min']), float(self.config['threshold_max']), rng,
            )
        return {
            'agents_key': self.config['agents_key'],
            'rate': rate,
            'threshold': thr,
            'pressure_k': float(self.config.get('pressure_k', 5.0)),
            'nutrient_key': self.config.get('nutrient_key', '') or '',
            'nutrient_km': float(self.config.get('nutrient_km', 0.5)),
            'nutrient_yield': float(self.config.get('nutrient_yield', 1.0)),
        }

    def inputs(self):
        return {'agent_id': 'string', 'agents': 'map[pymunk_agent]'}

    def outputs(self):
        return {'agents': 'map[pymunk_agent]'}

    def update(self, state, interval):
        rng = random
        agent_id = state['agent_id']
        agent = state['agents'].get(agent_id, {})
        if not agent:
            return {'agents': {}}

        rate, thresh = self._get_agent_gd_params(agent)

        pressure = float(agent.get('pressure', 0.0) or 0.0)
        pressure_k = float(self.config.get('pressure_k', 5.0))
        if pressure > 0 and pressure_k > 0:
            rate = rate * math.exp(-pressure / pressure_k)

        nutrient_key = self.config.get('nutrient_key', '') or ''
        if nutrient_key:
            local_dict = agent.get('local') or {}
            local_S = float(local_dict.get(nutrient_key, 0.0) or 0.0)
            km = float(self.config.get('nutrient_km', 0.5))
            if local_S <= 0.0:
                rate = 0.0
            else:
                rate = rate * (local_S / (km + local_S))

        m = float(agent.get('mass', 0.0))
        if not (m > 0.0):
            return {'agents': {}}

        dt = float(interval)
        dm = m * float(rate) * dt
        m_new = m + dm

        t = agent.get('type')
        if t not in ('circle', 'segment'):
            t = 'segment' if ('length' in agent and float(agent.get('length', 0.0)) > 0.0) else 'circle'

        if t == 'circle':
            r = float(agent.get('radius', 0.0))
            if r > 0.0:
                r_new = r * (m_new / m) ** 0.5
                update = {agent_id: {'type': t, 'mass': dm, 'radius': r_new - r}}
            else:
                update = {agent_id: {'type': t, 'mass': dm}}
        else:
            L = float(agent.get('length', 0.0))
            r = float(agent.get('radius', 0.0))
            if L > 0.0 and r > 0.0:
                L_new = L * (m_new / m)
                update = {agent_id: {'type': t, 'mass': dm, 'length': L_new - L}}
            else:
                update = {agent_id: {'type': t, 'mass': dm}}

        if nutrient_key and dm > 0.0:
            yield_coef = float(self.config.get('nutrient_yield', 1.0)) or 1.0
            update[agent_id]['exchange'] = {nutrient_key: -dm / yield_coef}

        # --- Division ---
        if m_new >= float(thresh):
            half_mass = 0.5 * m_new
            angle = float(agent.get('angle', 0.0))

            if t == 'circle':
                r = float(agent.get('radius', 1.0)) or 1.0
                r_d = r * (half_mass / m) ** 0.5
                loc1, loc2 = daughter_locations(agent, gap=r_d * 0.1, daughter_radius=r_d)
                eps = r_d * 0.01
                dvx = eps * math.cos(angle + math.pi / 2)
                dvy = eps * math.sin(angle + math.pi / 2)
                vx, vy = agent.get('velocity', (0.0, 0.0))
                half_adhesins = float(agent.get('adhesins', 0.0) or 0.0) / 2.0
                inherit_keys = ['elasticity', 'friction', 'angle', 'radius', 'length', 'velocity', 'type']
                base = {k: agent[k] for k in inherit_keys if k in agent}
                d1 = dict(base, type='circle', mass=half_mass, radius=r_d,
                          location=(float(loc1[0]), float(loc1[1])),
                          velocity=(float(vx + dvx), float(vy + dvy)),
                          adhesins=half_adhesins)
                d2 = dict(base, type='circle', mass=half_mass, radius=r_d,
                          location=(float(loc2[0]), float(loc2[1])),
                          velocity=(float(vx - dvx), float(vy - dvy)),
                          adhesins=half_adhesins)
                gd_cfg_1 = self._mutate_daughter_gd_config(rate, thresh, rng)
                gd_cfg_2 = self._mutate_daughter_gd_config(rate, thresh, rng)
                d1_gd = dict(agent.get('grow_divide', {})); d1_gd.pop('instance', None); d1_gd.setdefault('_type', 'process'); d1_gd['config'] = gd_cfg_1
                d2_gd = dict(agent.get('grow_divide', {})); d2_gd.pop('instance', None); d2_gd.setdefault('_type', 'process'); d2_gd['config'] = gd_cfg_2
                d1['grow_divide'] = d1_gd; d2['grow_divide'] = d2_gd
                # Asymmetric IB segregation (see _build_segment_daughters).
                ib_mother = float(agent.get('inclusion_body', 0.0) or 0.0)
                d1['inclusion_body'] = ib_mother
                d2['inclusion_body'] = 0.0
                if 'inclusion_body_proc' in agent:
                    ib_proc = dict(agent['inclusion_body_proc'])
                    ib_proc.pop('instance', None)
                    ib_proc.setdefault('_type', 'process')
                    d1['inclusion_body_proc'] = dict(ib_proc)
                    d2['inclusion_body_proc'] = dict(ib_proc)
                d1_id, d2_id = f"{agent_id}_0", f"{agent_id}_1"
                d1['id'], d2['id'] = d1_id, d2_id
            else:
                L = float(agent['length'])
                r = float(agent['radius'])
                L_d = L * (half_mass / m)
                gd_cfg_1 = self._mutate_daughter_gd_config(rate, thresh, rng)
                gd_cfg_2 = self._mutate_daughter_gd_config(rate, thresh, rng)
                d1_gd = dict(agent.get('grow_divide', {})); d1_gd.pop('instance', None); d1_gd.setdefault('_type', 'process'); d1_gd['config'] = gd_cfg_1
                d2_gd = dict(agent.get('grow_divide', {})); d2_gd.pop('instance', None); d2_gd.setdefault('_type', 'process'); d2_gd['config'] = gd_cfg_2
                d1, d2, d1_id, d2_id = _build_segment_daughters(
                    agent, agent_id, angle, r,
                    L_d, L_d, half_mass, half_mass,
                    d1_gd, d2_gd,
                )

            update.setdefault('_add', {})
            update['_add'][d1_id] = d1
            update['_add'][d2_id] = d2
            update.setdefault('_remove', [])
            update['_remove'].append(agent_id)

        return {'agents': update}


# =====================================================================
# AdderGrowDivide — length-based adder model
# =====================================================================

class AdderGrowDivide(Process):
    """Growth/division using the adder model (Taheri-Araghi et al. 2015).

    Each cell:
      * has its own per-cell α (h⁻¹) and Δ (added length, µm)
      * grows exponentially in length: L(t+dt) = L(t) · exp(α · dt)
      * divides when L >= L_birth + Δ
      * daughters get length (L_div / 2) · LogNormal(1, division_noise_cv)
        and fresh α, Δ samples
    """
    config_schema = {
        'agents_key':        {'_type': 'string',  '_default': 'agents'},
        # 40 min doubling: α = ln(2) / (2/3 h) = 1.5 · ln(2) ≈ 1.04 /h.
        # Modulations only slow growth from this baseline.
        'alpha_mean_per_h':  {'_type': 'float',   '_default': 1.04},
        'alpha_cv':          {'_type': 'float',   '_default': 0.18},
        'delta_mean':        {'_type': 'float',   '_default': 2.6},
        'delta_cv':          {'_type': 'float',   '_default': 0.20},
        'division_noise_cv': {'_type': 'float',   '_default': 0.07},
        'pressure_k':        {'_type': 'float',   '_default': 5.0},
        'nutrient_key':      {'_type': 'string',  '_default': ''},
        'nutrient_km':       {'_type': 'float',   '_default': 0.5},
        'nutrient_yield':    {'_type': 'float',   '_default': 1.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._initialized = False
        self._birth_length = 0.0
        self._alpha_per_s = 0.0
        self._delta = 0.0

    @staticmethod
    def _sample_lognormal(mean, cv, rng):
        if mean <= 0.0 or cv <= 0.0:
            return float(mean)
        sigma_sq = math.log(1.0 + cv * cv)
        mu = math.log(mean) - 0.5 * sigma_sq
        return math.exp(rng.gauss(mu, math.sqrt(sigma_sq)))

    @staticmethod
    def _sample_normal_pos(mean, cv, rng):
        if mean <= 0.0 or cv <= 0.0:
            return float(mean)
        return max(1e-6, rng.gauss(mean, mean * cv))

    def _ensure_initialized(self, agent, rng):
        if self._initialized:
            return
        L_now = float(agent.get('length', 0.0) or 0.0)
        self._birth_length = L_now if L_now > 0.0 else float(self.config['delta_mean'])
        alpha_h = self._sample_lognormal(
            float(self.config['alpha_mean_per_h']),
            float(self.config['alpha_cv']), rng,
        )
        self._alpha_per_s = alpha_h / 3600.0
        self._delta = self._sample_normal_pos(
            float(self.config['delta_mean']),
            float(self.config['delta_cv']), rng,
        )
        self._initialized = True

    def _daughter_config(self):
        """Slim config dict for daughter cells."""
        return {
            'agents_key': self.config['agents_key'],
            'alpha_mean_per_h': float(self.config['alpha_mean_per_h']),
            'alpha_cv': float(self.config['alpha_cv']),
            'delta_mean': float(self.config['delta_mean']),
            'delta_cv': float(self.config['delta_cv']),
            'division_noise_cv': float(self.config['division_noise_cv']),
            'pressure_k': float(self.config.get('pressure_k', 5.0)),
            'nutrient_key': self.config.get('nutrient_key', '') or '',
            'nutrient_km': float(self.config.get('nutrient_km', 0.5)),
            'nutrient_yield': float(self.config.get('nutrient_yield', 1.0)),
        }

    def inputs(self):
        return {'agent_id': 'string', 'agents': 'map[pymunk_agent]'}

    def outputs(self):
        return {'agents': 'map[pymunk_agent]'}

    def update(self, state, interval):
        rng = random
        agent_id = state['agent_id']
        agent = state['agents'].get(agent_id, {})
        if not agent:
            return {'agents': {}}

        self._ensure_initialized(agent, rng)

        L_old = float(agent.get('length', 0.0) or 0.0)
        m_old = float(agent.get('mass', 0.0) or 0.0)
        if L_old <= 0.0 or m_old <= 0.0:
            return {'agents': {}}

        alpha = float(self._alpha_per_s)
        pressure = float(agent.get('pressure', 0.0) or 0.0)
        pressure_k = float(self.config.get('pressure_k', 5.0))
        if pressure > 0.0 and pressure_k > 0.0:
            alpha *= math.exp(-pressure / pressure_k)

        nutrient_key = self.config.get('nutrient_key', '') or ''
        if nutrient_key:
            local_dict = agent.get('local') or {}
            local_S = float(local_dict.get(nutrient_key, 0.0) or 0.0)
            km = float(self.config.get('nutrient_km', 0.5))
            if local_S <= 0.0:
                alpha = 0.0
            else:
                alpha *= local_S / (km + local_S)

        dt = float(interval)
        L_new = L_old * math.exp(alpha * dt)
        m_new = m_old * (L_new / L_old)
        dm = m_new - m_old

        update = {agent_id: {'type': 'segment', 'length': L_new - L_old, 'mass': dm}}

        if nutrient_key and dm > 0.0:
            yield_coef = float(self.config.get('nutrient_yield', 1.0)) or 1.0
            update[agent_id]['exchange'] = {nutrient_key: -dm / yield_coef}

        if L_new < self._birth_length + self._delta:
            return {'agents': update}

        # ---- division ----
        noise_cv = float(self.config.get('division_noise_cv', 0.07))
        f1 = 0.5 * self._sample_lognormal(1.0, noise_cv, rng)
        f1 = max(0.05, min(0.95, f1))
        L_d1, L_d2 = L_new * f1, L_new * (1.0 - f1)
        m_d1, m_d2 = m_new * f1, m_new * (1.0 - f1)

        r = float(agent.get('radius', 0.0))
        angle = float(agent.get('angle', 0.0))

        slim = self._daughter_config()
        d1_gd = dict(agent.get('grow_divide', {}))
        d1_gd.pop('instance', None)
        d1_gd.setdefault('_type', 'process')
        d1_gd['config'] = slim
        d2_gd = dict(agent.get('grow_divide', {}))
        d2_gd.pop('instance', None)
        d2_gd.setdefault('_type', 'process')
        d2_gd['config'] = dict(slim)

        d1, d2, d1_id, d2_id = _build_segment_daughters(
            agent, agent_id, angle, r,
            L_d1, L_d2, m_d1, m_d2,
            d1_gd, d2_gd,
        )

        update.setdefault('_add', {})
        update['_add'][d1_id] = d1
        update['_add'][d2_id] = d2
        update.setdefault('_remove', [])
        update['_remove'].append(agent_id)

        return {'agents': update}
