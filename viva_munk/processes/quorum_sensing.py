"""
Quorum sensing via a diffusible autoinducer (AI) field.

Each cell reads its local AI concentration (sampled into ``cell['local']``
by CellFieldExchange), computes a Hill activation

    s = c^n / (K^n + c^n)

and deposits a net amount into the shared field through
``cell['exchange']``:

    production = rate · dt · V_bin              (forward-Euler, always stable)
    degradation = -c · (1 - exp(-k_deg · dt)) · V_bin   (analytic — never
                                                         overshoots into
                                                         negative c, even
                                                         when k_deg · dt ≫ 1)
    d_amt = production + degradation

``rate = basal + (max - basal) · s`` (nM/s — concentration rate in the
cell's own bin), and the downstream CellFieldExchange divides d_amt by
V_bin to convert to a concentration delta. Passing ``bin_volume`` in
explicitly keeps the rate user-facing in nM/s regardless of grid size.
"""
import math

from process_bigraph import Process


class QuorumSensing(Process):
    config_schema = {
        'agents_key':       {'_type': 'string', '_default': 'cells'},
        'ai_key':           {'_type': 'string', '_default': 'ai'},
        # Hill response of the receiver: s = c^n / (K^n + c^n).
        'hill_k':           {'_type': 'float', '_default': 1.0},
        'hill_n':           {'_type': 'float', '_default': 2.0},
        # Secretion (concentration rate in the cell's own bin, units of
        # AI-per-second). basal is the uninduced rate, max is the
        # fully-induced rate; effective rate is a Hill interpolation.
        'basal_secretion':  {'_type': 'float', '_default': 0.05},
        'max_secretion':    {'_type': 'float', '_default': 2.0},
        # First-order cell-mediated degradation of AI (e.g. AHL lactonases).
        # Applied analytically as c·(1 - exp(-k_deg·dt)) so any k_deg and
        # any interval stay stable — no oscillations even when k_deg·dt ≫ 1.
        'degradation_rate': {'_type': 'float', '_default': 0.0},
        # Volume of the field bin the cell lives in (µm³). Used to convert
        # per-cell concentration rates into the amounts that
        # CellFieldExchange deposits into the field. Keep this in sync
        # with the document's (xmax/nx)·(ymax/ny)·depth.
        'bin_volume':       {'_type': 'float', '_default': 1.0},
    }

    def inputs(self):
        return {'agent_id': 'string', 'agents': 'map[pymunk_agent]'}

    def outputs(self):
        return {'agents': 'map[pymunk_agent]'}

    def update(self, state, interval):
        agent_id = state['agent_id']
        agent = state['agents'].get(agent_id, {})
        if not agent:
            return {'agents': {}}

        dt = float(interval)
        ai_key = self.config['ai_key']
        local = agent.get('local') or {}
        c = float(local.get(ai_key, 0.0) or 0.0)

        K = float(self.config['hill_k'])
        n = float(self.config['hill_n'])
        cn = max(c, 0.0) ** n
        Kn = K ** n
        s = cn / (Kn + cn) if (Kn + cn) > 0.0 else 0.0

        basal = float(self.config['basal_secretion'])
        smax = float(self.config['max_secretion'])
        rate = basal + (smax - basal) * s
        k_deg = float(self.config.get('degradation_rate', 0.0))
        vol = float(self.config.get('bin_volume', 1.0))

        # Amounts deposited into the field this tick (nM·µm³).
        production_amt = rate * dt * vol
        # Analytic first-order decay of the cell's own bin: the
        # concentration that degrades in dt is c·(1 - exp(-k_deg·dt)),
        # equivalent in amount to c·(1 - exp(-k_deg·dt))·V_bin. Using
        # the analytic form makes the step stable for any k_deg·dt —
        # the forward-Euler form -k_deg·c·dt overshoots into negative
        # concentrations when k_deg·dt > 1 and produces oscillations.
        if k_deg > 0.0:
            degradation_amt = c * (1.0 - math.exp(-k_deg * dt)) * vol
        else:
            degradation_amt = 0.0
        d_amt = production_amt - degradation_amt

        return {'agents': {agent_id: {
            'type': agent.get('type', 'circle'),
            'qs_state': s,
            # exchange is merged additively across producers and reset by
            # CellFieldExchange after it deposits into the field bin.
            'exchange': {ai_key: d_amt},
        }}}


def make_quorum_sensing_process(
    agents_key='cells', ai_key='ai', interval=30.0, config=None,
):
    cfg = {'agents_key': agents_key, 'ai_key': ai_key}
    if config:
        cfg.update(config)
    return {
        '_type': 'process',
        'address': 'local:QuorumSensing',
        'config': cfg,
        'interval': float(interval),
        'inputs': {
            'agent_id': ['id'],
            'agents': ['..', '..', agents_key],
        },
        'outputs': {
            'agents': ['..', '..', agents_key],
        },
    }


def add_quorum_sensing_to_agents(
    initial_state, agents_key='cells', ai_key='ai',
    interval=30.0, config=None,
):
    agents = initial_state.get(agents_key, {})
    for agent_id, agent in agents.items():
        agent['quorum_sensing'] = make_quorum_sensing_process(
            agents_key=agents_key,
            ai_key=ai_key,
            interval=interval,
            config=config,
        )
    return initial_state
