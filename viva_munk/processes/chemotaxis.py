"""
Memory-based run/tumble chemotaxis.

A literal implementation of the rules:

  RUNNING:
    - Cell moves in a straight line at v = run_speed (default 20 µm/s).
    - It compares the concentration sensed NOW vs a smoothed memory of
      the past few seconds.
    - If concentration is INCREASING → run longer (suppress tumbling).
    - If concentration is DECREASING → run shorter (tumble sooner).

  TUMBLING:
    - Cell stops (motile_speed = 0).
    - Duration: fixed tumble_duration (default 0.1 s).
    - New direction: turn by Normal(0°, tumble_angle_sigma) relative to
      current heading (default 68°).

The tumble rate is

    λ(t) = λ_0 * exp(-k * dc/dt_smoothed)

with λ_0 = baseline_tumble_rate (default 1.0 /s) and k = sensitivity
(default 2.0 s/µM). λ is clamped to [tumble_rate_min, tumble_rate_max]
(default [0.1, 5.0] /s) to avoid extremes.

The smoothed derivative is computed via a one-variable exponential
moving average with time constant τ_memory (default 3.0 s):

    c_memory   += dt * (c_now - c_memory) / τ_memory
    dc_dt_smooth = (c_now - c_memory) / τ_memory

PymunkProcess reads each cell's `motile_speed` and `angle` from the
framework state and sets `body.velocity = motile_speed * (cos θ, sin θ)`
each substep, so the cell really moves at the prescribed speed and
the physics engine integrates its position.
"""
import math
import random

from process_bigraph import Process


class Chemotaxis(Process):
    config_schema = {
        'agents_key': {'_type': 'string', '_default': 'cells'},
        # Field key in cell['local'] that holds the attractant concentration.
        'ligand_key': {'_type': 'string', '_default': 'glucose'},
        # Run swimming speed.
        'run_speed': {'_type': 'float', '_default': 20.0},
        # Baseline tumble rate (/s) when dc/dt = 0.
        'baseline_tumble_rate': {'_type': 'float', '_default': 1.0},
        # Chemotactic sensitivity (s/µM). Tumble rate is multiplied by
        # exp(-sensitivity * dc/dt_smooth).
        'sensitivity': {'_type': 'float', '_default': 2.0},
        # Time constant of the smoothed memory used to compute dc/dt (s).
        'tau_memory': {'_type': 'float', '_default': 3.0},
        # Bounds for the (clamped) tumble rate (/s).
        'tumble_rate_min': {'_type': 'float', '_default': 0.1},
        'tumble_rate_max': {'_type': 'float', '_default': 5.0},
        # Tumble duration (s) — the cell stops swimming and reorients.
        'tumble_duration': {'_type': 'float', '_default': 0.1},
        # Std-dev of the per-tumble heading change, in radians (default
        # 68° ≈ 1.187 rad — close to E. coli's measured turn-angle stddev).
        'tumble_angle_sigma': {'_type': 'float', '_default': 1.187},
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
        agents = state.get('agents') or {}
        agent = agents.get(agent_id)
        if not agent:
            return {'agents': {}}

        dt = float(interval)
        local = agent.get('local') or {}
        ligand_key = self.config['ligand_key']
        c_now = float(local.get(ligand_key, 0.0) or 0.0)

        # Smoothed memory: exponential moving average with time constant
        # tau_memory. On the first tick we have no memory yet, so seed it
        # with the current value (gives dc/dt = 0 for the first tick).
        tau = float(self.config['tau_memory'])
        c_memory_in = agent.get('c_memory')
        if c_memory_in is None:
            c_memory = c_now
        else:
            c_memory = float(c_memory_in)
            c_memory += dt * (c_now - c_memory) / max(tau, 1e-9)
        dc_dt_smooth = (c_now - c_memory) / max(tau, 1e-9)

        # Tumble rate: λ = λ₀ * exp(-k * dc/dt_smooth), clamped.
        lam0 = float(self.config['baseline_tumble_rate'])
        k = float(self.config['sensitivity'])
        exponent = max(-20.0, min(20.0, -k * dc_dt_smooth))
        lam = lam0 * math.exp(exponent)
        lam_min = float(self.config['tumble_rate_min'])
        lam_max = float(self.config['tumble_rate_max'])
        lam = max(lam_min, min(lam_max, lam))

        run_speed = float(self.config['run_speed'])
        tumble_duration = float(self.config['tumble_duration'])
        tumble_sigma = float(self.config['tumble_angle_sigma'])

        # Persistent motile state across ticks.
        motile_state = agent.get('motile_state') or 'run'
        tumble_left = float(agent.get('tumble_time_left', 0.0) or 0.0)
        cur_angle = float(agent.get('angle', 0.0) or 0.0)

        # Default outputs (will be overwritten by branches below).
        new_motile_state = motile_state
        new_motile_speed = run_speed
        new_tumble_left = tumble_left
        angle_delta = 0.0

        if motile_state == 'tumble':
            new_tumble_left = tumble_left - dt
            if new_tumble_left <= 0.0:
                # Tumble done — back to running. The new heading was set
                # at the moment we entered tumble; nothing more to do.
                new_motile_state = 'run'
                new_tumble_left = 0.0
                new_motile_speed = run_speed
            else:
                # Still tumbling — stay stopped.
                new_motile_state = 'tumble'
                new_motile_speed = 0.0
        else:
            # Currently running. Decide whether to switch to tumble.
            p_tumble = 1.0 - math.exp(-lam * dt)
            if random.random() < p_tumble:
                # Enter tumble: stop, pick new heading, set timer.
                new_motile_state = 'tumble'
                new_tumble_left = tumble_duration
                new_motile_speed = 0.0
                angle_delta = random.gauss(0.0, tumble_sigma)
            else:
                # Keep running.
                new_motile_state = 'run'
                new_motile_speed = run_speed
                new_tumble_left = 0.0

        update = {
            'type': agent.get('type', 'segment'),
            'motile_speed': new_motile_speed,
            'motile_state': new_motile_state,
            'tumble_time_left': new_tumble_left,
            'c_memory': c_memory,
            'prev_ligand': c_now,
        }
        # Angle is an accumulate field — emit it as a delta on tumble.
        if angle_delta != 0.0:
            update['angle'] = angle_delta

        return {'agents': {agent_id: update}}


def make_chemotaxis_process(
    agents_key='cells', ligand_key='glucose', interval=0.1, config=None,
):
    """Build a per-cell Chemotaxis process spec."""
    cfg = {'agents_key': agents_key, 'ligand_key': ligand_key}
    if config:
        cfg.update(config)
    return {
        '_type': 'process',
        'address': 'local:Chemotaxis',
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


def add_chemotaxis_to_agents(
    initial_state, agents_key='cells', ligand_key='glucose',
    interval=0.1, config=None,
):
    """Attach a Chemotaxis process to every cell in initial_state[agents_key]."""
    agents = initial_state.get(agents_key, {})
    for agent_id, agent in agents.items():
        agent['chemotaxis'] = make_chemotaxis_process(
            agents_key=agents_key,
            ligand_key=ligand_key,
            interval=interval,
            config=config,
        )
    return initial_state
