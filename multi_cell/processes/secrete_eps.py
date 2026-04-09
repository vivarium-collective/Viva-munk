"""
SecreteEPS — a Process that secretes EPS particles as a function of cell size.

Each cell periodically emits small circle particles near its surface.
The secretion rate scales with cell mass.
"""
import math
import random
import uuid

from process_bigraph import Process


class SecreteEPS(Process):
    config_schema = {
        'agents_key': {'_type': 'string', '_default': 'cells'},
        'particles_key': {'_type': 'string', '_default': 'particles'},
        # Secretion rate: particles per second per unit mass
        'secretion_rate': {'_type': 'float', '_default': 0.001},
        # EPS particle properties
        'eps_radius': {'_type': 'float', '_default': 0.15},
        'eps_mass': {'_type': 'float', '_default': 0.01},
        'eps_elasticity': {'_type': 'float', '_default': 0.0},
        'eps_friction': {'_type': 'float', '_default': 0.5},
        # If True, only attached cells secrete EPS
        'requires_attached': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return {
            'agent_id': 'string',
            'agents': 'map[pymunk_agent]',
        }

    def outputs(self):
        return {
            'particles': 'map[pymunk_agent]',
        }

    def update(self, state, interval):
        agent_id = state['agent_id']
        agent = state['agents'].get(agent_id, {})
        if not agent:
            return {'particles': {}}

        # Gate secretion on attachment if configured
        if self.config.get('requires_attached', False):
            if float(agent.get('attached', 0.0) or 0.0) < 0.5:
                return {'particles': {}}

        mass = float(agent.get('mass', 0.0) or 0.0)
        if mass <= 0:
            return {'particles': {}}

        dt = float(interval)
        rate = float(self.config['secretion_rate'])

        # Expected number of particles this step
        expected = rate * mass * dt
        # Poisson draw
        n_particles = 0
        if expected > 0:
            n_particles = self._poisson(expected)

        if n_particles == 0:
            return {'particles': {}}

        # Get cell geometry for placement
        loc = agent.get('location', (0.0, 0.0))
        cx, cy = float(loc[0]), float(loc[1])
        angle = float(agent.get('angle', 0.0) or 0.0)
        cell_type = agent.get('type', 'circle')
        cell_radius = float(agent.get('radius', 0.5) or 0.5)
        cell_length = float(agent.get('length', 0.0) or 0.0)
        eps_r = float(self.config['eps_radius'])

        add = {}
        for _ in range(n_particles):
            # Place particle on the cell surface
            px, py = self._surface_point(
                cx, cy, angle, cell_type, cell_radius, cell_length, eps_r)

            # Small random velocity away from cell center
            dx, dy = px - cx, py - cy
            dist = math.hypot(dx, dy) or 1.0
            speed = 0.01  # tiny nudge outward
            vx, vy = speed * dx / dist, speed * dy / dist

            pid = f'eps_{uuid.uuid4().hex[:6]}'
            add[pid] = {
                'type': 'circle',
                'mass': float(self.config['eps_mass']),
                'radius': eps_r,
                'location': (px, py),
                'velocity': (vx, vy),
                'elasticity': float(self.config['eps_elasticity']),
                'friction': float(self.config['eps_friction']),
            }

        return {'particles': {'_add': add}}

    def _poisson(self, lam):
        """Simple Poisson sample for small lambda."""
        if lam < 30:
            # Knuth algorithm
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while True:
                k += 1
                p *= random.random()
                if p <= L:
                    return k - 1
        else:
            # Normal approximation
            return max(0, int(round(random.gauss(lam, math.sqrt(lam)))))

    def _surface_point(self, cx, cy, angle, cell_type, cell_radius, cell_length, eps_r):
        """Random point on the cell surface, offset by eps_r."""
        offset = cell_radius + eps_r + 0.01  # just outside the capsule

        if cell_type == 'segment' and cell_length > 0:
            # Pick a random point along the capsule length, then offset perpendicular
            t = random.uniform(-0.5, 0.5) * cell_length
            # Point along the axis
            ax = cx + t * math.cos(angle)
            ay = cy + t * math.sin(angle)
            # Random perpendicular direction
            side = random.choice([-1, 1])
            nx = -math.sin(angle) * side
            ny = math.cos(angle) * side
            return (ax + nx * offset, ay + ny * offset)
        else:
            # Circle: random angle
            theta = random.uniform(0, 2 * math.pi)
            return (cx + offset * math.cos(theta), cy + offset * math.sin(theta))


def make_secrete_eps_process(config=None, agents_key='cells', particles_key='particles', interval=30.0):
    """Create a SecreteEPS process spec to embed in an agent's state."""
    config = config or {}
    config.setdefault('agents_key', agents_key)
    config.setdefault('particles_key', particles_key)
    return {
        '_type': 'process',
        'address': 'local:SecreteEPS',
        'config': config,
        'interval': interval,
        'inputs': {
            'agent_id': ['id'],
            'agents': ['..', '..', agents_key],
        },
        'outputs': {
            'particles': ['..', '..', particles_key],
        },
    }


def add_secrete_eps_to_agents(initial_state, agents_key='cells', particles_key='particles', config=None, interval=10.0):
    """Add SecreteEPS process to each agent in the initial state."""
    agents = initial_state.get(agents_key, {})
    for agent_id, agent in agents.items():
        agent['secrete_eps'] = make_secrete_eps_process(
            config=dict(config) if config else None,
            agents_key=agents_key,
            particles_key=particles_key,
            interval=interval,
        )
    return initial_state
