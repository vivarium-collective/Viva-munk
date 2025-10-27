"""
TODO: use an actual grow/divide process for demo
"""
import random
import math
import uuid

import pymunk

from process_bigraph import Process


# def random_body_position(body):
#     ''' Pick a random point along the boundary of the body (rectangle) '''
#     length = body.length
#     width = body.width
#     edge = random.choice(['left', 'right', 'bottom', 'top'])
#
#     if edge == 'left':
#         # Random point along the left vertical edge
#         return (0, random.uniform(0, length))
#     elif edge == 'right':
#         # Random point along the right vertical edge
#         return (width, random.uniform(0, length))
#     elif edge == 'bottom':
#         # Random point along the bottom horizontal edge
#         return (random.uniform(0, width), 0)
#     elif edge == 'top':
#         # Random point along the top horizontal edge
#         return (random.uniform(0, width), length)


def daughter_locations(parent_state):
    p_loc = parent_state['location']
    p_angle = parent_state.get('angle', 0.0)

    if parent_state['type'] == 'segment':
        # offset along the rod axis
        parent_len = parent_state['length']
        pos_ratios = (-0.35, 0.35)
        offsets = [parent_len * r for r in pos_ratios]
        dxs = [o * math.cos(p_angle) for o in offsets]
        dys = [o * math.sin(p_angle) for o in offsets]
        return [[p_loc[0] + dxs[0], p_loc[1] + dys[0]],
                [p_loc[0] + dxs[1], p_loc[1] + dys[1]]]

    # circle: offset along an arbitrary axis (use angle to keep consistent orientation over time)
    r = parent_state['radius']
    d = max(2*r*0.8, 1.0)  # try to separate daughters ~0.8 diameters
    dx = (d/2) * math.cos(p_angle)
    dy = (d/2) * math.sin(p_angle)
    return [[p_loc[0] - dx, p_loc[1] - dy],
            [p_loc[0] + dx, p_loc[1] + dy]]

def local_impulse_point_for_shape(shape):
    """Return a random local point on the shape boundary in the body's local coords."""
    if isinstance(shape, pymunk.Circle):
        r = shape.radius
        theta = random.uniform(0, 2 * math.pi)
        return (r * math.cos(theta), r * math.sin(theta))

    if isinstance(shape, pymunk.Segment):
        # segment is from a->b in local coords; choose a random point along it, add small normal offset within radius
        t = random.random()
        ax, ay = shape.a
        bx, by = shape.b
        px, py = ax + t * (bx - ax), ay + t * (by - ay)
        # slight offset toward the “edge” to avoid perfectly central impulses
        nx, ny = -(by - ay), (bx - ax)  # unnormalized normal
        nlen = math.hypot(nx, ny) or 1.0
        nx, ny = nx / nlen, ny / nlen
        return (px + nx * shape.radius, py + ny * shape.radius)

    # fallback: body center
    return (0.0, 0.0)


class PymunkProcess(Process):
    config_schema = {
        'env_size': {'_type': 'float', '_default': 500},
        'substeps': {'_type': 'integer', '_default': 100},
        'damping_per_second': {'_type': 'float', '_default': 0.98},
        'gravity': {'_type': 'float', '_default': -9.81},
        'friction': {'_type': 'float', '_default': 0.8},
        'elasticity': {'_type': 'float', '_default': 0.0},
        'jitter_per_second': {'_type': 'float', '_default': 1e-2},  # (impulse std)
        'barriers': 'list[map]',
        'wall_thickness': {'_type': 'float', '_default': 100},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        # create the environment
        self.space = pymunk.Space()
        self.space.gravity = (0, self.config['gravity'])  # Gravity set for the space
        self.substeps = int(self.config.get('substeps', 100))
        self.damping_per_second = float(self.config.get('damping_per_second', 0.98))
        self.jitter_per_second = float(self.config.get('jitter_per_second', 1e-2))

        self.agents = {}

        # add walls
        env_size = self.config['env_size']
        boundary_thickness = self.config.get('wall_thickness', 100)
        walls = [
            pymunk.Segment(self.space.static_body,
                           (-boundary_thickness, -boundary_thickness),
                           (env_size + boundary_thickness, -boundary_thickness),
                           boundary_thickness
                           ),
            pymunk.Segment(self.space.static_body,
                           (env_size + boundary_thickness, -boundary_thickness),
                           (env_size + boundary_thickness, env_size + boundary_thickness),
                           boundary_thickness
                           ),
            pymunk.Segment(self.space.static_body,
                           (env_size + boundary_thickness, env_size + boundary_thickness),
                           (-boundary_thickness, env_size + boundary_thickness),
                           boundary_thickness
                           ),
            pymunk.Segment(self.space.static_body,
                           (-boundary_thickness, env_size + boundary_thickness),
                           (-boundary_thickness, -boundary_thickness),
                           boundary_thickness
                           )
        ]
        for wall in walls:
            wall.elasticity = self.config['elasticity']
            wall.friction = self.config['friction']
            self.space.add(wall)

        # add custom barriers
        for barrier in self.config.get('barriers', []):
            self.add_barrier(barrier)

    def add_barrier(self, barrier):
        start_x, start_y = barrier['start']
        end_x, end_y = barrier['end']
        start = pymunk.Vec2d(start_x, start_y)
        end = pymunk.Vec2d(end_x, end_y)
        thickness = barrier.get('thickness', 1)
        segment = pymunk.Segment(self.space.static_body, start, end, thickness)
        segment.elasticity = barrier.get('elasticity', 1.0)
        segment.friction = barrier.get('friction', 0.5)
        self.space.add(segment)

    def inputs(self):
        return {
            'agents': 'map[pymunk_agent]',     # make this map[enum[agent types]
            'particles': 'map[pymunk_agent]',  # these are simple particles, e.g. circles
        }

    def outputs(self):
        return {
            'agents': 'map[pymunk_agent]',
            'particles': 'map[pymunk_agent]', # these are simple particles, e.g. circles
        }

    def update(self, inputs, interval):
        """
        Efficiently sync agents + particles without allocating a combined dict,
        step the physics, then split outputs in one pass.
        """
        agents_in = inputs.get('agents', {}) or {}
        particles_in = inputs.get('particles', {}) or {}

        # sanity: IDs must be unique across ports this tick
        overlap = set(agents_in) & set(particles_in)
        if overlap:
            raise ValueError(
                f"PymunkProcess.update(): ID(s) present in both agents and particles: {sorted(overlap)}"
            )

        agent_ids = set(agents_in)
        particle_ids = set(particles_in)

        # ------- sync bodies (no combined dict) -------
        # local helper keeps all changes inside update()
        def _sync_bodies_dual(agents_dict, particles_dict):
            existing_ids = set(self.agents)
            new_ids = set(agents_dict) | set(particles_dict)

            # remove stale
            for dead_id in existing_ids - new_ids:
                body = self.agents[dead_id]['body']
                shape = self.agents[dead_id]['shape']
                self.space.remove(body, shape)
                del self.agents[dead_id]

            # add/update both maps (order doesn't matter)
            for d in (agents_dict, particles_dict):
                for _id, attrs in d.items():
                    self.manage_object(_id, attrs)

        _sync_bodies_dual(agents_in, particles_in)

        # ------- integrate -------
        n_steps = max(1, int(self.config.get('substeps', 100)))
        dt = float(interval) / n_steps
        d_step = self.damping_per_second ** max(dt, 0.0)

        for _ in range(n_steps):
            self.space.damping = d_step
            for body in self.space.bodies:
                self.apply_jitter_force(body, dt)
            self.space.step(dt)

        # ------- emit in one pass (no full_state build) -------
        agents_out = {}
        particles_out = {}

        for _id, obj in self.agents.items():
            # only report objects that were present on this tick's inputs
            # (keeps behavior consistent if you treat absence as deletion)
            if _id in agent_ids or _id in particle_ids:
                if obj['type'] == 'circle':
                    rec = {
                        'type': obj['type'],
                        'location': (obj['body'].position.x, obj['body'].position.y),
                        'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                        'inertia': obj['body'].moment,
                    }
                else:  # 'segment'
                    rec = {
                        'type': obj['type'],
                        'location': (obj['body'].position.x, obj['body'].position.y),
                        'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                        'inertia': obj['body'].moment,
                        'angle': obj['body'].angle,
                    }

                if _id in agent_ids:
                    agents_out[_id] = rec
                elif _id in particle_ids:
                    particles_out[_id] = rec

        return {'agents': agents_out, 'particles': particles_out}

    def apply_jitter_force(self, body, dt):
        shape = next(iter(body.shapes)) if body.shapes else None
        local_point = local_impulse_point_for_shape(shape) if shape else (0.0, 0.0)

        # Scale per-second noise to the step: σ_step ≈ σ_sec * sqrt(dt)
        sigma = self.jitter_per_second * math.sqrt(max(dt, 1e-12))
        fx = random.normalvariate(0.0, sigma)
        fy = random.normalvariate(0.0, sigma)
        body.apply_impulse_at_local_point((fx, fy), local_point)

    def update_bodies(self, agents):
        existing_ids = set(self.agents.keys())
        new_ids = set(agents.keys())

        # Remove objects not in the new state
        for agent_id in existing_ids - new_ids:
            body = self.agents[agent_id]['body']
            shape = self.agents[agent_id]['shape']
            self.space.remove(body, shape)
            del self.agents[agent_id]

        # Add or update existing objects
        for agent_id, attrs in agents.items():
            self.manage_object(agent_id, attrs)

    def manage_object(self, agent_id, attrs):
        agent = self.agents.get(agent_id)
        if not agent:
            self.create_new_object(agent_id, attrs)
            return

        body = agent['body']
        old_shape = agent['shape']
        old_type = agent['type']

        # robust defaults
        shape_type = attrs.get('type', old_type or 'circle')
        mass = float(attrs.get('mass', body.mass))
        vx, vy = attrs.get('velocity', (0.0, 0.0))
        body.mass = mass
        body.position = pymunk.Vec2d(*attrs.get('location', (body.position.x, body.position.y)))
        body.velocity = pymunk.Vec2d(vx, vy)

        needs_rebuild = False
        if shape_type == 'circle':
            radius = float(attrs['radius'])
            if not isinstance(old_shape, pymunk.Circle) or abs(
                    old_shape.radius - radius) > 1e-9 or old_type != 'circle':
                needs_rebuild = True
            if needs_rebuild:
                new_shape = pymunk.Circle(body, radius)
                body.moment = pymunk.moment_for_circle(mass, 0, radius)
        elif shape_type == 'segment':
            length = float(attrs['length'])
            radius = float(attrs['radius'])
            angle = float(attrs['angle'])
            # local endpoints
            start = pymunk.Vec2d(-length / 2, 0).rotated(angle)
            end = pymunk.Vec2d(length / 2, 0).rotated(angle)
            if not isinstance(old_shape, pymunk.Segment) or abs(
                    old_shape.radius - radius) > 1e-9 or old_type != 'segment':
                needs_rebuild = True
            if needs_rebuild:
                new_shape = pymunk.Segment(body, start, end, radius)
                body.moment = pymunk.moment_for_segment(mass, start, end, radius)
            body.angle = angle
            body.length = length
            body.width = radius * 2
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        # swap shape if needed
        if needs_rebuild:
            # preserve material params
            elasticity = attrs.get('elasticity', self.config['elasticity'])
            friction = attrs.get('friction', self.config['friction'])
            new_shape.elasticity = elasticity
            new_shape.friction = friction

            self.space.remove(old_shape)
            self.space.add(new_shape)
            agent['shape'] = new_shape

        # keep dict in sync
        agent['type'] = shape_type
        agent['mass'] = mass
        if shape_type == 'circle':
            agent['radius'] = radius
            agent['angle'] = None
            agent['length'] = None
        else:
            agent['radius'] = radius
            agent['angle'] = body.angle
            agent['length'] = body.length

    def create_new_object(self, agent_id, attrs):
        shape_type = attrs.get('type', 'circle')
        mass = float(attrs.get('mass', 1.0))
        vx, vy = attrs.get('velocity', (0.0, 0.0))
        pos = attrs.get('location', (0.0, 0.0))

        if shape_type == 'circle':
            radius = float(attrs['radius'])
            inertia = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, inertia)
            body.position = pos
            body.velocity = (vx, vy)
            shape = pymunk.Circle(body, radius)
            angle = None
            length = None
        elif shape_type == 'segment':
            length = float(attrs['length'])
            radius = float(attrs['radius'])
            angle = float(attrs['angle'])
            start = (-length / 2, 0)
            end = (length / 2, 0)
            inertia = pymunk.moment_for_segment(mass, start, end, radius)
            body = pymunk.Body(mass, inertia)
            body.position = pos
            body.velocity = (vx, vy)
            body.angle = angle
            body.length = length
            body.width = radius * 2
            shape = pymunk.Segment(body, start, end, radius)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        shape.elasticity = attrs.get('elasticity', self.config['elasticity'])
        shape.friction = attrs.get('friction', self.config['friction'])

        self.space.add(body, shape)
        self.agents[agent_id] = {
            'body': body,
            'shape': shape,
            'type': shape_type,
            'mass': mass,
            'radius': radius,
            'angle': angle,
            'length': length,
        }

    def get_state_update(self):
        state = {}
        for agent_id, obj in self.agents.items():
            if obj['type'] == 'circle':

                state[agent_id] = {
                    'type': obj['type'],
                    'location': (obj['body'].position.x, obj['body'].position.y),
                    'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                    # 'mass': obj['body'].mass,
                    'inertia': obj['body'].moment,
                    # 'radius': obj['shape'].radius
                }
            elif obj['type'] == 'segment':
                state[agent_id] = {
                    'type': obj['type'],
                    'location': (obj['body'].position.x, obj['body'].position.y),
                    'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                    # 'mass': obj['body'].mass,
                    'inertia': obj['body'].moment,
                    # 'length': obj['length'],
                    # 'radius': obj['shape'].radius,
                    'angle': obj['body'].angle
                }
        return state


def make_initial_state(
    n_microbes=2,
    n_particles=2,
    env_size=600.0,
    *,
    agents_key='cells',
    particles_key='particles',
    seed=None,
    elasticity=0.0,
    # circle knobs
    particle_radius_range=(1.0, 10.0),
    particle_mass_density=0.015,   # mass ≈ density * π r²
    particle_speed_range=(0.0, 10.0),
    # segment knobs
    microbe_length_range=(40.0, 120.0),
    microbe_radius_range=(6.0, 24.0),
    microbe_mass_density=0.02,     # mass ≈ density * length * (2r)
    microbe_speed_range=(0.0, 0.4),
    # placement & overlap
    margin=5.0,
    avoid_overlap_circles=True,
    min_gap=2.0,
    max_tries_per_circle=200,
):
    rng = random.Random(seed)
    agents, particles = {}, {}

    # --- helpers ---
    def random_id(prefix):
        return f"{prefix}_{uuid.uuid4().hex[:6]}"

    def rand_circle():
        r = rng.uniform(*particle_radius_range)
        x = rng.uniform(margin + r, env_size - (margin + r))
        y = rng.uniform(margin + r, env_size - (margin + r))
        speed = rng.uniform(*particle_speed_range)
        theta = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
        mass = particle_mass_density * math.pi * (r ** 2)
        return {
            'type': 'circle',
            'mass': mass,
            'radius': r,
            'location': (x, y),
            'velocity': (vx, vy),
            'elasticity': elasticity,
        }

    def circles_overlap(c1, c2):
        (x1, y1), r1 = c1['location'], c1['radius']
        (x2, y2), r2 = c2['location'], c2['radius']
        dx, dy = x1 - x2, y1 - y2
        return (dx*dx + dy*dy) < (r1 + r2 + min_gap) ** 2

    def place_circles(n):
        placed = []
        for _ in range(n):
            if avoid_overlap_circles:
                for _try in range(max_tries_per_circle):
                    cand = rand_circle()
                    if all(not circles_overlap(cand, prev) for prev in placed):
                        placed.append(cand)
                        break
                else:
                    placed.append(rand_circle())
            else:
                placed.append(rand_circle())
        return placed

    def rand_microbe(agent_id):
        L = rng.uniform(*microbe_length_range)
        rad = rng.uniform(*microbe_radius_range)
        ang = rng.uniform(-math.pi, math.pi)
        dx, dy = (L / 2) * math.cos(ang), (L / 2) * math.sin(ang)
        pad = rad + margin
        x = rng.uniform(pad + abs(dx), env_size - (pad + abs(dx)))
        y = rng.uniform(pad + abs(dy), env_size - (pad + abs(dy)))
        speed = rng.uniform(*microbe_speed_range)
        phi = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(phi), speed * math.sin(phi)
        mass = microbe_mass_density * L * (2 * rad)
        return {
            'id': agent_id,
            'type': 'segment',
            'mass': mass,
            'length': L,
            'radius': rad,
            'angle': ang,
            'location': (x, y),
            'velocity': (vx, vy),
            'elasticity': elasticity,
            'grow_divide': {
                '_type': 'edge',
                'address': 'local:GrowDivide',
                'config': {'rate': 0.05},
                'inputs': {'mass': ['mass']},
                'outputs': {'mass': ['mass']},
            }
        }

    # --- build objects ---
    for c in place_circles(n_particles):
        particles[random_id('p')] = c
    for _ in range(n_microbes):
        agent_id = random_id('a')
        agents[agent_id] = rand_microbe(agent_id=agent_id)

    return {agents_key: agents, particles_key: particles}


def get_mother_machine_config(
    env_size=600,
    spacer_thickness=5,
    channel_height=500,
    channel_space=50
):
    barriers = []
    y_start = 0  # Start at the bottom of the environment
    y_end = channel_height  # Height of the channel

    # Calculate how many barriers can fit within the env_size
    num_channels = int(env_size / (spacer_thickness + channel_space))

    # Generate barriers based on calculated number
    x_position = spacer_thickness + channel_space
    for _ in range(num_channels):
        barrier = {
            'start': (x_position, y_start),
            'end': (x_position, y_end),
            'thickness': spacer_thickness
        }
        barriers.append(barrier)

        # Update x_position for the next barrier, adding the space for the channel
        x_position += spacer_thickness + channel_space

    return {
        'env_size': env_size,
        'barriers': barriers
    }
