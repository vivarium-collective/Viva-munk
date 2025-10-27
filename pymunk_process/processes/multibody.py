"""
TODO: use an actual grow/divide process for demo
"""
import random
import math
import uuid

import pymunk

from process_bigraph import Process


def daughter_locations(parent_state):
    """
    Compute daughter cell locations based on the parent's position, angle, and geometry.
    For segments: daughters are offset along the long axis.
    For circles: daughters are offset along the parent's angle direction.
    """
    p_loc = parent_state['location']
    p_angle = parent_state.get('angle', 0.0)
    dtype = parent_state.get('type', 'circle')

    # Position ratios: symmetric offsets around the parent center
    pos_ratios = [-0.25, 0.25]
    daughter_locs = []

    if dtype == 'segment':
        parent_length = float(parent_state.get('length', 1.0))
        for ratio in pos_ratios:
            dx = parent_length * ratio * math.cos(p_angle)
            dy = parent_length * ratio * math.sin(p_angle)
            loc = [p_loc[0] + dx, p_loc[1] + dy]
            daughter_locs.append(loc)

    else:  # circle or fallback
        r = float(parent_state.get('radius', 1.0))
        # separate daughters by roughly one diameter (0.8–1.0 is typical)
        d = max(2 * r * 0.8, 1.0)
        for ratio in (-0.5, 0.5):
            dx = d * ratio * math.cos(p_angle)
            dy = d * ratio * math.sin(p_angle)
            loc = [p_loc[0] + dx, p_loc[1] + dy]
            daughter_locs.append(loc)

    return daughter_locs

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

import math
import uuid
import random

# -------------------------
# Utilities
# -------------------------

def make_id(prefix='id', nhex=6):
    return f"{prefix}_{uuid.uuid4().hex[:nhex]}"

def make_rng(seed=None):
    return random.Random(seed)

# -------------------------
# Mass/geometry conversions
# -------------------------

def circle_mass_from_radius(radius, density):
    # m = ρ π r^2
    return float(density) * math.pi * (float(radius) ** 2)

def circle_radius_from_mass(mass, density):
    # r = sqrt(m / (ρ π))
    return math.sqrt(float(mass) / (float(density) * math.pi))

def capsule_mass_from_length_radius(length, radius, density):
    # Approximate capsule as rectangle length L and diameter 2r:
    # m = ρ * (2r) * L  (ignoring hemispherical ends for simplicity; good for L >> r)
    return float(density) * (2.0 * float(radius)) * float(length)

def capsule_length_from_mass(mass, radius, density):
    # L = m / (ρ * 2r)
    return float(mass) / (float(density) * (2.0 * float(radius)))

# -------------------------
# Primitive builders (single objects)
# - Accept explicit geometry/mass; if one is missing, infer from density.
# - Velocity can be set directly, or drawn from (speed_range, heading).
# -------------------------

def build_particle(
    rng,
    env_size,
    *,
    elasticity=0.0,
    id_prefix='p',
    # position
    x=None, y=None, margin=5.0,
    # kinematics
    velocity=None, speed_range=(0.0, 10.0),
    # geometry / mass (circle)
    radius=None, mass=None, density=0.015
):
    # derive geometry/mass if needed
    if radius is None and mass is None:
        radius = rng.uniform(1.0, 10.0)
    if mass is None:
        mass = circle_mass_from_radius(radius, density)
    if radius is None:
        radius = circle_radius_from_mass(mass, density)

    # position
    if x is None or y is None:
        r = radius
        x = rng.uniform(margin + r, env_size - (margin + r))
        y = rng.uniform(margin + r, env_size - (margin + r))

    # velocity
    if velocity is None:
        speed = rng.uniform(*speed_range)
        theta = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
    else:
        vx, vy = velocity

    return make_id(id_prefix), {
        'type': 'circle',
        'mass': float(mass),
        'radius': float(radius),
        'location': (float(x), float(y)),
        'velocity': (float(vx), float(vy)),
        'elasticity': float(elasticity),
    }

def build_microbe(
    rng,
    env_size,
    *,
    agent_id=None,          # <-- optional explicit id
    elasticity=0.0,
    id_prefix='a',
    # placement & bounds
    x=None, y=None, angle=None, margin=5.0,
    # kinematics
    velocity=None, speed_range=(0.0, 0.4),
    # geometry / mass (capsule segment)
    length=None, radius=None, mass=None, density=0.02,
    length_range=(40.0, 120.0), radius_range=(6.0, 24.0),
):
    if length is None and mass is None:
        length = rng.uniform(*length_range)
    if radius is None:
        radius = rng.uniform(*radius_range)

    if mass is None:
        mass = capsule_mass_from_length_radius(length, radius, density)
    if length is None:
        length = capsule_length_from_mass(mass, radius, density)

    if angle is None:
        angle = rng.uniform(-math.pi, math.pi)

    dx, dy = (length / 2.0) * math.cos(angle), (length / 2.0) * math.sin(angle)
    pad = radius + margin
    if x is None or y is None:
        x = rng.uniform(pad + abs(dx), env_size - (pad + abs(dx)))
        y = rng.uniform(pad + abs(dy), env_size - (pad + abs(dy)))

    if velocity is None:
        speed = rng.uniform(*speed_range)
        phi = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(phi), speed * math.sin(phi)
    else:
        vx, vy = velocity

    _id = agent_id or make_id(id_prefix)
    return _id, {
        'id': _id,                 # <-- include id in the object
        'type': 'segment',
        'mass': float(mass),
        'length': float(length),
        'radius': float(radius),
        'angle': float(angle),
        'location': (float(x), float(y)),
        'velocity': (float(vx), float(vy)),
        'elasticity': float(elasticity),
    }

# -------------------------
# Placers (collections)
# -------------------------

def circles_overlap(c1, c2, extra_gap=0.0):
    (x1, y1), r1 = c1['location'], c1['radius']
    (x2, y2), r2 = c2['location'], c2['radius']
    dx, dy = x1 - x2, y1 - y2
    return (dx*dx + dy*dy) < (r1 + r2 + extra_gap) ** 2

def place_circles(
    rng, env_size, n,
    *,
    margin=5.0,
    avoid_overlap=True,
    extra_gap=2.0,
    max_tries=200,
    particle_kwargs=None
):
    particle_kwargs = dict(particle_kwargs or {})
    placed = []
    out = {}
    for _ in range(n):
        if avoid_overlap:
            for _try in range(max_tries):
                pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
                if all(not circles_overlap(cand, prev, extra_gap) for prev in placed):
                    placed.append(cand)
                    out[pid] = cand
                    break
            else:
                pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
                placed.append(cand)
                out[pid] = cand
        else:
            pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
            placed.append(cand)
            out[pid] = cand
    return out

def place_microbes(
    rng, env_size, n,
    *,
    margin=5.0,
    microbe_kwargs=None,
    ids=None,                 # <-- optional list of ids (len == n)
    id_factory=None,          # <-- optional callable -> str
):
    microbe_kwargs = dict(microbe_kwargs or {})
    out = {}
    for i in range(n):
        agent_id = None
        if ids is not None:
            agent_id = ids[i]
        elif id_factory is not None:
            agent_id = id_factory(i)
        aid, obj = build_microbe(rng, env_size, margin=margin, agent_id=agent_id, **microbe_kwargs)
        out[aid] = obj
    return out


# -------------------------
# High-level initializer
# -------------------------

def make_initial_state(
    n_microbes=2,
    n_particles=2,
    env_size=600.0,
    *,
    agents_key='cells',
    particles_key='particles',
    seed=None,
    elasticity=0.0,
    # particle defaults
    particle_radius_range=(1.0, 10.0),
    particle_mass_density=0.015,
    particle_speed_range=(0.0, 10.0),
    # microbe defaults
    microbe_length_range=(40.0, 120.0),
    microbe_radius_range=(6.0, 24.0),
    microbe_mass_density=0.02,
    microbe_speed_range=(0.0, 0.4),
    # placement
    margin=5.0,
    avoid_overlap_circles=True,
    min_gap=2.0,
    max_tries_per_circle=200,
):
    rng = make_rng(seed)

    particles = place_circles(
        rng, env_size, n_particles,
        margin=margin,
        avoid_overlap=avoid_overlap_circles,
        extra_gap=min_gap,
        max_tries=max_tries_per_circle,
        particle_kwargs=dict(
            elasticity=elasticity,
            density=particle_mass_density,
            speed_range=particle_speed_range,
            # you can override radius/mass/velocity here if desired
            # radius=..., mass=..., velocity=(vx, vy), x=..., y=...
        )
    )

    agents = place_microbes(
        rng, env_size, n_microbes,
        margin=margin,
        microbe_kwargs=dict(
            elasticity=elasticity,
            density=microbe_mass_density,
            speed_range=microbe_speed_range,
            length_range=microbe_length_range,
            radius_range=microbe_radius_range,
        ),
        # e.g., fixed IDs:
        # ids=[f"a_seed{i}" for i in range(n_microbes)],
        # or dynamic:
        # id_factory=lambda i: f"a_{uuid.uuid4().hex[:6]}",
    )

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
