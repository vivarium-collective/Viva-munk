"""
TODO: use an actual grow/divide process for demo
"""
import random
import math

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
            'agents': 'map[any]'  # make this map[enum[agent types]
        }

    def outputs(self):
        return {
            'agents': 'map[any]' # make this map[enum[agent types]
        }

    def update(self, inputs, interval):
        self.update_bodies(inputs['agents'])

        n_steps = int(self.config.get('substeps', 100))
        dt = interval / n_steps

        # per-step damping from per-second spec: d_step = dps ** dt
        d_step = self.damping_per_second ** dt

        for _ in range(n_steps):
            self.space.damping = d_step
            for body in self.space.bodies:
                self.apply_jitter_force(body, dt)  # pass dt
            self.space.step(dt)

        update = {
            'agents': self.capture_state()
        }
        #
        # for body in self.space.bodies:
        #     print("Body ID:", id(body))
        #     print("Position:", body.position)
        #     print("Velocity:", body.velocity)
        #     print("Mass:", body.mass)
        #     print("Angle:", body.angle)
        #
        # print("\n")
        # for shape in self.space.shapes:
        #     if shape.body.body_type == pymunk.Body.STATIC:
        #         continue
        #
        #     print("Shape Type:", type(shape))
        #     print("Body Position:", shape.body.position)  # Position is stored in the body, not the shape
        #     if isinstance(shape, pymunk.Segment):
        #         print("Segment Start:", shape.a)
        #         print("Segment End:", shape.b)
        #         print("Thickness:", shape.radius)

        # print(inputs['agents'].keys())
        # if len(inputs['agents']) > 1:
        #     x=1
        #     pass

        return update

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

    def capture_state(self):
        state = {}
        for agent_id, obj in self.agents.items():
            if obj['type'] == 'circle':
                state[agent_id] = {
                    'type': obj['type'],
                    'location': (obj['body'].position.x, obj['body'].position.y),
                    'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                    'mass': obj['body'].mass,
                    'inertia': obj['body'].moment,
                    'radius': obj['shape'].radius
                }
            elif obj['type'] == 'segment':
                state[agent_id] = {
                    'type': obj['type'],
                    'location': (obj['body'].position.x, obj['body'].position.y),
                    'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                    'mass': obj['body'].mass,
                    'inertia': obj['body'].moment,
                    'length': obj['length'],
                    'radius': obj['shape'].radius,
                    'angle': obj['body'].angle
                }
        return state
