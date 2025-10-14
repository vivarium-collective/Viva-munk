"""
TODO: use an actual grow/divide process for demo
"""
import random
import math

import pymunk

from process_bigraph import Process


def random_body_position(body):
    ''' Pick a random point along the boundary of the body (rectangle) '''
    length = body.length
    width = body.width
    edge = random.choice(['left', 'right', 'bottom', 'top'])

    if edge == 'left':
        # Random point along the left vertical edge
        return (0, random.uniform(0, length))
    elif edge == 'right':
        # Random point along the right vertical edge
        return (width, random.uniform(0, length))
    elif edge == 'bottom':
        # Random point along the bottom horizontal edge
        return (random.uniform(0, width), 0)
    elif edge == 'top':
        # Random point along the top horizontal edge
        return (random.uniform(0, width), length)


def daughter_locations(parent_state):
    parent_length = parent_state['length']
    parent_angle = parent_state['angle']
    parent_location = parent_state['location']
    pos_ratios = [-0.35, 0.35]  #[-0.25, 0.25]
    daughter_locations = []
    for daughter in range(2):
        dx = parent_length * pos_ratios[daughter] * math.cos(parent_angle)
        dy = parent_length * pos_ratios[daughter] * math.sin(parent_angle)
        location = [parent_location[0] + dx, parent_location[1] + dy]
        daughter_locations.append(location)
    return daughter_locations


class PymunkProcess(Process):
    config_schema = {
        'env_size': {
            '_type': 'float',
            '_default': 500
        },
        'damping': {
            '_type': 'float',
            '_default': 0.9
        },
        'gravity': {
            '_type': 'float',
            '_default': -9.81
        },
        'friction': {
            '_type': 'float',
            '_default': 0.8
        },
        'elasticity': {
            '_type': 'float',
            '_default': 0
        },
        'jitter_force': {
            '_type': 'float',
            '_default': 1e-2
        },
        'barriers': 'list[map]',
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        # create the environment
        self.space = pymunk.Space()
        self.space.gravity = (0, self.config['gravity'])  # Gravity set for the space
        self.space.damping = self.config['damping']  # Add some damping
        self.agents = {}

        # add walls
        env_size = self.config['env_size']
        boundary_thickness = 100  # Increase thickness if needed
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
            'agents': 'map[circle_agent]'
        }

    def outputs(self):
        return {
            'agents': 'map[circle_agent]'
        }

    def update(self, inputs, interval):
        self.update_bodies(inputs['agents'])

        n_steps = 100
        dt = interval / n_steps
        for _ in range(n_steps):
            # apply forces
            for body in self.space.bodies:
                self.apply_jitter_force(body)
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

    def apply_jitter_force(self, body):
        jitter_location = random_body_position(body)
        jitter_force = [
            random.normalvariate(0, self.config['jitter_force']),
            random.normalvariate(0, self.config['jitter_force'])]
        body.apply_impulse_at_local_point(
            jitter_force,
            jitter_location)

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
        new_shape = None  # Initialize new_shape to None

        # TODO -- make a new body???
        mass = attrs['mass']
        body.mass = mass
        body.position = pymunk.Vec2d(*attrs['location'])
        body.velocity = pymunk.Vec2d(*attrs['velocity'])

        shape_type = attrs.get('type', 'circle')
        if shape_type == 'circle':
            radius = attrs['radius']
            new_shape = pymunk.Circle(body, radius)
            body.moment = pymunk.moment_for_circle(mass, 0, radius)

        elif shape_type == 'segment':
            length = attrs['length']
            radius = attrs['radius']
            angle = attrs['angle']
            start = pymunk.Vec2d(-length / 2, 0).rotated(angle)
            end = pymunk.Vec2d(length / 2, 0).rotated(angle)
            new_shape = pymunk.Segment(body, start, end, radius)
            body.moment = pymunk.moment_for_segment(mass, start, end, radius)
            body.angle = angle
            body.length = length
            body.width = radius * 2

            # make sure to update the length in the agent dictionary
            self.agents[agent_id]['length'] = length

        # make sure to update the agent dictionary
        self.agents[agent_id]['mass'] = mass

        if new_shape:
            self.space.remove(body, old_shape)  # Remove old shape if necessary

            new_shape.elasticity = attrs.get('elasticity', self.config['elasticity'])  # TODO -- this elasticity and friction is the same as the walls...
            new_shape.friction = attrs.get('friction', self.config['friction'])
            self.space.add(body, new_shape)  # Add new shape to the space
            agent['shape'] = new_shape

    def create_new_object(self, agent_id, attrs):
        shape_type = attrs.get('type', 'circle')
        mass = attrs['mass']

        # Initialize body differently based on the type
        if shape_type == 'circle':
            radius = attrs['radius']
            inertia = pymunk.moment_for_circle(mass, 0, radius)  # correct inertia for circles
            body = pymunk.Body(mass, inertia)
            body.position = attrs['location']
            shape = pymunk.Circle(body, radius)
        elif shape_type == 'segment':
            length = attrs['length']
            radius = attrs['radius']  # this is the thickness of the segment
            angle = attrs['angle']
            start = (-length / 2, 0)
            end = (length / 2, 0)
            inertia = pymunk.moment_for_segment(mass, start, end, radius)  # correct inertia for segments
            body = pymunk.Body(mass, inertia)
            body.position = attrs['location']
            body.angle = angle
            body.length = length
            body.width = radius * 2
            shape = pymunk.Segment(body, start, end, radius)

        shape.elasticity = attrs.get('elasticity', self.config['elasticity'])  # TODO -- this elasticity and friction is the same as the walls...
        shape.friction = attrs.get('friction', self.config['friction'])
        self.space.add(body, shape)
        self.agents[agent_id] = {
            'body': body,
            'shape': shape,
            'type': shape_type,
            'mass': mass,
            'radius': radius,
            'angle': angle if shape_type == 'segment' else None,
            'length': length if shape_type == 'segment' else None,
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
