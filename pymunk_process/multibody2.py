import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio.v2 as imageio
from IPython.display import Image, display
from matplotlib.patches import Circle
import base64
from io import BytesIO
import pymunk

from process_bigraph import Step, Process, Composite, ProcessTypes


# make core
PYMUNK_CORE = ProcessTypes()


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

    def update_bodies(self, agents):
        existing_ids = set(self.agents.keys())
        new_ids = set(agents.keys())

        # Remove objects not in the new state
        for agent_id in existing_ids - new_ids:
            body, shape = self.agents[agent_id]['body'], self.agents[agent_id]['shape']
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

            # make sure to update the length in the agent dictionary
            self.agents[agent_id]['length'] = length

        # make sure to update the agent dictionary
        self.agents[agent_id]['mass'] = mass

        if new_shape:
            self.space.remove(old_shape)  # Remove old shape if necessary

            new_shape.elasticity = attrs.get('elasticity', 0.0)
            new_shape.friction = attrs.get('friction', 0.0)
            self.space.add(new_shape)  # Add new shape to the space
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
            start = pymunk.Vec2d(-length / 2, 0)
            end = pymunk.Vec2d(length / 2, 0)
            inertia = pymunk.moment_for_segment(mass, start, end, radius)  # correct inertia for segments
            body = pymunk.Body(mass, inertia)
            body.position = attrs['location']
            body.angle = angle
            body.length = length
            shape = pymunk.Segment(body, start, end, radius)

        shape.elasticity = attrs.get('elasticity', self.config['elasticity'])
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

    def update(self, inputs, interval):
        self.update_bodies(inputs['agents'])
        self.space.step(interval)
        update = {
            'agents': self.capture_state()
        }
        return update

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


# register types
circle_agent_type = {
    'type': 'string',  # TODO this should be 'enum[circle, segment]'
    'mass': 'float',
    'radius': 'float',
    'inertia': {'_type': 'float',
                '_default': float('inf')
                },
    'location': ('float', 'float'),
    'velocity': ('float', 'float'),
    'elasticity': 'float'
}
segment_agent_type = {
    '_inherit': 'circle_agent',
    'length': 'float',
    'angle': 'float',
}

PYMUNK_CORE.register('circle_agent', circle_agent_type)
PYMUNK_CORE.register('segment_agent', segment_agent_type)

# register process
PYMUNK_CORE.process_registry.register('multibody', PymunkProcess)


def run_simulation(initial_state, config, interval, steps):
    process = PymunkProcess(config, core=PYMUNK_CORE)
    state = initial_state

    timeline = []
    for step in range(steps):
        new_state = process.update(state, interval)
        timeline.append({
            'time': step * interval,
            **new_state
        })

        # update the state
        state = new_state

    return timeline


def growth_division_simulation(initial_state, config, interval, steps, growth_rate, division_threshold):
    process = PymunkProcess(config)
    state = dict(initial_state)  # Deep copy if nested dictionaries or complex objects in state

    timeline = []
    for step in range(steps):
        next_state = {'agents': {}}
        for agent_id, agent_properties in state['agents'].items():
            # Apply growth to mass and geometrical properties
            new_mass = agent_properties['mass'] * (1 + growth_rate)
            agent_properties['mass'] = new_mass

            if agent_properties['type'] == 'circle':
                new_radius = agent_properties['radius'] * (1 + growth_rate)
                agent_properties['radius'] = new_radius
            elif agent_properties['type'] == 'segment':
                new_length = agent_properties['length'] * (1 + growth_rate)
                agent_properties['length'] = new_length

            # Check for division
            if new_mass > division_threshold:
                # Adjust properties for division
                half_mass = new_mass / 2
                for i in [0, 1]:
                    new_agent_id = f"{agent_id}{i}"
                    new_agent_properties = agent_properties.copy()
                    new_agent_properties['mass'] = half_mass
                    if agent_properties['type'] == 'circle':
                        new_agent_properties['radius'] = new_radius / 1.414  # Reduce radius to keep area proportional
                    elif agent_properties['type'] == 'segment':
                        new_agent_properties['length'] = new_length / 2  # Halve the length

                    next_state['agents'][new_agent_id] = new_agent_properties
            else:
                # No division, just update the agent in the next state
                next_state['agents'][agent_id] = agent_properties

        # Simulate dynamics for this step
        new_state = process.update(next_state, interval)

        timeline.append({
            'time': step * interval,
            **new_state
        })

        # Prepare state for the next timestep
        state = new_state

    return timeline


class LineWidthData(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop('linewidth', 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72. / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def simulation_to_gif(data, config, filename='simulation.gif', skip_frames=1):
    if not os.path.exists('frames'):
        os.makedirs('frames')
    env_size = config['env_size']
    barriers = config.get('barriers', [])

    images = []
    for index, step in enumerate(data[::skip_frames]):
        fig, ax = plt.subplots()
        ax.set_xlim(0, env_size)
        ax.set_ylim(0, env_size)
        ax.set_aspect('equal')

        # Draw barriers
        for barrier in barriers:
            start_x, start_y = barrier['start']
            end_x, end_y = barrier['end']
            thickness = barrier.get('thickness', 1)  # Default thickness
            barrier_line = Line2D([start_x, end_x], [start_y, end_y], linewidth=thickness, color='gray')
            ax.add_line(barrier_line)

        # Draw agents
        for agent_id, obj in step['agents'].items():
            if obj.get('type') == 'circle':
                circle = Circle((obj['location'][0], obj['location'][1]),
                                obj['radius'],
                                fill=True)
                ax.add_patch(circle)
            elif obj.get('type') == 'segment':
                length = obj['length']
                thickness = obj['radius'] #* 2  # Visual thickness of the line
                angle = obj['angle']

                dx = np.cos(angle) * length / 2
                dy = np.sin(angle) * length / 2
                start_point = (obj['location'][0] - dx, obj['location'][1] - dy)
                end_point = (obj['location'][0] + dx, obj['location'][1] + dy)

                line = Line2D([start_point[0], end_point[0]],
                              [start_point[1], end_point[1]],
                              linewidth=thickness,
                              solid_capstyle='round')
                ax.add_line(line)

        ax.set_title(f"Time = {step['time']:.1f}")
        frame_filename = f'frames/frame_{index:04d}.png'
        plt.savefig(frame_filename)
        plt.close(fig)
        images.append(imageio.imread(frame_filename))

    imageio.mimsave(filename, images, duration=0.1, loop=0)

    # Clean up the frames directory
    for frame_filename in os.listdir('frames'):
        os.remove(f'frames/{frame_filename}')
    os.rmdir('frames')

    # Output file saved message
    print(f"GIF saved to {filename}")


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


def run_pymunk_experiment():
    initial_state = {
        'agents': {
            '0': {
                'type': 'circle',
                'mass': 10.0,
                'radius': 15,
                'location': (100, 200),
                'velocity': (0, 0),
                'elasticity': 0.0
            },
            '1': {
                'type': 'circle',
                'mass': 1.0,
                'radius': 25,
                'location': (100.5, 250),
                'velocity': (0, 10),
                'elasticity': 0.0
            },
            '2': {
                'type': 'segment',
                'mass': 20.0,
                'length': 50,  # Total length of the segment
                'radius': 10,  # Thickness of the segment
                'angle': 0.785,  # Angle in radians (approximately 45 degrees)
                'location': (300, 500),
                'velocity': (0, 0),
                'elasticity': 0.0
            },
            '3': {
                'type': 'segment',
                'mass': 100.0,
                'length': 80,  # Total length of the segment
                'radius': 20,  # Thickness of the segment
                'angle': -0.785,  # Angle in radians (approximately 45 degrees)
                'location': (400, 400),
                'velocity': (-0.1, 0),
                'elasticity': 0.0
            },
        }
    }

    # run simulation
    interval = 0.1
    steps = 600
    config = {
        'env_size': 600,
        'gravity': -9.81,
        'elasticity': 0.1
    }
    simulation_data2 = run_simulation(initial_state, config, interval, steps)

    # make video
    simulation_to_gif(simulation_data2, config=config, skip_frames=10)


def run_growth_division():
    initial_state = {
        'agents': {
            'X': {
                'type': 'segment',
                'mass': 20.0,
                'length': 50,  # Total length of the segment
                'radius': 10,  # Thickness of the segment
                'angle': 0.785,  # Angle in radians (approximately 45 degrees)
                'location': (200, 200),
                'velocity': (0, 0),
                'elasticity': 0.1
            },
        }
    }

    interval = 1
    steps = 1000
    growth_rate = 0.002
    configgr = {'env_size': 600,
                'gravity': 0}
    simulation_data4 = growth_division_simulation(
        initial_state,
        configgr,
        interval,
        steps,
        growth_rate,
        division_threshold=21
    )

    # make video
    simulation_to_gif(simulation_data4, config=configgr, skip_frames=10, filename='growth_division.gif')


def run_composition(initial_state, config, interval, steps):
    sim_state = {
        'multibody': {
            '_type': 'process',
            'interval': interval,
            'address': 'local:multibody',
            'config': config,
            'inputs': {
                'agents': ['agents'],
            },
            'outputs': {
                'agents': ['agents'],
            }
        },
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'agents': 'any',
                    'time': 'float'
                }
            },
            'inputs': {
                'agents': ['agents'],
                'time': ['global_time']
            },
        }
    }
    sim_state.update(initial_state)

    # make the composite
    sim = Composite(
        {'state': sim_state},
        core=PYMUNK_CORE
    )

    # run the simulation
    total_time = interval * steps
    sim.run(total_time)
    data = sim.gather_results()
    return data


def run_composition_experiment():
    initial_state = {
        'agents': {
            'X': {
                'type': 'segment',
                'mass': 20.0,
                'length': 50,  # Total length of the segment
                'radius': 10,  # Thickness of the segment
                'angle': 0.785,  # Angle in radians (approximately 45 degrees)
                'location': (200, 200),
                'velocity': (0, 0),
                'elasticity': 0.1
            },
        }
    }

    interval = 0.1
    steps = 1000
    # growth_rate = 0.002
    config = {'env_size': 600,'gravity': 0}
    run_composition(initial_state, config, interval, steps)



if __name__ == '__main__':
    # run_pymunk_experiment()
    # run_growth_division()
    run_composition_experiment()
