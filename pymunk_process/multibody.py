"""
=========================
Multibody physics process
=========================

Simulates collisions between cell bodies with a physics engine.
"""

import pytest
import random
import math
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# process-bigraph imports
from process_bigraph import Process, Composite, ProcessTypes
from pymunk_process.pymunk_minimal import PymunkMinimal as Pymunk
from pymunk_process.units import units, remove_units
from pymunk_process import REGISTER_TYPES
from pymunk_process.plots.snapshots import plot_snapshots, format_snapshot_data
from pymunk_process.plots.snapshots_video import make_video


DEFAULT_LENGTH_UNIT = units.um
DEFAULT_BOUNDS = [200 * DEFAULT_LENGTH_UNIT, 200 * DEFAULT_LENGTH_UNIT]
DEFAULT_MASS_UNIT = units.ng
DEFAULT_TIME_UNIT = units.s
DEFAULT_VELOCITY_UNIT = DEFAULT_LENGTH_UNIT / DEFAULT_TIME_UNIT

# constants
PI = math.pi


# helper functions
def daughter_locations(mother_location, state):
    """Given a mother's location, place daughters close"""
    mother_length = state['length']
    mother_x = mother_location[0]
    mother_y = mother_location[1]
    locations = []
    for daughter in range(2):
        location = [mother_x + random.gauss(0, 0.1) * mother_length,
                    mother_y + random.gauss(0, 0.1) * mother_length]
        locations.append(location)
    return locations


def sphere_volume_from_diameter(diameter):
    """Given a diameter for a sphere, return the volume"""
    radius = diameter / 2
    volume = 4 / 3 * (PI * radius ** 3)
    return volume


def make_random_position(bounds):
    """Return a random position with the [x, y] bounds"""
    return [
        np.random.uniform(0, bound.magnitude) * bound.units
        for bound in bounds]


def add_to_dict(d, added):
    for k, v in added.items():
        if k in d:
            d[k] += v
        else:
            d[k] = v
    return d


def remove_from_dict(d, removed):
    for k, v in removed.items():
        if k in d:
            d[k] -= v
        else:
            d[k] = -v
    return d


class Multibody(Process):
    """Multibody process for tracking cell bodies.

    Simulates collisions between
     cell bodies with a physics engine.

    :term:`Ports`:
    * ``agents``: The store containing all cell sub-compartments. Each cell in
      this store has values for location, diameter, mass.

    Arguments:
        parameters(dict): Accepts the following configuration keys:

        * **jitter** (float): random force applied to agents, in pN.
        * **bounds** (list): size of the environment in the units specified by length_unit, with ``[x, y]``.
        * **length_unit** (Quantity): standard length unit for the physics engine -- this includes the bounds.
        * **mass_unit** (Quantity): standard mass unit for the physics engine.
        * **velocity_unit** (Quantity): standard velocity unit for the physics engine.
        * ***animate*** (:py:class:`bool`): interactive matplotlib option to
          animate multibody. To run with animation turned on set True, and use
          the TKAgg matplotlib backend:

          .. code-block:: console

              $ MPLBACKEND=TKAgg python tumor_tcell/processes/neighbors.py
    """

    config_schema = {
        'jitter_force': {
            '_type': 'float',
            '_default': 1e-6,
        },
        'bounds': 'point2d',
        'length_unit': {
            '_type': 'string',
            '_default': 'um'
        },
        'mass_unit': {'_type': 'string', '_default': 'ng'},
        'time_unit': {'_type': 'string', '_default': 's'},
        'animate': 'boolean',
    }

    def __init__(self, config=None, core=None):
        # This means registration must be idempotent
        # make this operation fast if already registered (!)
        # REGISTER_TYPES(self.core)

        super().__init__(config, core)

        self.length_unit = units(self.config['length_unit'])
        self.mass_unit = units(self.config['mass_unit'])
        self.time_unit = units(self.config['time_unit'])
        self.velocity_unit = self.length_unit / self.time_unit
        self.cell_loc_units = {}

        # make the multibody object
        timestep = self.config.get('interval', 1.0)  # TODO -- how do we get the timestep default?
        pymunk_config = {
            'cell_shape': 'circle',
            'jitter_force': self.config['jitter_force'],
            'bounds': [
                b.to(self.length_unit).magnitude
                for b in config['bounds']],
            'physics_dt': min(timestep / 10, 0.1)}
        self.physics = Pymunk(pymunk_config)

        # interactive plot for visualization
        self.animate = self.config['animate']
        if self.animate:
            plt.ion()
            self.ax = plt.gca()
            self.ax.set_aspect('equal')

    def inputs(self):
        return {
            'agents': 'map[boundary:boundary]'
        }

    def outputs(self):
        return {
            'agents': 'map[boundary:boundary]'
        }

    def update(self, state, interval):
        agents = state.get('agents', {})

        # animate before update
        if self.animate:
            self.animate_frame(agents)

        # update physics with new agents
        munk_agents = self.bodies_remove_units(
            copy.deepcopy(agents))
        self.physics.update_bodies(munk_agents)

        # run simulation
        self.physics.run(interval)

        # get new cell positions and neighbors
        cell_positions = self.physics.get_body_positions()

        # add units to cell_positions
        cell_positions = self.location_add_units(cell_positions)

        update = {
            'agents': {
                cell_id: {
                    'boundary': {
                        'angle': cell_positions[cell_id]['angle'],
                        'location': tuple([
                            new_loc - old_loc
                            for new_loc, old_loc in zip(
                                cell_positions[cell_id]['location'],
                                agents[cell_id]['boundary']['location'])])
                    },
                } for cell_id in agents.keys()
            }
        }
        return update

    def bodies_remove_units(self, bodies):
        """Convert units to the standard, and then remove them

        This is required for interfacing the physics engine, which does not track units
        """
        for bodies_id, specs in bodies.items():
            # convert location
            bodies[bodies_id]['boundary']['location'] = [loc.to(self.length_unit).magnitude for loc in
                                                         specs['boundary']['location']]
            # convert diameter
            bodies[bodies_id]['boundary']['length'] = specs['boundary']['length'].to(self.length_unit).magnitude
            bodies[bodies_id]['boundary']['width'] = specs['boundary']['width'].to(self.length_unit).magnitude
            # convert mass
            bodies[bodies_id]['boundary']['mass'] = specs['boundary']['mass'].to(self.mass_unit).magnitude
            # convert velocity
            bodies[bodies_id]['boundary']['velocity'] = specs['boundary']['velocity'].to(self.velocity_unit).magnitude
        return bodies

    def location_add_units(self, bodies):
        for body_id, body_data in bodies.items():
            location = body_data['location']
            bodies[body_id] = {
                'location': [(loc * self.length_unit) for loc in location],
                'angle': body_data['angle']}
        return bodies

    def remove_length_units(self, value):
        return value.to(self.length_unit).magnitude

    def animate_frame(self, agents):
        """matplotlib interactive plot"""
        plt.cla()
        bounds = copy.deepcopy(self.config['bounds'])
        for cell_id, data in agents.items():
            # location, orientation, length
            data = data['boundary']
            x_center = self.remove_length_units(data['location'][0])
            y_center = self.remove_length_units(data['location'][1])
            length = self.remove_length_units(data['length'])

            # get bottom left position
            radius = (length / 2)
            x = x_center - radius
            y = y_center - radius

            # Create a circle
            circle = patches.Circle((x, y), radius, linewidth=1, edgecolor='b')
            self.ax.add_patch(circle)

        xl = self.remove_length_units(bounds[0])
        yl = self.remove_length_units(bounds[1])
        plt.xlim([-xl, 2 * xl])
        plt.ylim([-yl, 2 * yl])
        plt.draw()
        plt.pause(0.01)


def get_agent_config(
        location=None,
        bounds=None,
        velocity=None,
        # volume=None,
        length=None,
        width=None,
        mass=None,
        unit_system=None,
):
    unit_system = unit_system or {
        'length': DEFAULT_LENGTH_UNIT,
        'width': DEFAULT_LENGTH_UNIT,
        'mass': DEFAULT_MASS_UNIT,
        'time': DEFAULT_TIME_UNIT,
        'velocity': DEFAULT_LENGTH_UNIT / DEFAULT_TIME_UNIT,
    }

    length = length or 5 * unit_system['length']
    width = width or 1.0 * unit_system['length']
    mass = mass or 5 * unit_system['mass']
    volume = sphere_volume_from_diameter(length)
    velocity = velocity or 5 * unit_system['velocity']

    bounds = bounds or DEFAULT_BOUNDS
    if location:
        location = [
            loc * bounds[n] * unit_system['length']
            for n, loc in enumerate(location)]
    else:
        location = make_random_position(bounds)

    return {
        'grow': {
            '_type': 'process',
            'address': 'local:grow',
            'config': {
                'growth_rate': 0.006,
                'default_growth_noise': 1e-3,
            },
        },
        'boundary': {
            'location': location,
            'velocity': velocity,
            'volume': volume,
            'length': length,
            'width': width,
            'angle': random.uniform(0, 2 * PI),
            'mass': mass,
            'thrust': 0,
            'torque': 0,
        }
    }


@pytest.fixture
def core():
    corex = ProcessTypes()
    REGISTER_TYPES(corex)
    corex.process_registry.register('multibody', Multibody)
    return corex


def run_multibody(core):
    n_agents = 10
    bounds = DEFAULT_BOUNDS

    state = {
        'multibody': {
            '_type': 'process',
            'address': 'local:multibody',
            'config': {
                'bounds': bounds
            },
            'inputs': {
                'agents': ['agents'],
            },
            'outputs': {
                'agents': ['agents'],
            }
        },
        'agents': {
            str(i): get_agent_config(
                bounds=bounds)
            for i in range(n_agents)
        },
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'agents': 'map[boundary:boundary]',
                    'time': 'float'
                }
            },
            'inputs': {
                'agents': ['agents'],
                'time': ['global_time']
            },
        }
    }

    sim = Composite({
        'state': state
    }, core=core)

    # run the simulation
    sim.run(10)

    data = sim.gather_results()

    agents, fields = format_snapshot_data(data[('emitter',)])
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        out_dir='out',
    )

    # make video
    make_video(
        data={'fields': fields,
              'agents': agents},
        bounds=bounds,
        # plot_type='tag',
        # step=step,
        out_dir='out',
        filename=f"snapshots",
        # highlight_agents=highlight_agents,
        # show_timeseries=tagged_molecules,
    )

if __name__ == '__main__':
    core = ProcessTypes()
    REGISTER_TYPES(core)
    core.process_registry.register('multibody', Multibody)

    run_multibody(core)
