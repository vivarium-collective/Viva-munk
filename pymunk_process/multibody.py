"""
=========================
Multibody physics process
=========================

Simulates collisions between cell bodies with a physics engine.
"""

import random
import math
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# vivarium imports
from process_bigraph import Process, Composite, process_registry, types
from pymunk_process.pymunk_minimal import PymunkMinimal as Pymunk
from pymunk_process.units import units, remove_units


DEFAULT_LENGTH_UNIT = units.um
DEFAULT_BOUNDS = [200 * DEFAULT_LENGTH_UNIT, 200 * DEFAULT_LENGTH_UNIT]
DEFAULT_MASS_UNIT = units.ng
DEFAULT_VELOCITY_UNIT = units.um / units.s

# constants
PI = math.pi


# helper functions
def daughter_locations(mother_location, state):
    """Given a mother's location, place daughters close"""
    mother_diameter = state['diameter']
    mother_x = mother_location[0]
    mother_y = mother_location[1]
    locations = []
    for daughter in range(2):
        location = [mother_x + random.gauss(0, 0.1) * mother_diameter,
                    mother_y + random.gauss(0, 0.1) * mother_diameter]
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


# # Add a bounds type
# bounds_type = {
#     'x': 'float',
#     'y': 'float',
# }
# types.type_registry.register('bounds', bounds_type)
boundary_type = {
    'location': 'list',  # TODO make this work: 'tuple[2,float]',
    'diameter': 'length',
    'mass': 'mass',
    'velocity': 'length/time'
}
types.type_registry.register('boundary', boundary_type)
types.type_registry.register('unit', {'_super': 'string'})


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
        'bounds': {
            'x': 'float',
            'y': 'float',
        },
        'length_unit': {'_type': 'string', '_default': 'um'},
        'mass_unit': {'_type': 'string', '_default': 'ng'},
        'velocity_unit': {'_type': 'string', '_default': 'um/s'},
        'animate': 'boolean',
    }
    # config_schema = {
    #     'jitter_force': 1e-6,
    #     'bounds': remove_units(DEFAULT_BOUNDS),
    #     'length_unit': DEFAULT_LENGTH_UNIT,
    #     'mass_unit': DEFAULT_MASS_UNIT,
    #     'velocity_unit': DEFAULT_VELOCITY_UNIT,
    #     'animate': False,
    # }

    def __init__(self, config=None):
        super().__init__(config)

        self.length_unit = units(self.config['length_unit'])
        self.mass_unit = units(self.config['mass_unit'])
        self.velocity_unit = units(self.config['velocity_unit'])
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

    def schema(self):
        return {
            'agents': 'tree[boundary]'  # TODO -- make a simpler type "mapping"
        }

    def update(self, state, interval):
        agents = state['agents']

        # animate before update
        if self.animate:
            self.animate_frame(agents)

        # update physics with new agents
        agents = self.bodies_remove_units(agents)
        self.physics.update_bodies(agents)

        # run simulation
        self.physics.run(interval)

        # get new cell positions and neighbors
        cell_positions = self.physics.get_body_positions()

        # add units to cell_positions
        cell_positions = self.location_add_units(cell_positions)

        return {
            'agents': {
                cell_id: {
                    'boundary': {
                        'location': list(cell_positions[cell_id])},
                } for cell_id in agents.keys()
            }
        }

    def bodies_remove_units(self, bodies):
        """Convert units to the standard, and then remove them

        This is required for interfacing the physics engine, which does not track units
        """
        for bodies_id, specs in bodies.items():
            # convert location
            bodies[bodies_id]['boundary']['location'] = [loc.to(self.length_unit).magnitude for loc in
                                                         specs['boundary']['location']]
            # convert diameter
            bodies[bodies_id]['boundary']['diameter'] = specs['boundary']['diameter'].to(self.length_unit).magnitude
            # convert mass
            bodies[bodies_id]['boundary']['mass'] = specs['boundary']['mass'].to(self.mass_unit).magnitude
            # convert velocity
            bodies[bodies_id]['boundary']['velocity'] = specs['boundary']['velocity'].to(self.velocity_unit).magnitude
        return bodies

    def location_add_units(self, bodies):
        for body_id, location in bodies.items():
            bodies[body_id] = [(loc * self.length_unit) for loc in location]
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
            diameter = self.remove_length_units(data['diameter'])

            # get bottom left position
            radius = (diameter / 2)
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


process_registry.register('multibody', Multibody)


def get_agent_config(
        location=None,
        bounds=None,
        velocity=None,
        # volume=None,
        diameter=None,
        mass=None,
):
    diameter = diameter or 5 * DEFAULT_LENGTH_UNIT
    mass = mass or 5 * DEFAULT_MASS_UNIT
    volume = sphere_volume_from_diameter(diameter)

    bounds = bounds or DEFAULT_BOUNDS
    if location:
        location = [loc * bounds[n] for n, loc in enumerate(location)]
    else:
        location = make_random_position(bounds)

    return {
        'boundary': {
            'location': location,
            'velocity': velocity,
            'volume': volume,
            'diameter': diameter,
            'mass': mass,
            'thrust': 0,
            'torque': 0,
        }
    }


def test_multibody():
    n_agents = 10
    bounds = DEFAULT_BOUNDS
    state = {
        'multibody': {
            '_type': 'process',
            'address': 'local:multibody',
            'config': {
                'bounds': bounds
            },
            'wires': {
                'agents': 'agents'
            }
        },
        'agents': {
            str(i): get_agent_config(bounds=bounds) for i in range(n_agents)
        }
    }

    sim = Composite({
        'state': state
    })

    # run the simulation
    sim.run(10)

    pass


if __name__ == '__main__':
    test_multibody()
