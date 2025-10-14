import copy
import math
import os
import random

import numpy as np
from imageio import v2 as imageio
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from process_bigraph import Composite, gather_emitter_results

from pymunk_process import get_pymunk_core, PymunkProcess
from pymunk_process.processes.multibody import daughter_locations

PYMUNK_CORE = get_pymunk_core()


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


def growth_division_simulation(
        initial_state,
        config,
        interval,
        steps,
        growth_rate,
        division_threshold,
        *,
        threshold_noise_std=0.0,          # 0 = no noise
        threshold_noise_mode='relative',   # 'relative' or 'absolute'
        seed=None                          # for reproducibility
):
    """
    Simulate growth and division with a noisy division threshold.

    Parameters
    ----------
    division_threshold : float
        Base mass threshold for division.
    threshold_noise_std : float, optional
        Std dev of Gaussian noise applied to the threshold at each agent check.
        - If mode='relative', effective_threshold = threshold * (1 + N(0, std))
        - If mode='absolute', effective_threshold = threshold + N(0, std)
    threshold_noise_mode : {'relative', 'absolute'}, optional
        How to apply noise to the threshold.
    seed : int or None
        RNG seed for reproducibility.

    Notes
    -----
    - Noise is sampled independently *each timestep per agent*.
    - Effective threshold is clamped to > 0.
    """
    rng_np = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    process = PymunkProcess(config, core=PYMUNK_CORE)
    state = copy.deepcopy(initial_state)

    def noisy_threshold(base):
        if threshold_noise_std <= 0:
            return base
        if threshold_noise_mode == 'relative':
            eff = base * (1.0 + rng_np.normal(0.0, threshold_noise_std))
        elif threshold_noise_mode == 'absolute':
            eff = base + rng_np.normal(0.0, threshold_noise_std)
        else:
            raise ValueError("threshold_noise_mode must be 'relative' or 'absolute'")
        return max(eff, 1e-12)

    timeline = []
    for step in range(steps):
        next_state = {'agents': {}}

        for agent_id, agent in state['agents'].items():
            a = copy.deepcopy(agent)

            # growth
            a['mass'] = a['mass'] * (1 + growth_rate)
            if a['type'] == 'circle':
                a['radius'] = a['radius'] * (1 + growth_rate)
            elif a['type'] == 'segment':
                a['length'] = a['length'] * (1 + growth_rate)

            # noisy threshold for this agent at this step
            eff_threshold = noisy_threshold(division_threshold)

            # division check
            if a['mass'] > eff_threshold:
                half_mass = a['mass'] / 2
                locs = daughter_locations(a)  # your helper

                for i in (0, 1):
                    na = copy.deepcopy(a)
                    na['location'] = locs[i]
                    na['mass'] = half_mass
                    if na['type'] == 'circle':
                        # keep area proportional: r -> r / sqrt(2)
                        na['radius'] = na['radius'] / math.sqrt(2)
                    elif na['type'] == 'segment':
                        na['length'] = na['length'] / 2
                        # tiny angular jitter to avoid perfect overlaps (optional)
                        na['angle'] = na.get('angle', 0.0) + rng_np.normal(0.0, 0.02)

                    next_state['agents'][f"{agent_id}{i}"] = na
            else:
                next_state['agents'][agent_id] = a

        # dynamics step
        new_state = process.update(next_state, interval)

        timeline.append({
            'time': step * interval,
            **new_state
        })

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
        'elasticity': 0.1,
    }
    simulation_data2 = run_simulation(initial_state, config, interval, steps)

    # make video
    simulation_to_gif(simulation_data2, filename='circlesandsegments', config=config, skip_frames=10)


def run_growth_division():
    initial_state = {
        'agents': {
            'X': {
                '_type': 'segment_agent',
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
    steps = 1500
    growth_rate = 0.002
    configgr = {'env_size': 600,
                'gravity': 0,
                'jitter_force': 5e-2,
                }
    simulation_data4 = growth_division_simulation(
        initial_state,
        configgr,
        interval,
        steps,
        growth_rate,
        division_threshold=21,
        threshold_noise_std=0.1,
        threshold_noise_mode='relative',
    )

    # make video
    simulation_to_gif(
        simulation_data4,
        config=configgr,
        skip_frames=10,
        filename='growth_division.gif')


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
    data = gather_emitter_results(sim)
    return data


def run_composition_experiment():
    initial_state = {
        'agents': {
            'X': {
                '_type': 'segment_agent',
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
    config = {'env_size': 600, 'gravity': 0}
    results = run_composition(initial_state, config, interval, steps)

    # make video
    simulation_to_gif(
        results,
        config=config,
        skip_frames=10,
        filename='composite_experiment.gif')


if __name__ == '__main__':
    # run_pymunk_experiment()
    run_growth_division()
    # run_composition_experiment()
