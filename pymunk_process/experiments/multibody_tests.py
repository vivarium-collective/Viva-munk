"""
Tests for pymunk multibody simulations, including growth and division.
"""
import numpy as np
import copy
import random
import math

from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import emitter_from_wires
from pymunk_process import get_pymunk_core, PymunkProcess
from pymunk_process.processes.multibody import daughter_locations
from pymunk_process.plots.multibody_plots import simulation_to_gif

PYMUNK_CORE = get_pymunk_core()


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
        'cells': {
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

    processes = {
        'multibody': {
            "_type": "process",
            'address': 'local:pymunk_process',
            "config": config,
            "inputs": {
                "agents": ['cells'],
            },
            "outputs": {
                "agents": ['cells'],
            }
        }
    }

    # emitter state
    emitter_spec = {key: [key] for key in ['agents']}
    emitter_state = emitter_from_wires(emitter_spec)

    doc = {
        'state': {
            **initial_state,
            **processes,
            **{'emitter': emitter_state},
        }
    }

    sim = Composite(doc, core=PYMUNK_CORE)
    total_time = interval * steps
    sim.run(total_time)
    results = gather_emitter_results(sim)[('emitter',)]

    # make video
    simulation_to_gif(results, filename='circlesandsegments', config=config, skip_frames=10)



if __name__ == '__main__':
    run_pymunk_experiment()

