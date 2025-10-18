"""
Tests for pymunk multibody simulations, including growth and division.
"""
import numpy as np
import copy
import math
import random
from typing import Dict, Any, Tuple, List, Optional

from bigraph_viz import plot_bigraph, VisualizeTypes
from process_bigraph import Composite, gather_emitter_results, ProcessTypes
from process_bigraph.emitter import emitter_from_wires
from pymunk_process import core_import, PymunkProcess
from pymunk_process.processes.multibody import daughter_locations
from pymunk_process.plots.multibody_plots import simulation_to_gif


# core = VisualizeTypes()
class VivariumTypes(ProcessTypes, VisualizeTypes):
    def __init__(self):
        super().__init__()

core = VivariumTypes()
PYMUNK_CORE = core_import(core)




def make_initial_state(
    n_circles: int = 2,
    n_segments: int = 2,
    env_size: float = 600.0,
    *,
    key: str = "cells",
    seed: Optional[int] = None,
    elasticity: float = 0.0,
    # circle knobs
    circle_radius_range: Tuple[float, float] = (12.0, 30.0),
    circle_mass_density: float = 0.015,  # mass ≈ density * π r^2
    circle_speed_range: Tuple[float, float] = (0.0, 10.0),
    # segment knobs
    segment_length_range: Tuple[float, float] = (40.0, 120.0),
    segment_radius_range: Tuple[float, float] = (6.0, 24.0),  # visual thickness
    segment_mass_density: float = 0.02,  # mass ≈ density * length * (2*radius)
    segment_speed_range: Tuple[float, float] = (0.0, 0.4),
    # placement & overlap
    margin: float = 5.0,          # keep objects away from walls (world units)
    avoid_overlap_circles: bool = True,
    min_gap: float = 2.0,         # extra separation for circles (world units)
    max_tries_per_circle: int = 200,
) -> Dict[str, Any]:
    """
    Build an initial state dict:
        { key: { '0': {...}, '1': {...}, ... } }

    - Circles: random positions, radii; optional simple de-overlap.
    - Segments: random positions, lengths, radii, and angles; coarse placement
      that stays inside bounds (no strict de-overlap with others).
    - Mass is computed from simple density heuristics.

    Parameters let you tune sizes, counts, and env_size without touching call sites.
    """
    rng = random.Random(seed)
    agents: Dict[str, Dict[str, Any]] = {}

    # --- helpers ---
    def rand_circle() -> Dict[str, Any]:
        r = rng.uniform(*circle_radius_range)
        # place fully inside bounds (x,y in [margin+r, env_size-(margin+r)])
        x = rng.uniform(margin + r, env_size - (margin + r))
        y = rng.uniform(margin + r, env_size - (margin + r))
        speed = rng.uniform(*circle_speed_range)
        theta = rng.uniform(0, 2*math.pi)
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
        mass = circle_mass_density * math.pi * (r ** 2)
        return {
            'type': 'circle',
            'mass': float(mass),
            'radius': float(r),
            'location': (float(x), float(y)),
            'velocity': (float(vx), float(vy)),
            'elasticity': float(elasticity),
        }

    def circles_overlap(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
        (x1, y1), r1 = c1['location'], c1['radius']
        (x2, y2), r2 = c2['location'], c2['radius']
        dx, dy = x1 - x2, y1 - y2
        return (dx*dx + dy*dy) < (r1 + r2 + min_gap) ** 2

    def place_circles(n: int) -> List[Dict[str, Any]]:
        placed: List[Dict[str, Any]] = []
        for _ in range(n):
            if avoid_overlap_circles:
                for _try in range(max_tries_per_circle):
                    cand = rand_circle()
                    if all(not circles_overlap(cand, prev) for prev in placed):
                        placed.append(cand)
                        break
                else:
                    # fallback: accept even if overlapping after many tries
                    placed.append(rand_circle())
            else:
                placed.append(rand_circle())
        return placed

    def rand_segment() -> Dict[str, Any]:
        L = rng.uniform(*segment_length_range)
        rad = rng.uniform(*segment_radius_range)   # visual thickness in world units
        ang = rng.uniform(-math.pi, math.pi)

        # keep entire capsule inside bounds
        dx, dy = (L/2.0) * math.cos(ang), (L/2.0) * math.sin(ang)
        # bounding circle radius for capsule ≈ sqrt((L/2)^2 + (2*rad)^2)
        # but to be safe, pad endpoints by rad
        pad = rad + margin
        x = rng.uniform(pad + abs(dx), env_size - (pad + abs(dx)))
        y = rng.uniform(pad + abs(dy), env_size - (pad + abs(dy)))

        speed = rng.uniform(*segment_speed_range)
        phi = rng.uniform(0, 2*math.pi)
        vx, vy = speed * math.cos(phi), speed * math.sin(phi)

        # simple mass heuristic proportional to length * diameter
        mass = segment_mass_density * L * (2.0 * rad)
        return {
            'type': 'segment',
            'mass': float(mass),
            'length': float(L),
            'radius': float(rad),
            'angle': float(ang),
            'location': (float(x), float(y)),
            'velocity': (float(vx), float(vy)),
            'elasticity': float(elasticity),
        }

    # --- build circles then segments ---
    idx = 0
    for c in place_circles(n_circles):
        agents[str(idx)] = c
        idx += 1
    for _ in range(n_segments):
        agents[str(idx)] = rand_segment()
        idx += 1

    return {key: agents}


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
    initial_state = make_initial_state(
        n_circles=500,
        n_segments=5,
        env_size=600,
        seed=42,              # optional for reproducibility
        elasticity=0.0,
        circle_radius_range=(1, 10),
        segment_length_range=(50, 80),
        segment_radius_range=(10, 20)
    )

    # run simulation
    interval = 0.1
    steps = 600
    config = {
        'env_size': 600,
        'gravity': 0, #-9.81,
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
    emitter_spec = {'agents': ['cells'],
                    'time': ['global_time']}
    emitter_state = emitter_from_wires(emitter_spec)

    # complete document
    doc = {
        'state': {
            **initial_state,
            **processes,
            **{'emitter': emitter_state},
        }
    }

    # create the composite simulation
    sim = Composite(doc, core=PYMUNK_CORE)

    # Save composition JSON
    name = 'pymunk_growth_division'
    sim.save(filename=f'{name}.json', outdir='out')

    # Save visualization of the initial composition
    plot_state = {k: v for k, v in sim.state.items() if k not in ['global_time', 'emitter']}
    plot_schema = {k: v for k, v in sim.composition.items() if k not in ['global_time', 'emitter']}

    plot_bigraph(
        state=plot_state,
        schema=plot_schema,
        core=PYMUNK_CORE,
        out_dir='out',
        filename=f'{name}_viz',
        dpi='300',
        collapse_redundant_processes=True
    )


    # run the simulation
    total_time = interval * steps
    sim.run(total_time)
    results = gather_emitter_results(sim)[('emitter',)]

    print(f"Simulation completed with {len(results)} steps.")

    # make video
    simulation_to_gif(results,
                      filename='circlesandsegments',
                      config=config,
                      # skip_frames=10
                      )


if __name__ == '__main__':
    run_pymunk_experiment()

