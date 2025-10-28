'''
Tests for pymunk multibody simulations, including growth and division.
'''

from bigraph_viz import plot_bigraph, VisualizeTypes
from process_bigraph import Composite, gather_emitter_results, ProcessTypes
from process_bigraph.emitter import emitter_from_wires
from pymunk_process import core_import
from pymunk_process.processes.multibody import make_initial_state
from pymunk_process.processes.grow_divide import get_grow_divide_schema
from pymunk_process.plots.multibody_plots import simulation_to_gif


# core = VisualizeTypes()
class VivariumTypes(ProcessTypes, VisualizeTypes):
    def __init__(self):
        super().__init__()

core = VivariumTypes()
PYMUNK_CORE = core_import(core)


def run_pymunk_experiment():
    core = PYMUNK_CORE
    initial_state = make_initial_state(
        n_microbes=2,
        n_particles=100,
        env_size=600,
        elasticity=0.0,
        particle_radius_range=(1, 8),
        microbe_length_range=(50, 100),
        microbe_radius_range=(10, 15)
    )

    # run simulation
    interval = 0.1
    steps = 2000
    config = {
        'env_size': 600,
        'gravity': 0, #-9.81,
        'elasticity': 0.1,
    }

    processes = {
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': config,
            'inputs': {
                'agents': ['cells'],
                'particles': ['particles'],
            },
            'outputs': {
                'agents': ['cells'],
                'particles': ['particles'],
            }
        }
    }

    # emitter state
    emitter_spec = {'agents': ['cells'],
                    'particles': ['particles'],
                    'time': ['global_time']}
    emitter_state = emitter_from_wires(emitter_spec)

    # grow and divide schema
    cell_schema = get_grow_divide_schema(
        core=core,
        config={
            'agents_key': 'cells',
            'rate': 0.02,
            'threshold': 80.0,
            'mutate': True,
        }
    )

    # complete document
    doc = {
        'state': {
            **initial_state,
            **processes,
            **{'emitter': emitter_state},
        },
        'composition': cell_schema,
    }

    # create the composite simulation
    sim = Composite(doc, core=core)

    # Save composition JSON
    name = 'pymunk_growth_division'
    sim.save(filename=f'{name}.json', outdir='out')

    # Save visualization of the initial composition
    plot_state = {k: v for k, v in sim.state.items() if k not in ['global_time', 'emitter']}
    plot_schema = {k: v for k, v in sim.composition.items() if k not in ['global_time', 'emitter']}

    plot_bigraph(
        state=plot_state,
        schema=plot_schema,
        core=core,
        out_dir='out',
        filename=f'{name}_viz',
        dpi='300',
        collapse_redundant_processes=True
    )


    # run the simulation
    total_time = interval * steps
    sim.run(total_time)
    results = gather_emitter_results(sim)[('emitter',)]

    print(f'Simulation completed with {len(results)} steps.')

    # make video
    simulation_to_gif(results,
                      filename='circlesandsegments',
                      config=config,
                      color_by_phylogeny=True,
                      # skip_frames=10
                      )


if __name__ == '__main__':
    run_pymunk_experiment()

