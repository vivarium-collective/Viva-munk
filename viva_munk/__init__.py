"""
Registration-related functions
"""

__version__ = "0.0.1"

from process_bigraph import register_types as pb_register_types
from process_bigraph.types.process import register_types as pb_types_register
from process_bigraph.composite import Composite
from process_bigraph.emitter import RAMEmitter, SQLiteEmitter
from bigraph_schema.core import Core, BASE_TYPES
from bigraph_viz import register_types as viz_register_types

from viva_munk.processes.multibody import PymunkProcess
from viva_munk.processes.grow_divide import GrowDivide, AdderGrowDivide
from viva_munk.processes.remove_crossing import RemoveCrossing
from viva_munk.processes.secrete_eps import SecreteEPS
from viva_munk.processes.pressure import Pressure
from viva_munk.processes.diffusion_advection import DiffusionAdvection
from viva_munk.processes.cell_field_exchange import CellFieldExchange
from viva_munk.processes.chemotaxis import Chemotaxis
from viva_munk.processes.inclusion_body import InclusionBody, IBColony
from viva_munk.processes.quorum_sensing import QuorumSensing
from viva_munk.processes.field_decay import FieldDecay
# Spatio-flux processes referenced by spatio_flux.composites.particles.* —
# brownian_particles needs both BrownianMovement and ManageBoundaries to
# resolve their `local:` addresses at run time.
from spatio_flux.processes.particles import BrownianMovement, ManageBoundaries
# Spatio-flux Visualization Steps — heatmaps, GIFs, snapshot grids, and
# emitter-driven timeseries. Registered so dashboard users can attach them
# to any composite via the Visualizations tab.
from spatio_flux.visualizations import (
    FieldHeatmap,
    FieldAnimationGif,
    FieldSnapshotsGrid,
    ParticleTraces,
    TestSuiteTimeSeries,
)
from viva_munk.pymunk_agent_type import PymunkAgent, register_pymunk_agent_dispatches
from viva_munk.types import positive_types
from viva_munk.visualizations import MultibodyVizStep, CellMassTraces

# Register custom dispatches once at module import
register_pymunk_agent_dispatches()


def _patch_composite_realize_on_sentinels():
    """Ensure Composite re-realizes newly-added subtrees on `_add`/`_remove`.

    `Map.apply` drops `_add` entries straight into state without calling
    the value type's realize path, and `Composite.apply_updates` only
    triggers `core.realize` on schema-level merges — not on structural
    sentinels. So embedded process specs (e.g. `grow_divide` on a freshly
    divided daughter) are never instantiated: the spec is visible in
    state but its `'instance'` slot stays unset and the Process class
    never runs on that cell. Result: lineages freeze after the first
    division in every experiment that uses per-cell embedded processes.

    The fix is a one-line amend: when structural sentinels are detected,
    run `core.realize` (which follows `Link`/process specs and creates
    instances via `realize_link`) before `find_instance_paths` scans
    state for them.
    """
    _original_apply_updates = Composite.apply_updates

    def apply_updates_with_realize(self, updates):
        update_paths = _original_apply_updates(self, updates)
        if getattr(self, '_last_apply_structural', False):
            self.schema, self.state = self.core.realize(self.schema, self.state)
            self.find_instance_paths(self.state)
            self._build_view_project_cache()
        return update_paths

    Composite.apply_updates = apply_updates_with_realize


_patch_composite_realize_on_sentinels()


def _patch_list_apply_tuple_update():
    """Make ``apply(schema: List, state: list, update: tuple)`` do
    element-wise add when shape + numeric content match.

    JSON serialization drops the tuple type, so a state that started as a
    ``(x, y)`` tuple becomes a ``[x, y]`` list and schema inference picks
    ``List(_element=Float)`` instead of ``Tuple(_values=[Float, Float])``.
    The default ``apply(List)`` then does ``state + update`` — concat, not
    element-wise — and crashes on the ``list + tuple`` type mismatch the
    moment a process emits a tuple delta (``PymunkProcess`` does this for
    ``location``/``velocity`` every tick). Pin element-wise behavior for
    the JSON-roundtripped case so the dashboard's subprocess flow works.
    """
    from plum import dispatch
    from bigraph_schema.schema import List
    from bigraph_schema.methods.apply import apply as _apply

    @_apply.dispatch
    def apply_list_with_tuple_update(schema: List, state: list, update: tuple, path):
        if len(state) == len(update) and all(
            isinstance(s, (int, float)) for s in state
        ):
            return [s + u for s, u in zip(state, update)], []
        # Length mismatch or non-numeric: fall back to concat (matches
        # the default ``state + list(update)`` semantics).
        return list(state) + list(update), []


_patch_list_apply_tuple_update()


from viva_munk import composites as _composites  # noqa: E402,F401 — fires @composite_generator side-effects


def register_pymunk_types(core):
    # Use the optimized PymunkAgent Node subclass instead of a dict schema.
    # This eliminates per-field dispatch overhead in apply/reconcile/realize.
    core.register_type('pymunk_agent', PymunkAgent())
    # NOTE: viva_munk.types.positive_types overlaps with spatio_flux's on
    # ('positive_float', 'positive_array', 'concentration', 'set_float') —
    # viva_munk uses instances (PositiveFloat()), spatio_flux uses classes
    # (PositiveFloat). Registering both forms triggers a resolve conflict in
    # bigraph_schema. spatio_flux's set is a strict superset (adds count,
    # mass, delta_conc), so we let spatio_flux_register_types own these
    # types in core_import() and skip the duplicate loop here.


def register_processes(core):
    core.register_link('PymunkProcess', PymunkProcess)
    core.register_link('GrowDivide', GrowDivide)
    core.register_link('AdderGrowDivide', AdderGrowDivide)
    core.register_link('RemoveCrossing', RemoveCrossing)
    core.register_link('SecreteEPS', SecreteEPS)
    core.register_link('Pressure', Pressure)
    core.register_link('DiffusionAdvection', DiffusionAdvection)
    core.register_link('CellFieldExchange', CellFieldExchange)
    core.register_link('Chemotaxis', Chemotaxis)
    core.register_link('InclusionBody', InclusionBody)
    core.register_link('IBColony', IBColony)
    core.register_link('QuorumSensing', QuorumSensing)
    core.register_link('FieldDecay', FieldDecay)
    core.register_link('BrownianMovement', BrownianMovement)
    core.register_link('ManageBoundaries', ManageBoundaries)
    core.register_link('MultibodyVizStep', MultibodyVizStep)
    core.register_link('CellMassTraces', CellMassTraces)
    # Spatio-flux Visualization Steps
    core.register_link('FieldHeatmap', FieldHeatmap)
    core.register_link('FieldAnimationGif', FieldAnimationGif)
    core.register_link('FieldSnapshotsGrid', FieldSnapshotsGrid)
    core.register_link('ParticleTraces', ParticleTraces)
    core.register_link('TestSuiteTimeSeries', TestSuiteTimeSeries)
    core.register_link('Composite', Composite)
    core.register_link('RAMEmitter', RAMEmitter)
    core.register_link('SQLiteEmitter', SQLiteEmitter)


def core_import(core=None, config=None):
    if not core:
        core = Core(BASE_TYPES)
    pb_types_register(core)
    pb_register_types(core)
    viz_register_types(core)
    # Spatio-flux types (particle, position, bounds, fields, ...) are
    # referenced by spatio_flux.composites.particles.* and other composites
    # used in this workspace. Their Process classes (registered below) declare
    # ports like 'map[particle]' that won't resolve without these types.
    from spatio_flux import register_types as spatio_flux_register_types
    spatio_flux_register_types(core)
    register_pymunk_types(core)
    register_processes(core)
    return core
