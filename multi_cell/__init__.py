"""
Registration-related functions
"""

from process_bigraph import register_types as pb_register_types
from process_bigraph.types.process import register_types as pb_types_register
from process_bigraph.composite import Composite
from process_bigraph.emitter import RAMEmitter
from bigraph_schema.core import Core, BASE_TYPES
from bigraph_viz import register_types as viz_register_types
from bigraph_schema.schema import Tuple, Float, Enum
from multi_cell.processes.multibody import PymunkProcess
from multi_cell.processes.grow_divide import GrowDivide
from multi_cell.processes.remove_crossing import RemoveCrossing
from multi_cell.processes.secrete_eps import SecreteEPS
from multi_cell.pymunk_agent_type import PymunkAgent, register_pymunk_agent_dispatches

# Register custom dispatches once at module import
register_pymunk_agent_dispatches()


def register_pymunk_types(core):
    # Use the optimized PymunkAgent Node subclass instead of a dict schema.
    # This eliminates per-field dispatch overhead in apply/reconcile/realize.
    core.register_type('pymunk_agent', PymunkAgent())


def register_processes(core):
    core.register_link('PymunkProcess', PymunkProcess)
    core.register_link('GrowDivide', GrowDivide)
    core.register_link('RemoveCrossing', RemoveCrossing)
    core.register_link('SecreteEPS', SecreteEPS)
    core.register_link('Composite', Composite)
    core.register_link('RAMEmitter', RAMEmitter)


def core_import(core=None, config=None):
    if not core:
        core = Core(BASE_TYPES)
    pb_types_register(core)
    pb_register_types(core)
    viz_register_types(core)
    register_pymunk_types(core)
    register_processes(core)
    return core
