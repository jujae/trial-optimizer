"""
Trial Optimizer: Deep Learning Optimization of Graphical Procedures for Multiplicity Control

Implementation based on:
Zhan et al. (2022) - "Optimizing Graphical Procedures for Multiplicity Control 
in a Confirmatory Clinical Trial via Deep Learning"

Extended with:
Maurer & Bretz (2013) - "Multiple testing in group sequential trials using graphical approaches"
"""

from .graphical_procedure import (
    GraphicalProcedure,
    BonferroniProcedure,
    HolmProcedure,
    FixedSequenceProcedure,
    FallbackProcedure,
    TestResult,
)
from .spending_functions import (
    SpendingFunction,
    OBrienFleming,
    Pocock,
    Linear,
    HwangShihDeCani,
)
from .power_simulator import PowerSimulator
from .objectives import (
    # Core classes
    SuccessFunction,
    Objective,
    # Success functions
    MarginalRejection,
    GatedSuccess,
    # Objective functions
    WeightedSuccess,
)
from .neural_network import GraphicalProcedureNetwork
from .optimizer import (
    GraphicalProcedureOptimizer,
    COBYLAOptimizer,
    GridSearchOptimizer,
    optimize_graphical_procedure,
    optimize_sequential_procedure,
)
from .visualization import plot_graphical_procedure

__version__ = "0.1.0"
__all__ = [
    "GraphicalProcedure",
    "BonferroniProcedure",
    "HolmProcedure",
    "FixedSequenceProcedure",
    "FallbackProcedure",
    "TestResult",
    "SpendingFunction",
    "OBrienFleming",
    "Pocock",
    "Linear",
    "HwangShihDeCani",
    "PowerSimulator",
    # Core classes
    "SuccessFunction",
    "Objective",
    # Success functions
    "MarginalRejection",
    "GatedSuccess",
    # Objective functions
    "WeightedSuccess",
    "GraphicalProcedureNetwork",
    "GraphicalProcedureOptimizer",
    "COBYLAOptimizer",
    "GridSearchOptimizer",
    "optimize_graphical_procedure",
    "optimize_sequential_procedure",
    "plot_graphical_procedure",
]
