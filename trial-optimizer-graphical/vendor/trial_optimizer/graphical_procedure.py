"""
Graphical Procedure for Multiplicity Control

Implements the graphical approach for multiple testing procedures as described by
Bretz et al. (2009) and used in Zhan et al. (2022), with extensions for group
sequential designs from Maurer & Bretz (2013).

A graphical procedure is defined by:
1. Initial alpha allocation: w = (w_1, ..., w_m) where sum(w_i) = 1
2. Transition matrix: G where G_ij represents the fraction of alpha to transfer
   from hypothesis i to hypothesis j when i is rejected
3. (Sequential) Number of analyses: n_analyses (default=1 for standard procedures)
4. (Sequential) Information fractions and spending functions per hypothesis

Key constraints:
- w_i >= 0 for all i, and sum(w_i) = 1
- G_ii = 0 for all i (no self-loops)
- G_ij >= 0 for all i, j
- sum_j(G_ij) <= 1 for all i (can be < 1 to allow alpha to leave the system)

## Standard Procedures

The module provides class-based constructors for common procedures:

- `BonferroniProcedure`: Most conservative, no alpha propagation
- `HolmProcedure`: Uniformly more powerful than Bonferroni
- `FixedSequenceProcedure`: Tests in pre-specified order
- `FallbackProcedure`: Flexible initial weighting with sequential propagation

Example:
    >>> from trial_optimizer import HolmProcedure
    >>> proc = HolmProcedure(m=3, alpha=0.025)
    >>> result = proc.test(p_values=np.array([0.01, 0.02, 0.03]))
"""

import numpy as np
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass

from ._utils import (
    prepare_information_fractions,
    prepare_p_values,
    prepare_spending_functions,
    prepare_weights_transitions,
)


@dataclass
class TestResult:
    """
    Result of applying a graphical procedure to p-values.
    
    Handles both standard (n_analyses=1) and sequential (n_analyses>1) designs.
    For standard procedures, sequential-specific fields have default values.
    """
    rejected: np.ndarray  # Boolean array indicating which hypotheses were rejected
    rejection_order: List[Tuple[int, int]]  # Order of rejections: (hyp_idx, analysis_idx)
    rejection_analysis: Optional[np.ndarray] = None  # 1-based analysis number for each hypothesis (0 if not rejected)
    stopped_early: Optional[bool] = None  # Whether trial stopped before final analysis
    stop_analysis: Optional[int] = None  # 1-based analysis at which the procedure stopped
    graphs_by_analysis: Optional[List[dict]] = None  # Graph state at each analysis
    
    @property
    def num_rejected(self) -> int:
        return int(np.sum(self.rejected))
    
    @property
    def all_rejected(self) -> bool:
        return bool(np.all(self.rejected))
    
    @property
    def is_sequential(self) -> bool:
        """Whether this is a sequential testing result."""
        if self.stop_analysis is None:
            return False
        return self.stop_analysis > 1


class GraphicalProcedure:
    """
    Implements the graphical approach for multiple testing procedures.
    
    Supports both standard (single-stage, n_analyses=1) and sequential (multi-stage,
    n_analyses>1) group sequential designs.
    
    The procedure sequentially tests hypotheses and propagates alpha upon rejection.
    For sequential designs, nominal levels are computed at each analysis from the
    *current* weights and the hypothesis-specific spending functions.
    
    Parameters
    ----------
    weights : np.ndarray
        Initial alpha allocation weights, shape (m,). Must sum to 1.
    transitions : np.ndarray
        Transition matrix, shape (m, m). G[i,j] is the fraction of alpha 
        transferred from i to j when i is rejected. Diagonal must be 0.
    alpha : float
        Overall family-wise error rate (FWER) to control. Default is 0.025.
    n_analyses : int, optional
        Number of planned analyses (including final). Default is 1 (standard procedure).
    information_fractions : List[np.ndarray], optional
        List of information fractions at each analysis, one per hypothesis (length m).
        Each element should be an array of shape (n_analyses,) with values that:
        - Are non-decreasing (cannot decrease across analyses)
        - Reach 1.0 at some analysis
        - Stay at 1.0 after reaching it
        If None, uses equally spaced fractions for all hypotheses (default for n_analyses=1 is [1.0]).
    spending_function : List, optional
        List of alpha spending functions, one per hypothesis (length m).
        Each element can be:
                - Any callable with signature: f(t: float, alpha: float) -> float
                    (this includes `SpendingFunction` instances from `trial_optimizer.spending_functions`)
                If None, uses O'Brien-Fleming for all hypotheses.

    Example
    -------
    >>> # Standard 3-hypothesis example
    >>> weights = np.array([0.5, 0.25, 0.25])
    >>> transitions = np.array([
    ...     [0.0, 0.5, 0.5],
    ...     [1.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0]
    ... ])
    >>> proc = GraphicalProcedure(weights, transitions, alpha=0.025)
    >>> result = proc.test(p_values=np.array([0.01, 0.02, 0.03]))
    >>> 
    >>> # Sequential 2-hypothesis example with 2 analyses
    >>> proc_seq = GraphicalProcedure(
    ...     weights=np.array([0.5, 0.5]),
    ...     transitions=np.array([[0, 1], [1, 0]]),
    ...     alpha=0.025,
    ...     n_analyses=2,
    ...     spending_function=[OBrienFleming(), Pocock()]
    ... )
    """
    
    def __init__(
        self, 
        weights: np.ndarray, 
        transitions: np.ndarray, 
        alpha: float = 0.025,
        n_analyses: int = 1,
        information_fractions: Optional[List[np.ndarray]] = None,
        spending_function: Optional[List[Callable[[float, float], float]]] = None,
    ):
        weights_arr, transitions_arr, m = prepare_weights_transitions(weights, transitions)
        self.weights = weights_arr
        self.transitions = transitions_arr
        self.alpha = alpha
        self.m = m  # Number of hypotheses
        self.n_analyses = n_analyses
        self._spending_funcs: Optional[List[Callable[[float, float], float]]] = None

        self.information_fractions = prepare_information_fractions(
            information_fractions,
            m=self.m,
            n_analyses=self.n_analyses,
        )

        spending_funcs = prepare_spending_functions(spending_function, m=self.m)

        # Store parsed spending callables for runtime boundary computation.
        # Nominal levels are computed by first allocating alpha to hypotheses
        # (alpha_i = alpha * current_weight_i) and then applying the hypothesis-specific
        # spending functions at the current information fraction.
        self._spending_funcs = spending_funcs

        # Expose sequential design inputs for simulation/inspection
        self.spending_function = spending_function
        
        self._validate()
    
    def _validate(self):
        """Validate the graphical procedure parameters."""
        # Check dimensions
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {self.weights.shape}")
        if self.transitions.shape != (self.m, self.m):
            raise ValueError(
                f"transitions must be ({self.m}, {self.m}), got {self.transitions.shape}"
            )
        
        # Check weights sum to 1 (with tolerance)
        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1, got {np.sum(self.weights)}")
        
        # Check non-negativity
        if np.any(self.weights < -1e-10):
            raise ValueError("weights must be non-negative")
        if np.any(self.transitions < -1e-10):
            raise ValueError("transitions must be non-negative")
        
        # Check diagonal is zero
        if not np.allclose(np.diag(self.transitions), 0, atol=1e-10):
            raise ValueError("Diagonal of transitions must be zero (no self-loops)")
        
        # Check row sums <= 1
        row_sums = np.sum(self.transitions, axis=1)
        if np.any(row_sums > 1 + 1e-6):
            raise ValueError(f"Transition row sums must be <= 1, got max {row_sums.max()}")
    
    def test(self, p_values, stop_early: bool = True, return_graphs: bool = False) -> TestResult:
        """
        Apply the (possibly sequential) graphical testing procedure.
        
        Parameters
        ----------
        p_values : np.ndarray or List[np.ndarray]
            If `n_analyses == 1`, may provide a single array of p-values, shape (m,).
            If `n_analyses > 1`, provide either:
            - a list of arrays, one per analysis, each shape (m,), or
            - a 2D array of shape (n_analyses, m).

        stop_early : bool
            If True, stops when all hypotheses are rejected at an interim analysis.

        return_graphs : bool
            If True, includes per-analysis snapshots of the graph state in
            `TestResult.graphs_by_analysis`. Defaults to False to keep results lean.
        
        Returns
        -------
        TestResult
            Unified result object. Sequential metadata fields are populated.
        """
        p_values_by_analysis = prepare_p_values(
            p_values,
            m=self.m,
            n_analyses_planned=self.n_analyses,
        )

        # Unified Algorithm 1 engine (Maurer & Bretz for sequential; reduces to Bretz et al. for n_analyses=1)
        current_weights = self.weights.copy()
        current_transitions = self.transitions.copy()
        active = np.ones(self.m, dtype=bool)
        rejected = np.zeros(self.m, dtype=bool)
        rejection_analysis = np.zeros(self.m, dtype=int)
        rejection_order: List[Tuple[int, int]] = []
        graphs_by_analysis: Optional[List[dict]] = [] if return_graphs else None
        stopped_early_flag = False
        n_provided_analyses = len(p_values_by_analysis)
        final_analysis_idx = n_provided_analyses - 1

        for t in range(n_provided_analyses):
            pv = p_values_by_analysis[t]

            if graphs_by_analysis is not None:
                graphs_by_analysis.append({
                    'weights': current_weights.copy(),
                    'transitions': current_transitions.copy(),
                    'active': active.copy(),
                })

            while True:
                nominal_alpha = self.get_nominal_alpha(t, current_weights)
                can_reject = active & (pv <= nominal_alpha)
                if not np.any(can_reject):
                    break

                candidates = np.where(can_reject)[0]
                idx = int(candidates[np.argmin(pv[candidates])])

                rejected[idx] = True
                active[idx] = False
                rejection_analysis[idx] = t + 1
                rejection_order.append((idx, int(t)))

                # Update weights and transitions (Algorithm 1, Step 3)
                weight_to_propagate = float(current_weights[idx])
                current_weights[idx] = 0.0
                for j in range(self.m):
                    if active[j]:
                        current_weights[j] += weight_to_propagate * current_transitions[idx, j]

                G_new = current_transitions.copy()
                for i in range(self.m):
                    if active[i]:
                        for j in range(self.m):
                            if active[j] and i != j:
                                numerator = current_transitions[i, j] + current_transitions[i, idx] * current_transitions[idx, j]
                                denominator = 1 - current_transitions[i, idx] * current_transitions[idx, i]
                                if denominator > 1e-10:
                                    G_new[i, j] = numerator / denominator
                                else:
                                    G_new[i, j] = 0

                G_new[idx, :] = 0
                G_new[:, idx] = 0
                current_transitions = G_new

            if stop_early and np.all(rejected):
                stopped_early_flag = True
                final_analysis_idx = t
                break

        return TestResult(
            rejected=rejected,
            rejection_analysis=rejection_analysis,
            rejection_order=rejection_order,
            stopped_early=stopped_early_flag,
            stop_analysis=final_analysis_idx + 1,
            graphs_by_analysis=graphs_by_analysis,
        )
    
    def get_nominal_alpha(self, analysis_idx: int, weights: np.ndarray) -> np.ndarray:
        """
        Compute nominal significance levels for each hypothesis at given analysis.
        
        Parameters
        ----------
        analysis_idx : int
            Current analysis index (0-based)
        weights : np.ndarray
            Current weights for active hypotheses
        
        Returns
        -------
        np.ndarray
            Nominal alpha levels for each hypothesis
        """
        if self._spending_funcs is None:
            raise RuntimeError(
                "Internal error: spending functions not initialized. "
                "This should not happen unless the object was partially constructed."
            )

        # Correct ordering for sequential graphical testing:
        # 1) allocate alpha to each hypothesis using current weights
        # 2) apply hypothesis-specific spending function to compute the incremental spend at this look
        nominal = np.zeros(self.m, dtype=np.float64)
        for h in range(self.m):
            alpha_h = float(self.alpha * weights[h])
            if alpha_h <= 0:
                nominal[h] = 0.0
                continue

            t_curr = float(self.information_fractions[h][analysis_idx])
            if analysis_idx == 0:
                cum_prev = 0.0
            else:
                t_prev = float(self.information_fractions[h][analysis_idx - 1])
                # Avoid calling arbitrary user-provided spending functions at t=0.
                cum_prev = 0.0 if t_prev <= 0.0 else float(self._spending_funcs[h](t_prev, alpha_h))

            cum_curr = float(self._spending_funcs[h](t_curr, alpha_h))
            inc = cum_curr - cum_prev
            nominal[h] = 0.0 if inc < 0 else inc

        return nominal
    
    def test_batch(self, p_values_batch: np.ndarray) -> List[TestResult]:
        """
        Apply the graphical testing procedure to multiple sets of p-values.
        
        Parameters
        ----------
        p_values_batch : np.ndarray
            Array of p-values, shape (n_samples, m).
        
        Returns
        -------
        List[TestResult]
            List of TestResult objects, one for each sample.
        """
        return [self.test(pv) for pv in p_values_batch]
    
    def get_rejection_matrix(self, p_values_batch: np.ndarray) -> np.ndarray:
        """
        Get a boolean matrix of rejections for a batch of p-values.
        
        Parameters
        ----------
        p_values_batch : np.ndarray
            Array of p-values, shape (n_samples, m).
        
        Returns
        -------
        np.ndarray
            Boolean array, shape (n_samples, m), where True indicates rejection.
        """
        results = self.test_batch(p_values_batch)
        return np.array([r.rejected for r in results])
    
    def copy(self) -> "GraphicalProcedure":
        """Create a copy of this graphical procedure."""
        return GraphicalProcedure(
            weights=self.weights.copy(),
            transitions=self.transitions.copy(),
            alpha=self.alpha
        )
    
    def __repr__(self) -> str:
        if self.n_analyses == 1:
            return (
                f"GraphicalProcedure(m={self.m}, alpha={self.alpha}, "
                f"weights={self.weights.round(3)}, transitions=\n{self.transitions.round(3)})"
            )
        else:
            return (
                f"GraphicalProcedure(m={self.m}, alpha={self.alpha}, "
                f"n_analyses={self.n_analyses}, "
                f"weights={self.weights.round(3)})"
            )


class BonferroniProcedure(GraphicalProcedure):
    """
    Bonferroni procedure (equal weights, no propagation).
    
    The most conservative procedure - alpha is split equally among hypotheses
    with no propagation when hypotheses are rejected.
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    alpha : float
        Overall FWER to control. Default is 0.025.
    
    Example
    -------
    >>> proc = BonferroniProcedure(m=3, alpha=0.025)
    >>> # Each hypothesis tested at 0.025/3 = 0.00833
    """
    
    def __init__(self, m: int, alpha: float = 0.025):
        weights = np.ones(m) / m
        transitions = np.zeros((m, m))
        super().__init__(weights, transitions, alpha)


class HolmProcedure(GraphicalProcedure):
    """
    Holm procedure (equal weights, equal propagation to remaining).
    
    A uniformly more powerful procedure than Bonferroni. When a hypothesis
    is rejected, its alpha is distributed equally among remaining hypotheses.
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    alpha : float
        Overall FWER to control. Default is 0.025.
    
    Example
    -------
    >>> proc = HolmProcedure(m=3, alpha=0.025)
    >>> # Starts with each at 0.025/3, propagates equally on rejection
    """
    
    def __init__(self, m: int, alpha: float = 0.025):
        weights = np.ones(m) / m
        transitions = np.ones((m, m)) / (m - 1)
        np.fill_diagonal(transitions, 0)
        super().__init__(weights, transitions, alpha)


class FixedSequenceProcedure(GraphicalProcedure):
    """
    Fixed-sequence procedure.
    
    Tests hypotheses in a pre-specified order, passing all alpha to the
    next hypothesis upon rejection. Stops at the first non-rejection.
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    sequence : List[int], optional
        Order of testing (0-indexed). Default is [0, 1, ..., m-1].
    alpha : float
        Overall FWER to control. Default is 0.025.
    
    Example
    -------
    >>> # Test in order H1 -> H2 -> H3
    >>> proc = FixedSequenceProcedure(m=3, alpha=0.025)
    >>> # Or specify custom order
    >>> proc = FixedSequenceProcedure(m=3, sequence=[2, 0, 1], alpha=0.025)
    """
    
    def __init__(
        self,
        m: int,
        sequence: Optional[List[int]] = None,
        alpha: float = 0.025
    ):
        if sequence is None:
            sequence = list(range(m))
        
        weights = np.zeros(m)
        weights[sequence[0]] = 1.0
        
        transitions = np.zeros((m, m))
        for i in range(len(sequence) - 1):
            transitions[sequence[i], sequence[i + 1]] = 1.0
        
        super().__init__(weights, transitions, alpha)


class FallbackProcedure(GraphicalProcedure):
    """
    Fallback procedure.
    
    Allows flexible initial weighting with sequential propagation through
    hypotheses. Alpha passes to the next hypothesis in sequence upon rejection.
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    initial_weights : np.ndarray, optional
        Initial alpha weights. Default is equal weights. Must sum to 1.
    alpha : float
        Overall FWER to control. Default is 0.025.
    
    Example
    -------
    >>> # Equal initial weights
    >>> proc = FallbackProcedure(m=3, alpha=0.025)
    >>> # Custom initial weights
    >>> proc = FallbackProcedure(
    ...     m=3,
    ...     initial_weights=np.array([0.5, 0.3, 0.2]),
    ...     alpha=0.025
    ... )
    """
    
    def __init__(
        self,
        m: int,
        initial_weights: Optional[np.ndarray] = None,
        alpha: float = 0.025
    ):
        if initial_weights is None:
            initial_weights = np.ones(m) / m
        
        transitions = np.zeros((m, m))
        for i in range(m - 1):
            transitions[i, i + 1] = 1.0
        
        super().__init__(initial_weights, transitions, alpha)

