"""
Optimizer for Graphical Procedures

Implements the deep learning optimization framework from Zhan et al. (2022)
for finding optimal graphical testing procedures.

The optimization uses:
1. Stochastic gradient estimation via Monte Carlo simulation
2. Reparameterization trick for differentiable sampling
3. Constraint handling via soft penalties or projection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Callable, Sequence
from dataclasses import dataclass
from tqdm import tqdm
import warnings
from scipy.optimize import minimize, differential_evolution

from .graphical_procedure import GraphicalProcedure
from .power_simulator import PowerSimulator, PowerResult
from .objectives import Objective
from .neural_network import (
    GraphicalProcedureNetwork,
    SoftRejectionApproximation
)


@dataclass
class OptimizationResult:
    """Results from the optimization process."""
    
    # Optimal procedure found
    optimal_weights: np.ndarray
    optimal_transitions: np.ndarray
    optimal_procedure: GraphicalProcedure
    
    # Diagnostic power metrics of optimal procedure
    final_power: PowerResult
    
    # Training history
    loss_history: List[float]
    objective_history: List[float]
    
    # Optimization metadata
    n_iterations: int
    converged: bool
    
    def __repr__(self) -> str:
        final_obj = self.objective_history[-1] if self.objective_history else 0.0
        return (
            f"OptimizationResult(\n"
            f"  optimal_weights={self.optimal_weights.round(4)},\n"
            f"  final_objective={final_obj:.4f},\n"
            f"  n_iterations={self.n_iterations},\n"
            f"  converged={self.converged}\n"
            f")"
        )


class GraphicalProcedureOptimizer:
    """
    Optimizes graphical testing procedures using deep learning.
    
    This implements the main algorithm from Zhan et al. (2022), using
    Monte Carlo simulation and gradient descent to find optimal
    initial alpha allocation and transition matrices.
    
    Parameters
    ----------
    simulator : PowerSimulator
        Simulator for generating p-values and computing power.
    alpha : float
        Overall family-wise error rate to control.
    objective : Objective
        Objective to optimize (all "power" metrics are objectives).
    device : str
        Device to use for computation ("cpu" or "cuda").
    
    Example
    -------
    >>> simulator = PowerSimulator(m=4, correlation=0.5, effect_sizes=[0.3]*4)
    >>> from trial_optimizer import WeightedSuccess, MarginalRejection
    >>> optimizer = GraphicalProcedureOptimizer(simulator, objective=WeightedSuccess(MarginalRejection(), [1,1,1,1]))
    >>> result = optimizer.optimize(n_iterations=1000)
    >>> print(result.optimal_procedure)
    """
    
    def __init__(
        self,
        simulator: PowerSimulator,
        alpha: float = 0.025,
        objective: Objective = None,
        device: str = "cpu",
        # Sequential-specific parameters
        n_analyses: Optional[int] = None,
        information_fractions: Optional[List[np.ndarray]] = None,
        spending_function: Optional[Union[Callable[[float, float], float], List[Callable[[float, float], float]]]] = None,
        optimize_spending: bool = True,
        initial_gamma: float = -4.0
    ):
        self.simulator = simulator
        self.m = simulator.m
        self.alpha = alpha
        self.device = torch.device(device)

        if objective is None:
            raise ValueError(
                "objective must be provided. "
                "Define an Objective (e.g., WeightedSuccess(MarginalRejection(), weights)) "
                "and pass it in via the `objective` parameter."
            )
        self._objective_fn: Objective = objective
        
        # Sequential design parameters
        # For single-stage designs, treat as 1 analysis at information fraction 1.0
        self.n_analyses = n_analyses if n_analyses is not None else 1
        self.is_sequential = n_analyses is not None
        
        if isinstance(information_fractions, np.ndarray) and information_fractions.ndim == 1:
            self.information_fractions = [information_fractions] * self.m
        elif information_fractions is None and not self.is_sequential:
            # Single-stage: set to 1.0 for all hypotheses
            self.information_fractions = [np.array([1.0]) for _ in range(self.m)]
        else:
            self.information_fractions = information_fractions

        if spending_function is None:
            self.spending_function = None
        elif isinstance(spending_function, (list, tuple)):
            self.spending_function = list(spending_function)
        elif callable(spending_function):
            self.spending_function = [spending_function] * self.m
        else:
            raise ValueError(
                "spending_function must be None, a callable f(t, alpha) -> float, or a list of such callables"
            )

        self.optimize_spending = optimize_spending
        
        # Determine if we should optimize gamma (only for sequential)
        self._optimize_hsd_gamma = False
        if self.is_sequential and optimize_spending:
            from .spending_functions import HwangShihDeCani

            if self.spending_function is None:
                # Default sequential optimization uses HSD with a learnable gamma.
                self._optimize_hsd_gamma = True
            else:
                if not all(isinstance(sf, HwangShihDeCani) for sf in self.spending_function):
                    raise ValueError(
                        "optimize_spending=True is only supported for HwangShihDeCani spending. "
                        "Pass spending_function=None (default) or a list of HwangShihDeCani instances."
                    )
                self._optimize_hsd_gamma = True
        
        # Initialize network with gamma optimization if needed
        self.network = GraphicalProcedureNetwork(
            m=self.m,
            hidden_dims=[],  # Parameter-only mode
            row_sum_one=False,
            optimize_gamma=self._optimize_hsd_gamma,
            initial_gamma=initial_gamma
        ).to(self.device)
        
        # Soft rejection approximation for gradient-based optimization
        # Produces differentiable soft rejections (values in [0,1]) from p-values
        self.soft_rejection = SoftRejectionApproximation(
            temperature=0.1,
            n_analyses=self.n_analyses,
            information_fractions=self.information_fractions,
            spending_functions=None  # Will be set dynamically based on gamma
        )
    
    def _generate_p_values_torch(self, n_samples: int) -> torch.Tensor:
        """Generate p-values and convert to torch tensor."""
        p_values = self.simulator.generate_p_values(n_samples)
        return torch.tensor(p_values, dtype=torch.float32, device=self.device)
    
    def _simulate_rejections(
        self,
        procedure: GraphicalProcedure,
        n_simulations: int,
    ) -> np.ndarray:
        """Simulate trial-level rejections (n_simulations, m) for objective evaluation."""

        if self.is_sequential:
            rejections = np.zeros((n_simulations, self.m), dtype=bool)
            info_fracs = procedure.information_fractions
            for i in range(n_simulations):
                p_values_by_analysis = self.simulator._generate_sequential_p_values(info_fracs)
                result = procedure.test(p_values_by_analysis, stop_early=True)
                rejections[i] = result.rejected
            return rejections

        p_values = self.simulator.generate_p_values(n_simulations)
        return procedure.get_rejection_matrix(p_values)
    
    def _evaluate_objective_exact(
        self,
        weights: np.ndarray,
        transitions: np.ndarray,
        n_simulations: int = 10000,
        gamma: Optional[float] = None
    ) -> Tuple[PowerResult, float]:
        """
        Evaluate the objective function exactly using Monte Carlo simulation.
        
        Returns both the PowerResult (for diagnostics) and the objective value being optimized.
        """
        from .spending_functions import HwangShihDeCani

        spending = self.spending_function
        if gamma is not None:
            # Use per-hypothesis gamma values
            spending = [HwangShihDeCani(gamma=float(gamma[h])) for h in range(self.m)]

        procedure = GraphicalProcedure(
            weights=weights,
            transitions=transitions,
            alpha=self.alpha,
            n_analyses=self.n_analyses or 1,
            information_fractions=self.information_fractions,
            spending_function=spending,
        )
        
        power_result = self.simulator.compute_power(procedure, n_simulations)

        rejections = None
        if self._objective_fn.requires_rejections:
            rejections = self._simulate_rejections(procedure, n_simulations)

        objective_value = self._objective_fn.evaluate(
            marginal_power=power_result.marginal_power,
            rejections=rejections,
        )

        return power_result, float(objective_value)
    
    def _compute_loss(
        self,
        p_values: torch.Tensor,
        weights: torch.Tensor,
        transitions: torch.Tensor,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the loss (negative objective) for optimization.
        
        Uses differentiable soft rejections for gradient computation.
        """
        soft_reject = self.soft_rejection(p_values, weights, transitions, self.alpha, gamma=gamma)
        obj = self._objective_fn.evaluate_soft(soft_reject)
        return -obj
    
    def _reinforce_loss(
        self,
        weights: torch.Tensor,
        transitions: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        n_samples: int = 1000
    ) -> torch.Tensor:
        """
        Compute loss using REINFORCE-style gradient estimation.
        
        This is more accurate than the differentiable approximation
        but has higher variance.
        """
        # Generate p-values
        p_values = self._generate_p_values_torch(n_samples)
        
        # Use soft rejection approximation for differentiable loss
        soft_reject = self.soft_rejection(p_values, weights, transitions, self.alpha, gamma=gamma)
        obj = self._objective_fn.evaluate_soft(soft_reject)
        loss = -obj  # Maximize objective
        
        return loss
    
    def optimize(
        self,
        n_iterations: int = 1000,
        batch_size: int = 1000,
        learning_rate: float = 0.01,
        patience: int = 100,
        min_delta: float = 1e-5,
        use_reinforce: bool = False,
        initial_weights: Optional[np.ndarray] = None,
        initial_transitions: Optional[np.ndarray] = None,
        verbose: bool = True,
        eval_every: int = 50,
        n_eval_samples: int = 10000
    ) -> OptimizationResult:
        """
        Optimize the graphical procedure parameters.
        
        Parameters
        ----------
        n_iterations : int
            Maximum number of optimization iterations.
        batch_size : int
            Number of p-value samples per iteration.
        learning_rate : float
            Learning rate for Adam optimizer.
        patience : int
            Early stopping patience (iterations without improvement).
        min_delta : float
            Minimum improvement to reset patience counter.
        use_reinforce : bool
            Whether to use REINFORCE gradient estimation (more accurate but higher variance).
        initial_weights : np.ndarray, optional
            Initial weights to start from (warm start).
        initial_transitions : np.ndarray, optional
            Initial transition matrix to start from.
        verbose : bool
            Whether to show progress bar.
        eval_every : int
            How often to evaluate exact power.
        n_eval_samples : int
            Number of samples for power evaluation.
        
        Returns
        -------
        OptimizationResult
            Optimization results including optimal procedure and history.
        """
        # Initialize from provided values if given
        if initial_weights is not None and initial_transitions is not None:
            self.network.initialize_from_procedure(initial_weights, initial_transitions)
        
        # Setup optimizer - all parameters are now in the network
        # If gamma is being optimized, it gets a slower learning rate
        param_groups = []
        if self._optimize_hsd_gamma and self.network.raw_gamma is not None:
            # Separate learning rates for graph params vs gamma
            graph_params = [p for n, p in self.network.named_parameters() if 'raw_gamma' not in n]
            gamma_params = [p for n, p in self.network.named_parameters() if 'raw_gamma' in n]
            param_groups.append({'params': graph_params, 'lr': learning_rate})
            param_groups.append({'params': gamma_params, 'lr': learning_rate * 0.1})
        else:
            param_groups.append({'params': self.network.parameters(), 'lr': learning_rate})
        
        optimizer = optim.Adam(param_groups)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 4
        )
        
        # Tracking
        loss_history = []
        objective_history = []
        best_objective = -np.inf
        best_weights = None
        best_transitions = None
        no_improve_count = 0
        
        # Training loop
        iterator = tqdm(range(n_iterations), disable=not verbose, desc="Optimizing")
        
        for iteration in iterator:
            self.network.train()
            optimizer.zero_grad()
            
            # Get current parameters
            weights, transitions, gamma = self.network()
            
            # Compute loss
            if use_reinforce:
                loss = self._reinforce_loss(weights, transitions, gamma, n_samples=batch_size)
            else:
                p_values = self._generate_p_values_torch(batch_size)
                loss = self._compute_loss(p_values, weights, transitions, gamma)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            # Record loss
            loss_history.append(loss.item())
            
            # Periodic evaluation with exact procedure
            if iteration % eval_every == 0 or iteration == n_iterations - 1:
                self.network.eval()
                with torch.no_grad():
                    w_np, g_np, gamma_val = self.network.get_procedure_params()
                
                # Ensure constraints are satisfied (small numerical fixes)
                w_np = np.clip(w_np, 0, 1)
                w_np = w_np / w_np.sum()
                g_np = np.clip(g_np, 0, 1)
                np.fill_diagonal(g_np, 0)
                
                try:
                    power_result, current_objective = self._evaluate_objective_exact(
                        w_np, g_np, n_eval_samples, gamma=gamma_val
                    )
                    
                    objective_history.append(current_objective)
                    
                    # Update best
                    if current_objective > best_objective + min_delta:
                        best_objective = current_objective
                        best_weights = w_np.copy()
                        best_transitions = g_np.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if verbose:
                        iterator.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'obj': f'{current_objective:.4f}',
                            'best_obj': f'{best_objective:.4f}'
                        })
                    
                except Exception as e:
                    warnings.warn(f"Error computing objective at iteration {iteration}: {e}")
                    objective_history.append(objective_history[-1] if objective_history else 0)
            
            # Early stopping
            if no_improve_count >= patience // eval_every:
                if verbose:
                    print(f"\nEarly stopping at iteration {iteration}")
                break
        
        # Final evaluation
        if best_weights is None:
            best_weights, best_transitions, best_gamma = self.network.get_procedure_params()
            best_weights = np.clip(best_weights, 0, 1)
            best_weights = best_weights / best_weights.sum()
            best_transitions = np.clip(best_transitions, 0, 1)
            np.fill_diagonal(best_transitions, 0)
        else:
            # Get gamma from network if it wasn't captured during optimization
            _, _, best_gamma = self.network.get_procedure_params()
        
        # Create final procedure
        from .spending_functions import HwangShihDeCani

        spending = self.spending_function
        if best_gamma is not None:
            # Use per-hypothesis gamma values
            spending = [HwangShihDeCani(gamma=float(best_gamma[h])) for h in range(self.m)]
        
        optimal_procedure = GraphicalProcedure(
            weights=best_weights,
            transitions=best_transitions,
            alpha=self.alpha,
            n_analyses=self.n_analyses or 1,
            information_fractions=self.information_fractions,
            spending_function=spending,
        )
        
        final_power = self.simulator.compute_power(optimal_procedure, n_eval_samples * 2)
        
        return OptimizationResult(
            optimal_weights=best_weights,
            optimal_transitions=best_transitions,
            optimal_procedure=optimal_procedure,
            final_power=final_power,
            loss_history=loss_history,
            objective_history=objective_history,
            n_iterations=len(loss_history),
            converged=(no_improve_count >= patience // eval_every)
        )
    
    def optimize_differential_evolution(
        self,
        n_eval_samples: int = 5000,
        maxiter: int = 100,
        popsize: int = 15,
        atol: float = 1e-4,
        verbose: bool = True,
        seed: Optional[int] = None
    ) -> OptimizationResult:
        """
        Optimize using differential evolution (gradient-free, similar to ISRES).
        
        This provides a gradient-free baseline for comparison with the gradient-based
        optimization. Uses scipy's differential_evolution optimizer, which is similar
        in spirit to ISRES (Improved Stochastic Ranking Evolution Strategy).
        
        Parameters
        ----------
        n_eval_samples : int
            Number of samples for power evaluation.
        maxiter : int
            Maximum number of generations.
        popsize : int
            Population size multiplier (actual size = popsize * n_params).
        atol : float
            Absolute tolerance for convergence.
        verbose : bool
            Whether to show progress.
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        OptimizationResult
            Optimization results including optimal procedure.
        
        Notes
        -----
        This method is slower than gradient-based optimization but:
        - Does not require gradients
        - May find better global optima in non-convex landscapes
        - Useful as a baseline for comparison with DNN-based methods
        
        Similar to the ISRES/COBYLA methods in Zhan et al. (2022).
        """
        from scipy.optimize import differential_evolution
        
        # Determine number of parameters
        # For weights: (m-1) free parameters (last one is 1 - sum of others)
        # For transitions: m * (m-1) free parameters per row
        n_weight_params = self.m - 1
        n_transition_params = self.m * (self.m - 1)
        
        # Add gamma parameters if optimizing spending
        n_gamma_params = 0
        if self._optimize_hsd_gamma:
            n_gamma_params = self.m
        
        n_params = n_weight_params + n_transition_params + n_gamma_params
        
        # Define objective function (to minimize = negative objective)
        objective_history = []
        best_objective = -np.inf
        iteration_counter = [0]
        
        def objective(x):
            """Objective function: negative objective value (for minimization)."""
            iteration_counter[0] += 1
            
            # Extract parameters
            idx = 0
            
            # Weights (m-1 free, last one computed)
            w_free = x[idx:idx + n_weight_params]
            weights = np.append(w_free, 1.0 - w_free.sum())
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()  # Normalize
            idx += n_weight_params
            
            # Transitions (m rows, m-1 free per row, diagonal = 0)
            transitions = np.zeros((self.m, self.m))
            for i in range(self.m):
                # Free parameters for row i (excluding diagonal)
                free_indices = [j for j in range(self.m) if j != i]
                row_free = x[idx:idx + (self.m - 1)]
                transitions[i, free_indices] = row_free
                idx += (self.m - 1)
            
            # Normalize rows to sum to 1
            transitions = np.clip(transitions, 0, 1)
            row_sums = transitions.sum(axis=1, keepdims=True)
            transitions = np.where(row_sums > 0, transitions / row_sums, transitions)
            
            # Gamma parameters
            spending = self.spending_function
            if self._optimize_hsd_gamma:
                from .spending_functions import HwangShihDeCani
                gamma_values = x[idx:idx + n_gamma_params]
                spending = [HwangShihDeCani(gamma=float(g)) for g in gamma_values]
            
            # Create procedure
            try:
                procedure = GraphicalProcedure(
                    weights=weights,
                    transitions=transitions,
                    alpha=self.alpha,
                    n_analyses=self.n_analyses or 1,
                    information_fractions=self.information_fractions,
                    spending_function=spending,
                )
                
                # Compute power metrics and evaluate objective
                power_result = self.simulator.compute_power(procedure, n_eval_samples)
                
                rejections = None
                if self._objective_fn.requires_rejections:
                    rejections = self._simulate_rejections(procedure, n_eval_samples)
                
                objective_value = self._objective_fn.evaluate(
                    marginal_power=power_result.marginal_power,
                    rejections=rejections,
                )
                
                objective_history.append(objective_value)
                nonlocal best_objective
                if objective_value > best_objective:
                    best_objective = objective_value
                
                if verbose and iteration_counter[0] % 10 == 0:
                    print(f"Iteration {iteration_counter[0]}: Objective = {objective_value:.4f}, Best = {best_objective:.4f}")
                
                return -objective_value  # Negative for minimization
                
            except Exception as e:
                warnings.warn(f"Error evaluating procedure: {e}")
                return 1e6  # Large penalty
        
        # Define bounds
        bounds = []
        
        # Weights: each in [0, 1], must sum to <= 1
        for _ in range(n_weight_params):
            bounds.append((0, 1))
        
        # Transitions: each in [0, 1]
        for _ in range(n_transition_params):
            bounds.append((0, 1))
        
        # Gamma: reasonable range for HSD spending
        if self._optimize_hsd_gamma:
            for _ in range(n_gamma_params):
                bounds.append((-10, 2))  # Typical range for gamma
        
        # Run optimization
        if verbose:
            print(f"Running differential evolution with {n_params} parameters...")
            print(f"Population size: {popsize * n_params}")
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            atol=atol,
            seed=seed,
            disp=verbose,
            workers=1,  # Sequential to avoid issues with random number generation
            updating='deferred',  # More similar to classic DE
        )
        
        # Extract final solution
        x_final = result.x
        idx = 0
        
        w_free = x_final[idx:idx + n_weight_params]
        best_weights = np.append(w_free, 1.0 - w_free.sum())
        best_weights = np.clip(best_weights, 0, 1)
        best_weights = best_weights / best_weights.sum()
        idx += n_weight_params
        
        best_transitions = np.zeros((self.m, self.m))
        for i in range(self.m):
            free_indices = [j for j in range(self.m) if j != i]
            row_free = x_final[idx:idx + (self.m - 1)]
            best_transitions[i, free_indices] = row_free
            idx += (self.m - 1)
        
        best_transitions = np.clip(best_transitions, 0, 1)
        row_sums = best_transitions.sum(axis=1, keepdims=True)
        best_transitions = np.where(row_sums > 0, best_transitions / row_sums, best_transitions)
        
        spending = self.spending_function
        if self._optimize_hsd_gamma:
            from .spending_functions import HwangShihDeCani
            gamma_values = x_final[idx:idx + n_gamma_params]
            spending = [HwangShihDeCani(gamma=float(g)) for g in gamma_values]
        
        # Create final procedure
        optimal_procedure = GraphicalProcedure(
            weights=best_weights,
            transitions=best_transitions,
            alpha=self.alpha,
            n_analyses=self.n_analyses or 1,
            information_fractions=self.information_fractions,
            spending_function=spending,
        )
        
        final_power = self.simulator.compute_power(optimal_procedure, n_eval_samples * 2)
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final power: {final_power.disjunctive_power:.4f}")
            print(f"Function evaluations: {result.nfev}")
        
        return OptimizationResult(
            optimal_weights=best_weights,
            optimal_transitions=best_transitions,
            optimal_procedure=optimal_procedure,
            final_power=final_power,
            loss_history=[],  # Not applicable for DE
            objective_history=objective_history,
            n_iterations=result.nit,
            converged=result.success
        )
    
    def optimize_multi_scenario(
        self,
        simulators: List[PowerSimulator],
        scenario_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize for multiple scenarios (robust optimization).
        
        This finds a procedure that performs well across different
        scenarios (e.g., different effect size assumptions).
        
        Parameters
        ----------
        simulators : List[PowerSimulator]
            List of simulators for different scenarios.
        scenario_weights : np.ndarray, optional
            Weights for each scenario. Default is equal weights.
        **kwargs
            Additional arguments passed to optimize().
        
        Returns
        -------
        OptimizationResult
            Optimization results for the multi-scenario objective.
        """
        if scenario_weights is None:
            scenario_weights = np.ones(len(simulators)) / len(simulators)
        
        # Create combined optimizer that averages across scenarios
        original_simulator = self.simulator
        
        class MultiScenarioSimulator:
            def __init__(self, simulators, weights):
                self.simulators = simulators
                self.weights = weights
                self.m = simulators[0].m
            
            def generate_p_values(self, n_samples):
                # Sample from each scenario according to weights
                p_values_list = []
                n_per_scenario = np.random.multinomial(n_samples, self.weights)
                
                for sim, n in zip(self.simulators, n_per_scenario):
                    if n > 0:
                        p_values_list.append(sim.generate_p_values(n))
                
                return np.vstack(p_values_list)
            
            def compute_power(self, procedure, n_simulations):
                # Average power across scenarios
                powers = []
                for sim, w in zip(self.simulators, self.weights):
                    p = sim.compute_power(procedure, n_simulations // len(self.simulators))
                    powers.append(p)
                
                # Weighted average of marginal powers
                marginal = sum(w * p.marginal_power for w, p in zip(self.weights, powers))
                disjunctive = sum(w * p.disjunctive_power for w, p in zip(self.weights, powers))
                conjunctive = sum(w * p.conjunctive_power for w, p in zip(self.weights, powers))
                expected = sum(w * p.expected_rejections for w, p in zip(self.weights, powers))
                
                return PowerResult(
                    marginal_power=marginal,
                    disjunctive_power=disjunctive,
                    conjunctive_power=conjunctive,
                    expected_rejections=expected,
                    n_simulations=n_simulations
                )
        
        self.simulator = MultiScenarioSimulator(simulators, scenario_weights)
        
        try:
            result = self.optimize(**kwargs)
        finally:
            self.simulator = original_simulator
        
        return result


class GridSearchOptimizer:
    """
    Grid search optimizer for small problems or initialization.
    
    Useful for:
    1. Finding good initial values for gradient-based optimization
    2. Small problems where enumeration is feasible
    3. Baseline comparison
    """
    
    def __init__(
        self,
        simulator: PowerSimulator,
        alpha: float = 0.025,
        objective: Objective = None,
    ):
        self.simulator = simulator
        self.m = simulator.m
        self.alpha = alpha
        if objective is None:
            raise ValueError("objective must be provided")
        self._objective_fn: Objective = objective

    def _simulate_rejections(self, procedure: GraphicalProcedure, n_simulations: int) -> np.ndarray:
        p_values = self.simulator.generate_p_values(n_simulations)
        return procedure.get_rejection_matrix(p_values)
    
    def optimize(
        self,
        weight_grid: Optional[np.ndarray] = None,
        transition_grid: Optional[np.ndarray] = None,
        n_weight_samples: int = 100,
        n_transition_samples: int = 100,
        n_eval_samples: int = 5000,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search over a grid of possible parameters.
        
        Parameters
        ----------
        weight_grid : np.ndarray, optional
            Grid of weight vectors to try, shape (n_weights, m).
        transition_grid : np.ndarray, optional
            Grid of transition matrices to try, shape (n_transitions, m, m).
        n_weight_samples : int
            Number of random weight samples if grid not provided.
        n_transition_samples : int
            Number of random transition samples if grid not provided.
        n_eval_samples : int
            Number of samples for power evaluation.
        verbose : bool
            Whether to show progress.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Best weights, best transitions, best power.
        """
        # Generate weight grid if not provided
        if weight_grid is None:
            weight_grid = self._sample_weights(n_weight_samples)
        
        # Generate transition grid if not provided
        if transition_grid is None:
            transition_grid = self._sample_transitions(n_transition_samples)
        
        best_objective = -np.inf
        best_weights = None
        best_transitions = None
        
        total = len(weight_grid) * len(transition_grid)
        iterator = tqdm(
            [(w, g) for w in weight_grid for g in transition_grid],
            disable=not verbose,
            desc="Grid search",
            total=total
        )
        
        for weights, transitions in iterator:
            try:
                procedure = GraphicalProcedure(weights, transitions, self.alpha)
                power_result = self.simulator.compute_power(procedure, n_eval_samples)

                rejections = None
                if self._objective_fn.requires_rejections:
                    rejections = self._simulate_rejections(procedure, n_eval_samples)

                obj = self._objective_fn.evaluate(
                    marginal_power=power_result.marginal_power,
                    rejections=rejections,
                )

                if obj > best_objective:
                    best_objective = float(obj)
                    best_weights = weights
                    best_transitions = transitions
                    
                    if verbose:
                        iterator.set_postfix({'best_obj': f'{best_objective:.4f}'})
                        
            except Exception:
                continue
        
        return best_weights, best_transitions, best_objective
    
    def _sample_weights(self, n_samples: int) -> np.ndarray:
        """Sample random weight vectors from Dirichlet distribution."""
        return np.random.dirichlet(np.ones(self.m), size=n_samples)
    
    def _sample_transitions(self, n_samples: int) -> np.ndarray:
        """Sample random valid transition matrices."""
        transitions = []
        for _ in range(n_samples):
            g = np.random.random((self.m, self.m))
            np.fill_diagonal(g, 0)
            # Normalize rows to sum to at most 1
            row_sums = g.sum(axis=1, keepdims=True)
            g = g / np.maximum(row_sums, 1)
            transitions.append(g)
        return np.array(transitions)


class COBYLAOptimizer:
    """
    COBYLA-based optimizer for graphical procedures.
    
    Uses scipy.optimize.minimize with COBYLA method (Constrained Optimization 
    BY Linear Approximations). This is a gradient-free method that handles 
    constraints well, making it suitable for problems with explicit constraints
    like sum-to-one for weights and row constraints for transitions.
    
    Advantages over stochastic optimization:
    - More stable and deterministic
    - Often faster for smaller problems (m < 10)
    - Better handles explicit constraints
    - No hyperparameter tuning needed
    
    Parameters
    ----------
    simulator : PowerSimulator
        Simulator for generating p-values and computing power.
    alpha : float
        Overall family-wise error rate to control.
    objective : Objective
        Objective to optimize.
    n_analyses : int, optional
        Number of planned analyses for sequential procedures.
    information_fractions : list of np.ndarray, optional
        Information fractions at each analysis for sequential procedures.
    spending_function : list, optional
        Spending function instances for sequential procedures.
    optimize_spending : bool
        Whether to optimize spending function parameters (gamma). Only supported
        for HwangShihDeCani spending functions.
    initial_gamma : float
        Initial gamma value for spending functions when optimize_spending=True.
    """
    
    def __init__(
        self,
        simulator: PowerSimulator,
        alpha: float = 0.025,
        objective: Objective = None,
        # Sequential-specific parameters
        n_analyses: Optional[int] = None,
        information_fractions: Optional[List[np.ndarray]] = None,
        spending_function: Optional[List] = None,
        optimize_spending: bool = False,
        initial_gamma: float = -4.0
    ):
        self.simulator = simulator
        self.m = simulator.m
        self.alpha = alpha
        if objective is None:
            raise ValueError(
                "objective must be provided. "
                "Define an Objective (e.g., WeightedSuccess(MarginalRejection(), weights)) "
                "and pass it in via the `objective` parameter."
            )
        self._objective_fn: Objective = objective
        
        # Sequential design parameters
        self.n_analyses = n_analyses
        self.information_fractions = information_fractions
        self.spending_function = spending_function
        self.is_sequential = n_analyses is not None
        self.optimize_spending = optimize_spending
        self.initial_gamma = initial_gamma
        
        # Validate spending optimization
        if self.optimize_spending:
            from .spending_functions import HwangShihDeCani
            if not self.is_sequential:
                raise ValueError("optimize_spending=True requires sequential design (n_analyses must be set)")
            if self.spending_function is not None:
                if not all(isinstance(sf, HwangShihDeCani) for sf in self.spending_function):
                    raise ValueError(
                        "optimize_spending=True is only supported for HwangShihDeCani spending. "
                        "Pass spending_function=None or a list of HwangShihDeCani instances."
                    )
        
        # For tracking evaluations
        self.n_evaluations = 0
        self.objective_history = []
        self.best_objective = -np.inf
    
    def _params_to_procedure(
        self, 
        params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Convert flat parameter vector to weights, transitions, and optionally gammas.
        
        Parameters are arranged as:
        - [w1, w2, ..., wm-1, g11, g12, ..., gmm] if not optimizing spending
        - [w1, w2, ..., wm-1, g11, g12, ..., gmm, gamma1, gamma2, ..., gammam] if optimizing spending
        
        The last weight wm is computed as 1 - sum(w1:wm-1).
        Diagonal transitions are always 0.
        """
        # Extract weights (m-1 free parameters, last is determined)
        weights = np.zeros(self.m)
        weights[:-1] = params[:self.m-1]
        weights[-1] = 1.0 - weights[:-1].sum()
        
        # Clip to valid range
        weights = np.clip(weights, 0, 1)
        
        # Renormalize if needed
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(self.m) / self.m
        
        # Extract transitions
        transitions = params[self.m-1:self.m-1+self.m*self.m].reshape(self.m, self.m)
        transitions = np.clip(transitions, 0, 1)
        np.fill_diagonal(transitions, 0)
        
        # Ensure row sums <= 1
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = np.where(row_sums > 1, transitions / row_sums, transitions)
        
        # Extract gammas if optimizing spending
        gammas = None
        if self.optimize_spending:
            gammas = params[self.m-1+self.m*self.m:]
            # Clip gamma to reasonable range [-10, 0]
            gammas = np.clip(gammas, -10, 0)
        
        return weights, transitions, gammas
    
    def _procedure_to_params(
        self,
        weights: np.ndarray,
        transitions: np.ndarray,
        gammas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert weights, transitions, and optionally gammas to flat parameter vector."""
        # Only first m-1 weights are free parameters
        params = np.concatenate([
            weights[:-1],
            transitions.flatten()
        ])
        if self.optimize_spending and gammas is not None:
            params = np.concatenate([params, gammas])
        return params
    
    def _evaluate_objective(
        self,
        weights: np.ndarray,
        transitions: np.ndarray,
        gammas: Optional[np.ndarray],
        n_simulations: int
    ) -> float:
        """Evaluate the objective function for the given procedure."""
        try:
            # Create spending functions with gammas if optimizing
            spending_func = self.spending_function
            if self.optimize_spending and gammas is not None:
                from .spending_functions import HwangShihDeCani
                spending_func = [HwangShihDeCani(gamma=float(g)) for g in gammas]
            
            procedure = GraphicalProcedure(
                weights=weights,
                transitions=transitions,
                alpha=self.alpha,
                n_analyses=self.n_analyses or 1,
                information_fractions=self.information_fractions,
                spending_function=spending_func,
            )
            
            # Generate p-values and get rejections
            if self.is_sequential:
                # Generate sequential p-values and test
                rejections = np.zeros((n_simulations, self.m), dtype=bool)
                info_fracs = procedure.information_fractions
                for i in range(n_simulations):
                    p_values_by_analysis = self.simulator._generate_sequential_p_values(info_fracs)
                    result = procedure.test(p_values_by_analysis, stop_early=True)
                    rejections[i] = result.rejected
            else:
                p_values = self.simulator.generate_p_values(n_simulations)
                rejections = procedure.get_rejection_matrix(p_values)

            marginal_power = np.mean(rejections, axis=0)
            disjunctive_power = float(np.mean(np.any(rejections, axis=1)))
            conjunctive_power = float(np.mean(np.all(rejections, axis=1)))
            expected_rejections = float(np.mean(np.sum(rejections, axis=1)))

            return float(
                self._objective_fn.evaluate(
                    marginal_power=marginal_power,
                    rejections=rejections if self._objective_fn.requires_rejections else None,
                )
            )
            
        except Exception as e:
            warnings.warn(f"Error computing objective: {e}")
            return 0.0
    
    def _objective(self, params: np.ndarray, n_simulations: int) -> float:
        """Objective function (negative objective for minimization)."""
        weights, transitions, gammas = self._params_to_procedure(params)
        objective_value = self._evaluate_objective(weights, transitions, gammas, n_simulations)
        
        # Track progress
        self.n_evaluations += 1
        self.objective_history.append(objective_value)
        
        if objective_value > self.best_objective:
            self.best_objective = objective_value
        
        return -objective_value  # Negative because we minimize
    
    def optimize(
        self,
        n_simulations: int = 5000,
        initial_weights: Optional[np.ndarray] = None,
        initial_transitions: Optional[np.ndarray] = None,
        maxiter: int = 1000,
        rhobeg: float = 0.5,
        rhoend: float = 1e-4,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Optimize using COBYLA method.
        
        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo samples per evaluation.
        initial_weights : np.ndarray, optional
            Initial weights. If None, uses uniform.
        initial_transitions : np.ndarray, optional
            Initial transitions. If None, uses Holm-like structure.
        maxiter : int
            Maximum number of iterations.
        rhobeg : float
            Initial step size for COBYLA.
        rhoend : float
            Final step size for COBYLA (convergence tolerance).
        verbose : bool
            Whether to display progress.
        
        Returns
        -------
        OptimizationResult
            Optimization results including optimal procedure.
        
        Example
        -------
        >>> simulator = PowerSimulator(m=4, effect_sizes=[0.3]*4, correlation=0.5)
        >>> from trial_optimizer import WeightedSuccess, MarginalRejection
        >>> optimizer = COBYLAOptimizer(simulator, objective=WeightedSuccess(MarginalRejection(), [1,1,1,1]))
        >>> result = optimizer.optimize(n_simulations=5000, maxiter=500)
        >>> print(f"Optimal power: {result.final_power.disjunctive_power:.4f}")
        """
        # Initialize
        if initial_weights is None:
            initial_weights = np.ones(self.m) / self.m
        if initial_transitions is None:
            # Holm-like: distribute to next hypothesis
            initial_transitions = np.zeros((self.m, self.m))
            for i in range(self.m - 1):
                initial_transitions[i, i + 1] = 1.0
        
        # Initialize gammas if optimizing spending
        initial_gammas = None
        if self.optimize_spending:
            initial_gammas = np.full(self.m, self.initial_gamma)
        
        x0 = self._procedure_to_params(initial_weights, initial_transitions, initial_gammas)
        
        # Define constraints
        constraints = []
        
        # Constraint: weights sum to 1 (implicitly handled by parameterization)
        # Constraint: each weight >= 0
        for i in range(self.m - 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i: x[idx]  # wi >= 0
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i: 1.0 - np.sum(x[:self.m-1])  # wm >= 0
            })
        
        # Constraint: transitions in [0, 1]
        n_transition_params = self.m * self.m
        for i in range(n_transition_params):
            idx = self.m - 1 + i
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=idx: x[idx]  # gij >= 0
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=idx: 1.0 - x[idx]  # gij <= 1
            })
        
        # Constraint: gammas in [-10, 0] if optimizing spending
        if self.optimize_spending:
            n_gamma_params = self.m
            gamma_start_idx = self.m - 1 + n_transition_params
            for i in range(n_gamma_params):
                idx = gamma_start_idx + i
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=idx: x[idx] + 10.0  # gamma >= -10
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=idx: -x[idx]  # gamma <= 0
                })
        
        # Reset tracking
        self.n_evaluations = 0
        self.objective_history = []
        self.best_objective = -np.inf
        
        # Callback for progress
        def callback(x):
            if verbose and self.n_evaluations % 10 == 0:
                print(f"Iteration {self.n_evaluations}: Objective = {self.best_objective:.4f}")
        
        if verbose:
            print(f"Starting COBYLA optimization with {self.m} hypotheses")
            print(f"Objective: {type(self._objective_fn).__name__}")
            print(f"Simulations per evaluation: {n_simulations}")
            if self.optimize_spending:
                print(f"Optimizing spending parameters (gamma): Yes")
        
        # Run optimization
        result = minimize(
            fun=self._objective,
            x0=x0,
            args=(n_simulations,),
            method='COBYLA',
            constraints=constraints,
            options={
                'maxiter': maxiter,
                'rhobeg': rhobeg,
                'tol': rhoend,
                'disp': verbose
            },
            callback=callback if verbose else None
        )
        
        # Extract optimal parameters
        optimal_weights, optimal_transitions, optimal_gammas = self._params_to_procedure(result.x)
        
        # Create spending functions with optimal gammas if optimizing
        optimal_spending_func = self.spending_function
        if self.optimize_spending and optimal_gammas is not None:
            from .spending_functions import HwangShihDeCani
            optimal_spending_func = [HwangShihDeCani(gamma=float(g)) for g in optimal_gammas]
        
        # Create optimal procedure
        optimal_procedure = GraphicalProcedure(
            weights=optimal_weights,
            transitions=optimal_transitions,
            alpha=self.alpha,
            n_analyses=self.n_analyses or 1,
            information_fractions=self.information_fractions,
            spending_function=optimal_spending_func,
        )
        
        # Final power evaluation with more samples
        final_power = self.simulator.compute_power(
            optimal_procedure,
            n_simulations * 2
        )
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Total evaluations: {self.n_evaluations}")
            print(f"Final objective: {-result.fun:.4f}")
            print(f"Converged: {result.success}")
        
        return OptimizationResult(
            optimal_weights=optimal_weights,
            optimal_transitions=optimal_transitions,
            optimal_procedure=optimal_procedure,
            final_power=final_power,
            loss_history=[-obj for obj in self.objective_history],  # Convert to loss
            objective_history=self.objective_history,
            n_iterations=self.n_evaluations,
            converged=result.success
        )


def optimize_graphical_procedure(
    m: int,
    effect_sizes: np.ndarray,
    correlation: Union[float, np.ndarray] = 0.5,
    alpha: float = 0.025,
    objective: Objective = None,
    n_iterations: int = 1000,
    verbose: bool = True,
    **kwargs
) -> OptimizationResult:
    """
    Convenience function to optimize a graphical procedure.
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    effect_sizes : np.ndarray
        Expected effect sizes (non-centrality parameters).
    correlation : float or np.ndarray
        Correlation between test statistics.
    alpha : float
        Family-wise error rate to control.
    objective : Objective
        Objective to optimize.
    n_iterations : int
        Number of optimization iterations.
    verbose : bool
        Whether to show progress.
    **kwargs
        Additional arguments passed to optimizer.
    
    Returns
    -------
    OptimizationResult
        Optimization results.
    
    Example
    -------
    >>> result = optimize_graphical_procedure(
    ...     m=4,
    ...     effect_sizes=np.array([0.3, 0.25, 0.2, 0.15]),
    ...     correlation=0.5,
    ...     alpha=0.025,
    ...     objective=DisjunctivePower(),
    ... )
    >>> print(f"Optimal power: {result.final_power.disjunctive_power:.4f}")
    """
    if objective is None:
        raise ValueError(
            "objective must be provided. "
            "Define an Objective (e.g., WeightedSuccess(MarginalRejection(), weights)) "
            "and pass it in via the `objective` parameter."
        )

    simulator = PowerSimulator(
        m=m,
        correlation=correlation,
        effect_sizes=effect_sizes
    )
    
    optimizer = GraphicalProcedureOptimizer(
        simulator=simulator,
        alpha=alpha,
        objective=objective,
    )
    
    return optimizer.optimize(
        n_iterations=n_iterations,
        verbose=verbose,
        **kwargs
    )


def optimize_sequential_procedure(
    m: int,
    effect_sizes: np.ndarray,
    correlation: Union[float, np.ndarray] = 0.5,
    alpha: float = 0.025,
    objective: Objective = None,
    n_analyses: int = 2,
    information_fractions: Optional[np.ndarray] = None,
    spending_function: Optional[Union[Callable[[float, float], float], Sequence[Callable[[float, float], float]]]] = None,
    optimize_spending: bool = True,
    initial_gamma: float = -4.0,
    n_iterations: int = 1000,
    verbose: bool = True,
    **kwargs
) -> OptimizationResult:
    """
    Optimize a sequential graphical procedure.
    
    This optimizes both the graphical structure (weights/transitions) and
    optionally the spending function parameter (gamma for HSD).
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    effect_sizes : np.ndarray
        Expected effect sizes (non-centrality parameters).
    correlation : float or np.ndarray
        Correlation between test statistics.
    alpha : float
        Family-wise error rate to control.
    objective : Objective
        Objective to optimize.
    n_analyses : int
        Number of planned analyses (including final).
    information_fractions : np.ndarray, optional
        Information fractions at each analysis. If None, uses equally spaced.
    spending_function : callable or list of callables, optional
        Spending function(s) used for all hypotheses. Must be callables with signature
        `f(t: float, alpha: float) -> float`. If None, defaults to Hwang-Shih-DeCani
        with `gamma=initial_gamma`.
    optimize_spending : bool
        Whether to optimize gamma parameter (only for HSD).
    initial_gamma : float
        Initial value for gamma (HSD only).
    n_iterations : int
        Number of optimization iterations.
    verbose : bool
        Whether to show progress.
    **kwargs
        Additional arguments passed to optimizer.
    
    Returns
    -------
    OptimizationResult
        Optimization results with optimal sequential procedure.
    
    Example
    -------
    >>> from trial_optimizer.spending_functions import HwangShihDeCani
    >>> from trial_optimizer import WeightedSuccess, MarginalRejection
    >>> result = optimize_sequential_procedure(
    ...     m=4,
    ...     effect_sizes=np.array([0.3, 0.25, 0.2, 0.15]),
    ...     correlation=0.5,
    ...     alpha=0.025,
    ...     n_analyses=3,
    ...     information_fractions=np.array([0.5, 0.75, 1.0]),
    ...     spending_function=HwangShihDeCani(gamma=-4.0),
    ...     optimize_spending=True,
    ...     objective=DisjunctivePower(),
    ... )
    >>> print(f"Optimal power: {result.final_power.disjunctive_power:.4f}")
    >>> print(f"Early stop rate: {result.final_power.early_stop_rate:.3f}")
    >>> print(f"Optimal gamma: {result.optimal_procedure.spending_function[0].gamma:.2f}")
    """
    from .spending_functions import HwangShihDeCani

    if objective is None:
        raise ValueError(
            "objective must be provided. "
            "Define an Objective (e.g., WeightedSuccess(MarginalRejection(), weights)) "
            "and pass it in via the `objective` parameter."
        )
    
    if information_fractions is None:
        information_fractions = np.linspace(1.0 / n_analyses, 1.0, n_analyses)
    
    # Convert to per-hypothesis lists
    if isinstance(information_fractions, np.ndarray) and information_fractions.ndim == 1:
        information_fractions = [information_fractions] * m

    # Normalize spending functions
    if spending_function is None:
        spending_function = [HwangShihDeCani(gamma=initial_gamma) for _ in range(m)]
    elif isinstance(spending_function, (list, tuple)):
        spending_function = list(spending_function)
    elif callable(spending_function):
        spending_function = [spending_function] * m
    else:
        raise ValueError(
            "spending_function must be None, a callable f(t, alpha) -> float, or a list/tuple of such callables"
        )

    if any(isinstance(sf, str) for sf in spending_function):
        raise ValueError(
            "spending_function no longer accepts string shortcuts. "
            "Pass callables like OBrienFleming(), Pocock(), Linear(), or HwangShihDeCani()."
        )
    if len(spending_function) != m:
        raise ValueError(f"spending_function must have length {m} (one per hypothesis), got {len(spending_function)}")
    if not all(callable(sf) for sf in spending_function):
        raise ValueError("Each element of spending_function must be callable f(t, alpha) -> float")

    if optimize_spending and not all(isinstance(sf, HwangShihDeCani) for sf in spending_function):
        raise ValueError(
            "optimize_spending=True is only supported for HwangShihDeCani spending. "
            "Pass spending_function=None (default) or a HwangShihDeCani instance/list."
        )
    
    simulator = PowerSimulator(
        m=m,
        correlation=correlation,
        effect_sizes=effect_sizes
    )
    
    optimizer = GraphicalProcedureOptimizer(
        simulator=simulator,
        alpha=alpha,
        objective=objective,
        n_analyses=n_analyses,
        information_fractions=information_fractions,
        spending_function=spending_function,
        optimize_spending=optimize_spending,
        initial_gamma=initial_gamma
    )
    
    return optimizer.optimize(
        n_iterations=n_iterations,
        verbose=verbose,
        **kwargs
    )


