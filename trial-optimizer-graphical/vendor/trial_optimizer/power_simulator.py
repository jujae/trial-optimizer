"""
Power Simulation for Clinical Trial Hypotheses

Simulates p-values and computes power metrics for graphical testing procedures.
Based on the simulation methodology in Zhan et al. (2022).

The simulation generates correlated test statistics under specified effect sizes
and then converts them to p-values.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, List, Union, Callable
from dataclasses import dataclass

from .graphical_procedure import GraphicalProcedure, TestResult


def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """
    Find the nearest positive definite matrix to A.
    
    Uses the method from Higham (1988).
    """
    # Check if already positive definite
    try:
        np.linalg.cholesky(A)
        return A
    except np.linalg.LinAlgError:
        pass
    
    # Symmetrize
    B = (A + A.T) / 2
    
    # Compute the symmetric polar factor
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    
    # Get the nearest PSD matrix
    A_psd = (B + H) / 2
    A_psd = (A_psd + A_psd.T) / 2
    
    # Iteratively ensure positive definiteness
    k = 0
    while True:
        try:
            np.linalg.cholesky(A_psd)
            break
        except np.linalg.LinAlgError:
            # Add small value to diagonal
            min_eig = np.min(np.real(np.linalg.eigvals(A_psd)))
            A_psd += (-min_eig * (1 + 1e-6) + 1e-8) * np.eye(A.shape[0])
            k += 1
            if k > 100:
                # Fallback: just add enough to make it work
                A_psd = A_psd + 0.01 * np.eye(A.shape[0])
                break
    
    # Ensure diagonal is exactly 1 for correlation matrix
    d = np.sqrt(np.diag(A_psd))
    A_psd = A_psd / np.outer(d, d)
    
    # Final check and fix
    try:
        np.linalg.cholesky(A_psd)
    except np.linalg.LinAlgError:
        # Last resort: blend with identity
        A_psd = 0.99 * A_psd + 0.01 * np.eye(A.shape[0])
    
    return A_psd


@dataclass
class PowerResult:
    """Results from power simulation."""
    
    # Per-hypothesis metrics
    marginal_power: np.ndarray  # P(reject H_i) for each hypothesis
    
    # Joint metrics
    disjunctive_power: float  # P(reject at least one hypothesis)
    conjunctive_power: float  # P(reject all hypotheses)
    
    # Expected number of rejections
    expected_rejections: float
    
    # Sample size used in simulation
    n_simulations: int
    
    # Sequential-specific metrics (optional)
    early_stop_rate: Optional[float] = None  # Proportion of trials stopping early
    
    # Custom power metrics (if any)
    custom_power: Optional[dict] = None
    
    def __repr__(self) -> str:
        return (
            f"PowerResult(\n"
            f"  marginal_power={self.marginal_power.round(4)},\n"
            f"  disjunctive_power={self.disjunctive_power:.4f},\n"
            f"  conjunctive_power={self.conjunctive_power:.4f},\n"
            f"  expected_rejections={self.expected_rejections:.4f},\n"
            f"  n_simulations={self.n_simulations}\n"
            f")"
        )


class PowerSimulator:
    """
    Simulates power for graphical testing procedures.
    
    This class generates correlated test statistics under a multivariate
    normal distribution, converts them to p-values, and applies a graphical
    testing procedure to compute power.
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    correlation : np.ndarray or float
        Correlation matrix (m x m) or scalar for equicorrelated structure.
    effect_sizes : np.ndarray
        Non-centrality parameters (effect sizes) for each hypothesis.
        Under H_0: effect = 0, under H_1: effect > 0.
    one_sided : bool
        Whether to use one-sided tests (default True for superiority trials).
    seed : int, optional
        Random seed for reproducibility.
    
    Example
    -------
    >>> simulator = PowerSimulator(
    ...     m=3,
    ...     correlation=0.5,  # equicorrelated
    ...     effect_sizes=np.array([0.3, 0.25, 0.2]),
    ... )
    >>> power = simulator.compute_power(procedure, n_simulations=10000)
    """
    
    def __init__(
        self,
        m: int,
        correlation: Union[np.ndarray, float],
        effect_sizes: np.ndarray,
        one_sided: bool = True,
        seed: Optional[int] = None
    ):
        self.m = m
        self.effect_sizes = np.asarray(effect_sizes, dtype=np.float64)
        self.one_sided = one_sided
        self.rng = np.random.default_rng(seed)
        
        # Build correlation matrix
        if np.isscalar(correlation):
            self.correlation = (1 - correlation) * np.eye(m) + correlation * np.ones((m, m))
        else:
            self.correlation = np.asarray(correlation, dtype=np.float64)
        
        # Ensure correlation matrix is positive definite
        self.correlation = _nearest_positive_definite(self.correlation)
        
        # Compute the Cholesky decomposition for generating correlated samples
        self._chol = np.linalg.cholesky(self.correlation)
        
        self._validate()
    
    def _validate(self):
        """Validate parameters."""
        if len(self.effect_sizes) != self.m:
            raise ValueError(
                f"effect_sizes must have length {self.m}, got {len(self.effect_sizes)}"
            )
        if self.correlation.shape != (self.m, self.m):
            raise ValueError(
                f"correlation must be ({self.m}, {self.m}), got {self.correlation.shape}"
            )
        # Check positive semi-definiteness
        eigvals = np.linalg.eigvalsh(self.correlation)
        if np.any(eigvals < -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite")
    
    def generate_test_statistics(self, n_samples: int) -> np.ndarray:
        """
        Generate correlated test statistics.
        
        Under the alternative, test statistics follow:
        Z ~ N(effect_sizes, correlation)
        
        Parameters
        ----------
        n_samples : int
            Number of simulation samples.
        
        Returns
        -------
        np.ndarray
            Array of test statistics, shape (n_samples, m).
        """
        # Generate standard normal samples and correlate them
        z_standard = self.rng.standard_normal((n_samples, self.m))
        z_correlated = z_standard @ self._chol.T
        
        # Shift by effect sizes (non-centrality parameters)
        z_shifted = z_correlated + self.effect_sizes
        
        return z_shifted
    
    def generate_p_values(self, n_samples: int) -> np.ndarray:
        """
        Generate p-values from simulated test statistics.
        
        Parameters
        ----------
        n_samples : int
            Number of simulation samples.
        
        Returns
        -------
        np.ndarray
            Array of p-values, shape (n_samples, m).
        """
        z_stats = self.generate_test_statistics(n_samples)
        
        if self.one_sided:
            # One-sided: P(Z > z) under H_0
            p_values = 1 - stats.norm.cdf(z_stats)
        else:
            # Two-sided: 2 * P(Z > |z|) under H_0
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
        
        return p_values
    
    def _generate_sequential_p_values(
        self,
        information_fractions
    ) -> List[np.ndarray]:
        """
        Generate correlated p-values across sequential analyses.
        
        For group sequential designs, test statistics at different analyses
        are correlated because they share accumulated data. The correlation
        between statistics at times t1 < t2 is sqrt(t1/t2).
        
        Parameters
        ----------
        information_fractions : List[np.ndarray]
            List of m arrays of shape (n_analyses,): Per-hypothesis information fractions.
        
        Returns
        -------
        List[np.ndarray]
            List of p-value arrays, one per analysis. Each array has shape (m,).
        """
        # Convert list format to array format
        if isinstance(information_fractions, list):
            # List of arrays: stack to create (m, n_analyses)
            info_frac = np.vstack(information_fractions)
            n_analyses = info_frac.shape[1]
        else:
            # Already an array (m, n_analyses)
            info_frac = information_fractions
            n_analyses = info_frac.shape[1]
        
        # Build correlation matrix for (n_analyses * m) test statistics
        # With per-hypothesis information fractions, the correlation structure
        # becomes more complex:
        #   - Within analysis (across hypotheses): original correlation matrix
        #   - Across analyses (same hypothesis h): sqrt(t1_h / t2_h)
        #   - Across analyses (different hypotheses h1, h2): 
        #     sqrt(min(t1_h1, t2_h2) / max(t1_h1, t2_h2)) * corr(h1, h2)
        total_dim = n_analyses * self.m
        sequential_corr = np.zeros((total_dim, total_dim))
        
        for k1 in range(n_analyses):
            for k2 in range(n_analyses):
                # Block for analyses k1 and k2
                i1_start = k1 * self.m
                i1_end = (k1 + 1) * self.m
                i2_start = k2 * self.m
                i2_end = (k2 + 1) * self.m
                
                # Compute element-wise correlations
                for h1 in range(self.m):
                    for h2 in range(self.m):
                        t1_h1 = info_frac[h1, k1]
                        t2_h2 = info_frac[h2, k2]
                        
                        # Time correlation for these two measurements
                        time_corr = np.sqrt(min(t1_h1, t2_h2) / max(t1_h1, t2_h2))
                        
                        # Hypothesis correlation
                        hyp_corr = self.correlation[h1, h2]
                        
                        # Combined correlation
                        sequential_corr[i1_start + h1, i2_start + h2] = time_corr * hyp_corr
        
        # Generate correlated standard normal samples
        sequential_chol = np.linalg.cholesky(
            _nearest_positive_definite(sequential_corr)
        )
        z_standard = self.rng.standard_normal(total_dim)
        z_correlated = sequential_chol @ z_standard
        
        # Add effect sizes (scaled by sqrt of information fraction)
        # At time t, test statistic ~ N(effect * sqrt(t), 1)
        p_values_by_analysis = []
        for k in range(n_analyses):
            idx_start = k * self.m
            idx_end = (k + 1) * self.m
            
            # Scale effect sizes by sqrt(information fraction) per hypothesis
            z_k = np.zeros(self.m)
            for h in range(self.m):
                t_h = info_frac[h, k]
                z_k[h] = z_correlated[idx_start + h] + self.effect_sizes[h] * np.sqrt(t_h)
            
            # Convert to p-values
            if self.one_sided:
                p_k = 1 - stats.norm.cdf(z_k)
            else:
                p_k = 2 * (1 - stats.norm.cdf(np.abs(z_k)))
            
            p_values_by_analysis.append(p_k)
        
        return p_values_by_analysis
    
    def compute_power(
        self,
        procedure: GraphicalProcedure,
        n_simulations: int = 10000,
        custom_power_fn: Optional[Callable[[np.ndarray], float]] = None,
        stop_early: bool = True
    ) -> PowerResult:
        """
        Compute power metrics via Monte Carlo simulation.
        
        Parameters
        ----------
        procedure : GraphicalProcedure
            The graphical testing procedure to evaluate (set `n_analyses>1` for sequential designs).
        n_simulations : int
            Number of Monte Carlo simulations.
        custom_power_fn : callable, optional
            Custom function that takes rejection matrix (n_sim, m) and returns 
            either a float (scalar power) or a dict of power metrics.
        stop_early : bool
            For sequential procedures: whether to stop when all hypotheses rejected.
        
        Returns
        -------
        PowerResult
            Power metrics from the simulation.
        """
        # Check if sequential procedure
        is_sequential = bool(getattr(procedure, "n_analyses", 1) > 1)
        
        if is_sequential:
            # Sequential procedure: generate correlated test statistics across analyses
            rejections = np.zeros((n_simulations, self.m), dtype=bool)
            early_stops = 0
            
            for i in range(n_simulations):
                # Generate correlated test statistics for all analyses
                p_values_by_analysis = self._generate_sequential_p_values(
                    procedure.information_fractions
                )
                
                result = procedure.test(p_values_by_analysis, stop_early=stop_early)
                rejections[i] = result.rejected
                if result.stopped_early:
                    early_stops += 1
            
            early_stop_rate = early_stops / n_simulations
        else:
            # Standard procedure: single analysis
            p_values = self.generate_p_values(n_simulations)
            rejections = procedure.get_rejection_matrix(p_values)
            early_stop_rate = None
        
        # Compute power metrics
        marginal_power = np.mean(rejections, axis=0)
        disjunctive_power = np.mean(np.any(rejections, axis=1))
        conjunctive_power = np.mean(np.all(rejections, axis=1))
        expected_rejections = np.mean(np.sum(rejections, axis=1))
        
        # Custom power if provided
        custom_power = None
        if custom_power_fn is not None:
            result = custom_power_fn(rejections)
            if isinstance(result, dict):
                custom_power = result
            else:
                custom_power = {"custom": result}
        
        return PowerResult(
            marginal_power=marginal_power,
            disjunctive_power=disjunctive_power,
            conjunctive_power=conjunctive_power,
            expected_rejections=expected_rejections,
            n_simulations=n_simulations,
            early_stop_rate=early_stop_rate,
            custom_power=custom_power
        )
    
    def compute_power_differentiable(
        self,
        weights: np.ndarray,
        transitions: np.ndarray,
        alpha: float,
        p_values: np.ndarray,
        power_type: str = "disjunctive"
    ) -> float:
        """
        Compute power for given parameters (used in optimization).
        
        This is a simplified version used for optimization that doesn't
        create new GraphicalProcedure objects each time.
        """
        procedure = GraphicalProcedure(weights, transitions, alpha)
        rejections = procedure.get_rejection_matrix(p_values)
        
        if power_type == "disjunctive":
            return float(np.mean(np.any(rejections, axis=1)))
        elif power_type == "conjunctive":
            return float(np.mean(np.all(rejections, axis=1)))
        elif power_type == "expected":
            return float(np.mean(np.sum(rejections, axis=1)))
        elif power_type.startswith("marginal_"):
            idx = int(power_type.split("_")[1])
            return float(np.mean(rejections[:, idx]))
        else:
            raise ValueError(f"Unknown power_type: {power_type}")


class WeightedPowerObjective:
    """
    Defines a weighted power objective for optimization.
    
    The objective is a weighted sum of marginal powers:
    objective = sum_i(lambda_i * P(reject H_i))
    
    This allows prioritizing certain hypotheses over others.
    
    Parameters
    ----------
    lambda_weights : np.ndarray
        Weights for each hypothesis in the objective.
        Higher weight = higher priority in optimization.
    constraint_type : str
        Type of additional constraint:
        - "none": No constraint
        - "min_power": Minimum power constraint for each hypothesis
        - "hierarchical": Enforce hierarchical testing structure
    min_powers : np.ndarray, optional
        Minimum required power for each hypothesis (if constraint_type="min_power").
    """
    
    def __init__(
        self,
        lambda_weights: np.ndarray,
        constraint_type: str = "none",
        min_powers: Optional[np.ndarray] = None
    ):
        self.lambda_weights = np.asarray(lambda_weights)
        self.constraint_type = constraint_type
        self.min_powers = min_powers
    
    def compute_objective(self, marginal_powers: np.ndarray) -> float:
        """Compute the weighted power objective."""
        return float(np.dot(self.lambda_weights, marginal_powers))
    
    def compute_constraint_violation(self, marginal_powers: np.ndarray) -> float:
        """Compute constraint violation (0 if satisfied)."""
        if self.constraint_type == "none":
            return 0.0
        elif self.constraint_type == "min_power":
            if self.min_powers is None:
                return 0.0
            violations = np.maximum(0, self.min_powers - marginal_powers)
            return float(np.sum(violations))
        else:
            return 0.0
    
    def __call__(self, marginal_powers: np.ndarray) -> Tuple[float, float]:
        """Returns (objective, constraint_violation)."""
        return (
            self.compute_objective(marginal_powers),
            self.compute_constraint_violation(marginal_powers)
        )


def generate_scenarios(
    m: int,
    base_effect_sizes: np.ndarray,
    effect_variations: List[np.ndarray],
    correlation: Union[float, np.ndarray] = 0.5,
) -> List[PowerSimulator]:
    """
    Generate multiple simulation scenarios for robust optimization.
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    base_effect_sizes : np.ndarray
        Baseline effect sizes.
    effect_variations : List[np.ndarray]
        List of effect size variations to create scenarios.
    correlation : float or np.ndarray
        Correlation structure.
    
    Returns
    -------
    List[PowerSimulator]
        List of PowerSimulator objects for different scenarios.
    """
    scenarios = []
    
    # Add base scenario
    scenarios.append(PowerSimulator(
        m=m, 
        correlation=correlation, 
        effect_sizes=base_effect_sizes
    ))
    
    # Add variation scenarios
    for variation in effect_variations:
        scenarios.append(PowerSimulator(
            m=m,
            correlation=correlation,
            effect_sizes=variation
        ))
    
    return scenarios
