"""
Alpha Spending Functions for Group Sequential Designs

Provides common and customizable spending functions for controlling Type I error
in group sequential trials with multiple interim analyses.

Class-based API:
    >>> spending = HwangShihDeCani(gamma=-2)
    >>> alpha_spent = spending(t=0.5, alpha=0.025)
    >>> params = spending.get_parameters()  # {'gamma': -2}

All spending functions satisfy:
- f(0, alpha) = 0 (no alpha spent at start)
- f(1, alpha) = alpha (all alpha spent at end)
- Monotonically increasing in t

## Creating Custom Spending Functions

To create your own spending function:

Example:
    >>> from trial_optimizer.spending_functions import SpendingFunction
    >>> 
    >>> class MySpending(SpendingFunction):
    ...     def __init__(self, param=1.0):
    ...         self.param = param
    ...     
    ...     def __call__(self, t: float, alpha: float) -> float:
    ...         if t <= 0:
    ...             return 0.0
    ...         if t >= 1.0:
    ...             return alpha
    ...         return alpha * (t ** self.param)
    ...     
    ...     def get_parameters(self):
    ...         return {'param': self.param}
    >>> 
    >>> # Use in sequential procedure
    >>> from trial_optimizer import GraphicalProcedure
    >>> proc = GraphicalProcedure(
    ...     weights=..., transitions=...,
    ...     n_analyses=2,
    ...     spending_function=[MySpending(param=2)] * m
    ... )

Requirements:
1. Subclass SpendingFunction
2. Implement __call__(t: float, alpha: float) -> float
3. Optionally implement get_parameters() for optimization
4. Optionally implement set_parameters(**params) for parameter updates
5. Ensure f(0, alpha) = 0
6. Ensure f(1, alpha) = alpha
7. Ensure monotonically increasing in t
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Union


class SpendingFunction(ABC):
    """
    Abstract base class for alpha spending functions.
    
    All spending functions must implement __call__ and return cumulative
    alpha spent at given information fraction.
    """
    
    @abstractmethod
    def __call__(self, t: float, alpha: float) -> float:
        """
        Compute cumulative alpha spent at information fraction t.
        
        Parameters
        ----------
        t : float
            Information fraction (0 to 1)
        alpha : float
            Total alpha to spend
        
        Returns
        -------
        float
            Cumulative alpha spent at information fraction t
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return tunable parameters for optimization.
        
        Returns
        -------
        dict
            Dictionary of parameter names and values
        """
        return {}

    def cumulative_alpha(self, information_fractions: Union[np.ndarray, list], alpha: float) -> np.ndarray:
        """Compute cumulative alpha spent at each information fraction.

        Parameters
        ----------
        information_fractions : array-like
            Information fractions for each analysis (values in [0, 1]).
        alpha : float
            Total alpha to spend.

        Returns
        -------
        np.ndarray
            Cumulative alpha spent at each analysis, shape (n_analyses,).
        """
        info = np.asarray(information_fractions, dtype=np.float64)
        return np.asarray([self(float(t), alpha) for t in info], dtype=np.float64)

    def incremental_alpha(self, information_fractions: Union[np.ndarray, list], alpha: float) -> np.ndarray:
        """Compute incremental alpha spent at each analysis.

        This converts cumulative spending f(t, alpha) into per-analysis increments.

        Parameters
        ----------
        information_fractions : array-like
            Information fractions for each analysis (values in [0, 1]).
        alpha : float
            Total alpha to spend.

        Returns
        -------
        np.ndarray
            Incremental alpha at each analysis, shape (n_analyses,).
        """
        cumulative = self.cumulative_alpha(information_fractions, alpha)
        cum_with_zero = np.concatenate([[0.0], cumulative])
        return np.diff(cum_with_zero)

    def set_parameters(self, **params) -> None:
        """Set tunable parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    def __repr__(self) -> str:
        params = self.get_parameters()
        if params:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            return f"{self.__class__.__name__}({params_str})"
        return f"{self.__class__.__name__}()"


class OBrienFleming(SpendingFunction):
    """
    O'Brien-Fleming spending function.
    
    Conservative early, spending more alpha at later analyses. This is the
    most conservative spending function, spending very little alpha at early
    analyses and reserving most for the final analysis.
    
    Formula: 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
    where Phi is the standard normal CDF and z_{alpha/2} = Phi^(-1)(1 - alpha/2).
    
    References
    ----------
    Lan, K. K., & DeMets, D. L. (1983). Discrete sequential boundaries for
    clinical trials. Biometrika, 70(3), 659-663.
    
    O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure
    for clinical trials. Biometrics, 549-556.
    """
    
    def __call__(self, t: float, alpha: float) -> float:
        from scipy.stats import norm
        if t <= 0:
            return 0.0
        if t >= 1.0:
            return alpha
        z_alpha_2 = norm.ppf(1 - alpha / 2)
        return 2 * (1 - norm.cdf(z_alpha_2 / np.sqrt(t)))


class Pocock(SpendingFunction):
    """
    Pocock spending function.
    
    Balanced spending across analyses. This function provides more balanced
    spending across analyses compared to O'Brien-Fleming, making it easier
    to stop early for efficacy.
    
    Formula: alpha * log(1 + (e - 1) * t)
    
    References
    ----------
    Pocock, S. J. (1977). Group sequential methods in the design and
    analysis of clinical trials. Biometrika, 64(2), 191-199.
    """
    
    def __call__(self, t: float, alpha: float) -> float:
        if t <= 0:
            return 0.0
        if t >= 1.0:
            return alpha
        return alpha * np.log(1 + (np.e - 1) * t)


class Linear(SpendingFunction):
    """
    Linear spending function.
    
    Alpha spent proportional to information fraction. This is the most
    aggressive spending function, useful when early stopping for efficacy
    is desirable.
    
    Formula: alpha * t
    """
    
    def __call__(self, t: float, alpha: float) -> float:
        return alpha * min(max(t, 0.0), 1.0)


class HwangShihDeCani(SpendingFunction):
    """
    Hwang-Shih-DeCani spending function with parameter gamma.
    
    Flexible spending function that includes O'Brien-Fleming (gamma → -∞)
    and Pocock-like (gamma → 0) as special cases. The gamma parameter can
    be optimized for specific trial designs.
    
    Parameters
    ----------
    gamma : float, optional
        Shape parameter (default -4).
        - gamma < 0: conservative early (similar to O'Brien-Fleming)
        - gamma = 0: linear spending
        - gamma > 0: aggressive early (similar to Pocock)
    
    Formula
    -------
    For gamma ≠ 0: alpha * (1 - exp(-gamma * t)) / (1 - exp(-gamma))
    For gamma = 0: alpha * t (linear)
    
    References
    ----------
    Hwang, I. K., Shih, W. J., & De Cani, J. S. (1990). Group sequential
    designs using a family of type I error probability spending functions.
    Statistics in medicine, 9(12), 1439-1445.
    """
    
    def __init__(self, gamma: float = -4):
        self.gamma = gamma
    
    def __call__(self, t: float, alpha: float) -> float:
        if t <= 0:
            return 0.0
        if t >= 1.0:
            return alpha
        
        if abs(self.gamma) < 1e-10:
            # gamma ≈ 0: linear spending
            return alpha * t
        else:
            # General case
            return alpha * (1 - np.exp(-self.gamma * t)) / (1 - np.exp(-self.gamma))
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'gamma': self.gamma}
