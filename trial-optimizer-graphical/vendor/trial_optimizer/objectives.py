"""Objective definitions for optimizing graphical procedures.

The refactored architecture separates objectives into two components:

1. **Success Function**: Defines what "success" means for each hypothesis
   - Default: rejection (marginal power)
   - Can be customized for gating (e.g., H3 success depends on H1)
   
2. **Weights**: Combines per-hypothesis success into a scalar objective
   - Allows flexible prioritization of endpoints
   - Enables weighted combinations

This design makes objectives more composable and explicit about what
is being optimized.

Core classes:
- SuccessFunction: Defines per-hypothesis success (shape: [m])
- Objective: Combines success with weights → scalar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLikeBool = Union[np.ndarray, Sequence[bool]]


class SuccessFunction:
    """Defines per-hypothesis success metrics.
    
    Returns a vector of length m representing the success of each hypothesis.
    For standard power optimization, success = P(reject H_i).
    For gated scenarios, success can depend on other hypotheses.
    """
    
    requires_rejections: bool = False
    
    def evaluate(
        self,
        *,
        marginal_power: np.ndarray,
        rejections: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns success vector of shape (m,)."""
        raise NotImplementedError
    
    def evaluate_soft(self, soft_reject):
        """Returns success vector from soft rejections.
        
        Args:
            soft_reject: torch.Tensor of shape (n_samples, m)
            
        Returns:
            torch.Tensor of shape (m,) with per-hypothesis success
        """
        raise NotImplementedError


class Objective:
    """Base objective interface.
    
    Combines per-hypothesis success with weights to produce a scalar objective.
    
    Objectives can be evaluated:
    - exactly from simulated boolean rejections (numpy)
    - approximately from differentiable soft rejections (torch)
    """

    requires_rejections: bool = False

    def evaluate(
        self,
        *,
        marginal_power: np.ndarray,
        rejections: Optional[np.ndarray] = None,
    ) -> float:
        """Returns scalar objective value."""
        raise NotImplementedError

    def evaluate_soft(self, soft_reject):
        """Evaluate from soft rejection probs.

        `soft_reject` is expected to be a torch.Tensor of shape (n_samples, m)
        with values in [0, 1]. We intentionally avoid importing torch at module
        import time so the core package remains importable without torch.
        
        Returns:
            Scalar objective value (torch.Tensor with shape ())
        """

        raise NotImplementedError


# ============================================================================
# Success Functions: Define per-hypothesis success
# ============================================================================

@dataclass(frozen=True)
class MarginalRejection(SuccessFunction):
    """Default: success = P(reject H_i) for each hypothesis."""
    
    def evaluate(
        self,
        *,
        marginal_power: np.ndarray,
        rejections: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return marginal_power
    
    def evaluate_soft(self, soft_reject):
        # Returns per-hypothesis rejection probabilities: shape (m,)
        return soft_reject.mean(dim=0)


@dataclass(frozen=True)
class GatedSuccess(SuccessFunction):
    """Gated success: H_i success may depend on other hypotheses.
    
    Example: H3 only counts as success if H1 is also rejected.
    
    Args:
        dependencies: dict mapping hypothesis index to list of required hypothesis indices
                     e.g., {2: [0]} means H3 requires H1
        fallback_to_marginal: if True, use marginal success when dependencies not met
    """
    
    dependencies: Dict[int, List[int]]
    fallback_to_marginal: bool = False
    
    requires_rejections: bool = True
    
    def evaluate(
        self,
        *,
        marginal_power: np.ndarray,
        rejections: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if rejections is None:
            raise ValueError("GatedSuccess requires rejections for evaluation")
        
        rej = np.asarray(rejections, dtype=bool)
        if rej.ndim != 2:
            raise ValueError(f"rejections must be 2D (n_sim, m), got shape {rej.shape}")
        n, m = rej.shape
        
        success = np.zeros(m, dtype=np.float64)
        
        for i in range(m):
            if i in self.dependencies:
                # Check if dependencies are met
                deps = self.dependencies[i]
                deps_met = np.all(rej[:, deps], axis=1)  # Shape: (n,)
                gated_success = rej[:, i] & deps_met
                success[i] = np.mean(gated_success)
            else:
                # No dependencies: use marginal
                success[i] = marginal_power[i]
        
        return success
    
    def evaluate_soft(self, soft_reject):
        import torch
        
        n, m = soft_reject.shape
        success = torch.zeros(m, dtype=soft_reject.dtype, device=soft_reject.device)
        
        for i in range(m):
            if i in self.dependencies:
                # Soft AND: multiply probabilities
                deps = self.dependencies[i]
                deps_prob = torch.prod(soft_reject[:, deps], dim=1)  # P(all deps rejected)
                gated_prob = soft_reject[:, i] * deps_prob  # P(H_i and deps)
                success[i] = gated_prob.mean()
            else:
                # No dependencies: use marginal
                success[i] = soft_reject[:, i].mean()
        
        return success


# ============================================================================
# Objective Functions: Combine success with weights
# ============================================================================

@dataclass(frozen=True)
class WeightedSuccess(Objective):
    """Weighted sum of per-hypothesis success: sum_i w_i * success_i.
    
    This is the primary objective for most use cases.
    
    Args:
        success_fn: SuccessFunction defining per-hypothesis success
        weights: weight for each hypothesis
        normalize_weights: if True, normalize weights to sum to 1
    """
    
    success_fn: SuccessFunction
    weights: np.ndarray
    normalize_weights: bool = True
    
    def __post_init__(self):
        # Validate and normalize weights
        object.__setattr__(self, 'weights', np.asarray(self.weights, dtype=np.float64))
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {self.weights.shape}")
        if self.normalize_weights:
            s = float(np.sum(self.weights))
            if s <= 0:
                raise ValueError("weights must sum to a positive value")
            object.__setattr__(self, 'weights', self.weights / s)
    
    @property
    def requires_rejections(self) -> bool:  # type: ignore[override]
        return self.success_fn.requires_rejections
    
    def evaluate(
        self,
        *,
        marginal_power: np.ndarray,
        rejections: Optional[np.ndarray] = None,
    ) -> float:
        success = self.success_fn.evaluate(
            marginal_power=marginal_power,
            rejections=rejections,
        )
        
        if success.shape != self.weights.shape:
            raise ValueError(f"success shape {success.shape} does not match weights shape {self.weights.shape}")
        
        return float(np.dot(self.weights, success))
    
    def evaluate_soft(self, soft_reject):
        import torch
        
        success = self.success_fn.evaluate_soft(soft_reject)  # Shape: (m,)
        weights_tensor = torch.tensor(self.weights, dtype=soft_reject.dtype, device=soft_reject.device)
        
        return torch.dot(weights_tensor, success)
