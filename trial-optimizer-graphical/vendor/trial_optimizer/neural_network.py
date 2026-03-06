"""
Neural Network for Optimizing Graphical Procedures

Implements the deep learning architecture from Zhan et al. (2022) for
learning optimal initial alpha allocation (weights) and transition matrix
for graphical testing procedures.

Key challenges:
1. Output constraints: weights must sum to 1, transitions must be valid
2. Non-differentiable testing procedure: use surrogate objectives
3. Complex optimization landscape: careful initialization and training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import numpy as np


class SoftmaxConstraint(nn.Module):
    """Ensure outputs sum to 1 using softmax."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


class TransitionMatrixLayer(nn.Module):
    """
    Produces a valid transition matrix from unconstrained parameters.
    
    Constraints enforced:
    1. G_ii = 0 (no self-loops)
    2. G_ij >= 0 (non-negative)
    3. sum_j(G_ij) <= 1 (row sums at most 1)
    
    The output can optionally be further constrained to have row sums exactly 1.
    """
    
    def __init__(self, m: int, row_sum_one: bool = False):
        """
        Parameters
        ----------
        m : int
            Number of hypotheses (matrix will be m x m).
        row_sum_one : bool
            If True, enforce row sums = 1. If False, row sums <= 1.
        """
        super().__init__()
        self.m = m
        self.row_sum_one = row_sum_one
        
        # Create mask to zero out diagonal
        self.register_buffer(
            'diag_mask', 
            1 - torch.eye(m)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform unconstrained parameters to valid transition matrix.
        
        Parameters
        ----------
        x : torch.Tensor
            Unconstrained parameters, shape (..., m, m) or (..., m*(m-1)).
        
        Returns
        -------
        torch.Tensor
            Valid transition matrix, shape (..., m, m).
        """
        # Reshape if flattened (only off-diagonal elements)
        if x.shape[-1] == self.m * (self.m - 1):
            # Unflatten to matrix, inserting zeros on diagonal
            batch_shape = x.shape[:-1]
            x_matrix = torch.zeros(*batch_shape, self.m, self.m, device=x.device, dtype=x.dtype)
            
            # Fill off-diagonal elements
            idx = 0
            for i in range(self.m):
                for j in range(self.m):
                    if i != j:
                        x_matrix[..., i, j] = x[..., idx]
                        idx += 1
            x = x_matrix
        
        # Apply sigmoid to ensure non-negativity and bound to [0, 1]
        x = torch.sigmoid(x)
        
        # Zero out diagonal
        x = x * self.diag_mask
        
        if self.row_sum_one:
            # Normalize rows to sum to 1
            # Add small epsilon to avoid division by zero
            row_sums = x.sum(dim=-1, keepdim=True) + 1e-10
            x = x / row_sums
        else:
            # Ensure row sums <= 1 by scaling if necessary
            row_sums = x.sum(dim=-1, keepdim=True)
            scale = torch.clamp(row_sums, min=1.0)
            x = x / scale
        
        return x


class GraphicalProcedureNetwork(nn.Module):
    """
    Neural network that outputs optimal graphical procedure parameters.
    
    The network can be:
    1. Parameter-only: Directly learns the procedure parameters
    2. Context-aware: Takes scenario information as input
    
    Architecture based on Zhan et al. (2022).
    
    Parameters
    ----------
    m : int
        Number of hypotheses.
    hidden_dims : List[int]
        Dimensions of hidden layers. Empty list for parameter-only mode.
    input_dim : int, optional
        Input dimension for context-aware mode. If None, parameter-only mode.
    dropout : float
        Dropout rate for regularization.
    row_sum_one : bool
        Whether transition matrix rows should sum to exactly 1.
    
    Example
    -------
    >>> # Parameter-only mode (learns fixed procedure)
    >>> net = GraphicalProcedureNetwork(m=4, hidden_dims=[])
    >>> weights, transitions = net()
    
    >>> # Context-aware mode (adapts to input)
    >>> net = GraphicalProcedureNetwork(m=4, hidden_dims=[64, 32], input_dim=10)
    >>> context = torch.randn(batch_size, 10)
    >>> weights, transitions = net(context)
    """
    
    def __init__(
        self,
        m: int,
        hidden_dims: List[int] = [],
        input_dim: Optional[int] = None,
        dropout: float = 0.0,
        row_sum_one: bool = False,
        optimize_gamma: bool = False,
        initial_gamma: float = -4.0
    ):
        super().__init__()
        self.m = m
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.parameter_only = (input_dim is None or len(hidden_dims) == 0)
        self.optimize_gamma = optimize_gamma
        
        # Output dimensions
        self.weight_dim = m
        self.transition_dim = m * (m - 1)  # Off-diagonal elements only
        self.output_dim = self.weight_dim + self.transition_dim
        
        if self.parameter_only:
            # Directly learnable parameters
            self.raw_weights = nn.Parameter(torch.zeros(m))
            self.raw_transitions = nn.Parameter(torch.zeros(m, m))
            
            # Per-hypothesis gamma parameters for HSD spending (if optimizing)
            if optimize_gamma:
                self.raw_gamma = nn.Parameter(torch.full((m,), initial_gamma, dtype=torch.float32))
            else:
                self.raw_gamma = None
        else:
            # Build MLP for context-aware mode
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, self.output_dim))
            self.mlp = nn.Sequential(*layers)
        
        # Constraint layers
        self.weight_constraint = SoftmaxConstraint()
        self.transition_constraint = TransitionMatrixLayer(m, row_sum_one)
    
    def forward(
        self, 
        x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to produce graphical procedure parameters.
        
        Parameters
        ----------
        x : torch.Tensor, optional
            Input context, shape (batch_size, input_dim). Required for context-aware mode.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            (weights, transitions, gamma) where:
            - weights: shape (batch_size, m) or (m,) for parameter-only mode
            - transitions: shape (batch_size, m, m) or (m, m) for parameter-only mode
            - gamma: shape (m,) or None if not optimizing gamma
        """
        if self.parameter_only:
            weights = self.weight_constraint(self.raw_weights)
            transitions = self.transition_constraint(self.raw_transitions)
            gamma = self.raw_gamma if self.optimize_gamma else None
            
            # If batch input provided, expand to match
            if x is not None:
                batch_size = x.shape[0]
                weights = weights.unsqueeze(0).expand(batch_size, -1)
                transitions = transitions.unsqueeze(0).expand(batch_size, -1, -1)
            
            return weights, transitions, gamma
        else:
            if x is None:
                raise ValueError("Input required for context-aware mode")
            
            # MLP forward pass
            output = self.mlp(x)
            
            # Split output into weights and transitions
            raw_weights = output[..., :self.weight_dim]
            raw_transitions = output[..., self.weight_dim:]
            
            # Apply constraints
            weights = self.weight_constraint(raw_weights)
            transitions = self.transition_constraint(raw_transitions)
            
            return weights, transitions, None  # Context-aware mode doesn't support gamma yet
    
    def get_procedure_params(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get procedure parameters as numpy arrays.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
            (weights, transitions, gamma) as numpy arrays.
            gamma is None if not optimizing gamma.
        """
        with torch.no_grad():
            weights, transitions, gamma = self.forward()
            gamma_np = gamma.cpu().numpy() if gamma is not None else None
            return (
                weights.cpu().numpy(),
                transitions.cpu().numpy(),
                gamma_np
            )
    
    def initialize_from_procedure(
        self, 
        weights: np.ndarray, 
        transitions: np.ndarray
    ):
        """
        Initialize network to output a specific procedure.
        
        This is useful for warm-starting from a known good procedure.
        
        Parameters
        ----------
        weights : np.ndarray
            Initial weights, shape (m,).
        transitions : np.ndarray
            Initial transition matrix, shape (m, m).
        """
        if not self.parameter_only:
            raise ValueError("Can only initialize parameter-only networks")
        
        with torch.no_grad():
            # Inverse softmax for weights
            # softmax(x) = w => x = log(w) + c (c is arbitrary constant)
            log_weights = torch.log(torch.tensor(weights, dtype=torch.float32) + 1e-10)
            log_weights = log_weights - log_weights.mean()  # Center for stability
            self.raw_weights.copy_(log_weights)
            
            # Inverse sigmoid for transitions
            # sigmoid(x) = g => x = log(g / (1 - g))
            g = torch.tensor(transitions, dtype=torch.float32)
            g = torch.clamp(g, 1e-6, 1 - 1e-6)  # Avoid log(0)
            raw_trans = torch.log(g / (1 - g))
            # Zero out diagonal (will be masked anyway)
            raw_trans.fill_diagonal_(0)
            self.raw_transitions.copy_(raw_trans)


class GraphicalProcedureNetworkV2(nn.Module):
    """
    Alternative architecture using separate networks for weights and transitions.
    
    This can provide more flexibility and potentially better optimization.
    """
    
    def __init__(
        self,
        m: int,
        hidden_dims: List[int] = [64, 32],
        input_dim: int = 0,
        dropout: float = 0.1,
        optimize_gamma: bool = False,
        initial_gamma: float = -4.0
    ):
        super().__init__()
        self.m = m
        self.input_dim = input_dim
        self.optimize_gamma = optimize_gamma
        
        # Per-hypothesis gamma parameters for HSD spending (if optimizing)
        if optimize_gamma:
            self.raw_gamma = nn.Parameter(torch.full((m,), initial_gamma, dtype=torch.float32))
        else:
            self.raw_gamma = None
        
        # If no input, use learnable embeddings
        if input_dim == 0:
            self.input_embedding = nn.Parameter(torch.randn(1, 32))
            input_dim = 32
        else:
            self.input_embedding = None
        
        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims[:-1]:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        
        # Separate heads for weights and transitions
        final_hidden = hidden_dims[-1] if hidden_dims else input_dim
        
        self.weight_head = nn.Sequential(
            nn.Linear(prev_dim, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, m)
        )
        
        self.transition_head = nn.Sequential(
            nn.Linear(prev_dim, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, m * m)
        )
        
        # Constraints
        self.weight_constraint = SoftmaxConstraint()
        self.transition_constraint = TransitionMatrixLayer(m)
    
    def forward(
        self, 
        x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if x is None:
            if self.input_embedding is not None:
                x = self.input_embedding
            else:
                raise ValueError("Input required")
        
        # Encode
        features = self.encoder(x)
        
        # Separate heads
        raw_weights = self.weight_head(features)
        raw_transitions = self.transition_head(features).view(-1, self.m, self.m)
        
        # Apply constraints
        weights = self.weight_constraint(raw_weights)
        transitions = self.transition_constraint(raw_transitions)
        
        # Remove batch dim if singleton
        if weights.shape[0] == 1 and self.input_embedding is not None:
            weights = weights.squeeze(0)
            transitions = transitions.squeeze(0)
        
        gamma = self.raw_gamma if self.optimize_gamma else None
        
        return weights, transitions, gamma


class SoftRejectionApproximation(nn.Module):
    """
    Differentiable approximation of hypothesis rejections in graphical testing procedures.
    
    This class produces soft rejections (values in [0,1]) instead of hard boolean rejections,
    enabling gradient-based optimization. Objective functions then use these soft rejections
    via their evaluate_soft() methods to compute differentiable objective values.
    
    Handles both single-stage (n_analyses=1) and sequential designs automatically.
    For sequential designs, models rejections at each interim analysis with proper
    alpha spending and early stopping logic.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        n_analyses: int = 1,
        information_fractions: Optional[np.ndarray] = None,
        spending_functions: Optional[List] = None
    ):
        """
        Parameters
        ----------
        temperature : float
            Temperature for sigmoid approximation. Lower = closer to hard threshold.
        n_analyses : int
            Number of analyses (1 for single-stage, >1 for sequential).
        information_fractions : np.ndarray, optional
            Information fractions for each hypothesis at each analysis, shape (m, n_analyses).
            If None, assumes equal spacing.
        spending_functions : list of SpendingFunction, optional
            Per-hypothesis spending functions. If None, uses single-stage alpha.
        """
        super().__init__()
        self.temperature = temperature
        self.n_analyses = n_analyses
        # Convert to numpy array if needed
        if information_fractions is not None and not isinstance(information_fractions, np.ndarray):
            information_fractions = np.array(information_fractions)
        self.information_fractions = information_fractions
        self.spending_functions = spending_functions
    
    def forward(
        self,
        p_values: torch.Tensor,
        weights: torch.Tensor,
        transitions: torch.Tensor,
        alpha: float = 0.025,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute soft rejection indicators for given p-values and procedure.
        
        For sequential designs (n_analyses > 1), models rejections at each interim
        analysis with early stopping. For single-stage (n_analyses=1), reduces to
        simple threshold comparison with alpha propagation.
        
        Parameters
        ----------
        p_values : torch.Tensor
            P-values at final analysis, shape (batch_size, m).
        weights : torch.Tensor
            Initial alpha weights, shape (m,) or (batch_size, m).
        transitions : torch.Tensor
            Transition matrix, shape (m, m) or (batch_size, m, m).
        alpha : float
            Overall significance level.
        gamma : torch.Tensor, optional
            Per-hypothesis gamma parameters for HSD spending, shape (m,).
            If None, uses spending_functions or full alpha.
        
        Returns
        -------
        torch.Tensor
            Soft rejection indicators, shape (batch_size, m).
        """
        batch_size = p_values.shape[0]
        m = p_values.shape[1]
        device = p_values.device
        
        # Expand weights and transitions if needed
        if weights.dim() == 1:
            weights = weights.unsqueeze(0).expand(batch_size, -1)
        if transitions.dim() == 2:
            transitions = transitions.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Convert p-values to z-scores for sequential analysis
        z_final = -torch.distributions.Normal(0, 1).icdf(torch.clamp(p_values, 1e-10, 1 - 1e-10))
        
        # Initialize rejection tracking (for early stopping in sequential designs)
        cumulative_soft_reject = torch.zeros((batch_size, m), device=device)
        
        # Process each analysis sequentially
        for analysis_idx in range(self.n_analyses):
            # Compute information fraction for this analysis
            if self.information_fractions is not None:
                info_frac = torch.tensor(
                    self.information_fractions[:, analysis_idx],
                    dtype=torch.float32,
                    device=device
                )
            else:
                # Default: equal spacing
                info_frac = torch.full((m,), (analysis_idx + 1) / self.n_analyses, device=device)
            
            # Test statistics at this information fraction
            # Z_k = Z_final * sqrt(I_k) where I_k is information fraction
            z_at_analysis = z_final * torch.sqrt(info_frac.unsqueeze(0))
            p_at_analysis = torch.distributions.Normal(0, 1).cdf(-z_at_analysis)
            
            # Compute nominal alpha boundaries for this analysis
            if gamma is not None and self.n_analyses > 1:
                # Differentiable HSD spending function computation
                # Formula: alpha * (1 - exp(-gamma * t)) / (1 - exp(-gamma))
                t = info_frac  # (m,)
                g = gamma  # (m,)
                
                # For gamma ≈ 0, use linear: alpha * t
                # For gamma ≠ 0, use HSD formula
                # Use smooth transition to handle gamma near 0
                eps = 1e-3
                gamma_abs = torch.abs(g)
                
                # Linear component (for small gamma)
                linear_spending = t
                
                # HSD component (for non-zero gamma)
                exp_neg_gamma = torch.exp(-g)
                exp_neg_gamma_t = torch.exp(-g * t)
                hsd_spending = (1 - exp_neg_gamma_t) / (1 - exp_neg_gamma + 1e-10)
                
                # Smooth blend based on |gamma|
                blend_weight = torch.sigmoid((gamma_abs - eps) / eps)
                cumulative_spent = linear_spending * (1 - blend_weight) + hsd_spending * blend_weight
                
                nominal_alphas = cumulative_spent * alpha
            elif self.spending_functions is not None and self.n_analyses > 1:
                # Use spending functions to get cumulative alphas spent
                # Then compute incremental alpha for this analysis
                cumulative_alphas = torch.tensor(
                    [sf(float(info_frac[h]), 1.0)  # sf(t, alpha) returns alpha_spent
                     for h, sf in enumerate(self.spending_functions)],
                    dtype=torch.float32,
                    device=device
                )
                # For now, use cumulative alpha as nominal (simplification)
                # In exact sequential testing, we'd use nominal alpha per stage
                # but for differentiable approximation, cumulative works
                nominal_alphas = cumulative_alphas * alpha
            else:
                # Single-stage: use full alpha
                nominal_alphas = torch.full((m,), alpha, device=device)
            
            # Apply graphical procedure at this analysis
            # Initial alpha allocation
            current_alpha = nominal_alphas.unsqueeze(0) * weights  # (batch_size, m)
            
            # Soft rejection at this analysis
            soft_reject = torch.sigmoid((current_alpha - p_at_analysis) / self.temperature)
            
            # Iteratively update alpha allocation (within this analysis)
            for _ in range(m - 1):
                # Expected alpha to propagate
                propagated = torch.einsum('bi,bij->bj', soft_reject * current_alpha, transitions)
                current_alpha = nominal_alphas.unsqueeze(0) * weights + propagated
                # Clamp to [0, nominal_alphas]
                current_alpha = torch.clamp(current_alpha, min=0.0)
                current_alpha = torch.minimum(current_alpha, nominal_alphas.unsqueeze(0))
                
                # Update soft rejections
                soft_reject = torch.sigmoid((current_alpha - p_at_analysis) / self.temperature)
            
            # Early stopping: once rejected, stays rejected
            # Accumulate rejections across analyses
            cumulative_soft_reject = torch.maximum(cumulative_soft_reject, soft_reject)
        
        return cumulative_soft_reject
    
    def compute_power(
        self,
        p_values: torch.Tensor,
        weights: torch.Tensor,
        transitions: torch.Tensor,
        alpha: float = 0.025,
        power_type: str = "disjunctive"
    ) -> torch.Tensor:
        """
        Compute approximate power metric.
        
        Parameters
        ----------
        p_values : torch.Tensor
            P-values, shape (batch_size, m).
        weights : torch.Tensor
            Initial alpha weights.
        transitions : torch.Tensor
            Transition matrix.
        alpha : float
            Overall significance level.
        power_type : str
            Type of power: "disjunctive", "conjunctive", "expected", "marginal".
        
        Returns
        -------
        torch.Tensor
            Scalar power estimate.
        """
        soft_reject = self.forward(p_values, weights, transitions, alpha)
        
        if power_type == "disjunctive":
            # P(at least one rejection) ≈ 1 - prod(1 - soft_reject)
            power = 1 - torch.prod(1 - soft_reject, dim=1)
        elif power_type == "conjunctive":
            # P(all rejections) ≈ prod(soft_reject)
            power = torch.prod(soft_reject, dim=1)
        elif power_type == "expected":
            # E[number of rejections] = sum(soft_reject)
            power = torch.sum(soft_reject, dim=1)
        elif power_type == "marginal":
            # Return all marginal powers
            power = soft_reject
        else:
            raise ValueError(f"Unknown power_type: {power_type}")
        
        return power.mean()
