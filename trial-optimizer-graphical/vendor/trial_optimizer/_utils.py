"""Internal helpers used by GraphicalProcedure.

This module is intentionally internal (underscored) to avoid growing the public API.

The functions here focus on preparing/coercing/validating user inputs (weights,
transitions, p-values, information fractions, spending functions) into a single
representation used by the core algorithms.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np


SpendingCallable = Callable[[float, float], float]


def prepare_weights_transitions(
    weights: np.ndarray,
    transitions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(transitions, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {w.shape}")
    m = int(w.shape[0])
    if g.shape != (m, m):
        raise ValueError(f"transitions must be ({m}, {m}), got {g.shape}")
    return w, g, m


def prepare_information_fractions(
    information_fractions: Optional[Sequence[np.ndarray]],
    *,
    m: int,
    n_analyses: int,
) -> List[np.ndarray]:
    if information_fractions is None:
        if n_analyses == 1:
            default_fractions = np.array([1.0], dtype=np.float64)
        else:
            default_fractions = np.linspace(1.0 / n_analyses, 1.0, n_analyses, dtype=np.float64)
        return [default_fractions.copy() for _ in range(m)]

    if not isinstance(information_fractions, (list, tuple)):
        raise ValueError(
            f"information_fractions must be a list/tuple of length {m} (one per hypothesis), "
            f"got {type(information_fractions).__name__}"
        )
    if len(information_fractions) != m:
        raise ValueError(
            f"information_fractions must have length {m} (one per hypothesis), got {len(information_fractions)}"
        )

    info_fracs: List[np.ndarray] = []
    for h, info_frac in enumerate(information_fractions):
        arr = np.asarray(info_frac, dtype=np.float64)
        if arr.shape != (n_analyses,):
            raise ValueError(
                f"information_fractions for hypothesis {h} must have length {n_analyses}, got {arr.shape}"
            )
        if not np.all(np.diff(arr) >= 0):
            raise ValueError(
                f"information_fractions for hypothesis {h} must be non-decreasing (cannot decrease across analyses)"
            )
        if not np.any(np.isclose(arr, 1.0)):
            raise ValueError(f"information_fractions for hypothesis {h} must reach 1.0 at some analysis")

        reached_one = np.where(np.isclose(arr, 1.0))[0]
        if len(reached_one) > 0:
            first_one_idx = int(reached_one[0])
            if not np.allclose(arr[first_one_idx:], 1.0):
                raise ValueError(
                    f"information_fractions for hypothesis {h} must stay at 1.0 after reaching 1.0 "
                    f"(cannot have additional data after completion)"
                )

        info_fracs.append(arr.copy())

    return info_fracs


def prepare_spending_functions(
    spending_function: Optional[Sequence[SpendingCallable]],
    *,
    m: int,
) -> List[SpendingCallable]:
    if spending_function is None:
        from .spending_functions import OBrienFleming

        return [OBrienFleming() for _ in range(m)]

    if not isinstance(spending_function, (list, tuple)):
        raise ValueError(
            f"spending_function must be a list/tuple of length {m} (one per hypothesis), "
            f"got {type(spending_function).__name__}"
        )
    if len(spending_function) != m:
        raise ValueError(
            f"spending_function must have length {m} (one per hypothesis), got {len(spending_function)}"
        )
    if any(isinstance(sf, str) for sf in spending_function):
        raise ValueError(
            "spending_function no longer accepts string shortcuts. "
            "Pass callables like OBrienFleming(), Pocock(), Linear(), or HwangShihDeCani()."
        )

    sfs = list(spending_function)
    if not all(callable(sf) for sf in sfs):
        raise ValueError("Each element of spending_function must be callable f(t, alpha) -> float")

    return sfs


def prepare_p_values(
    p_values: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    m: int,
    n_analyses_planned: int,
) -> List[np.ndarray]:
    """Prepare p-values into list-of-arrays (one array per analysis).

    Accepts:
    - 1D array shape (m,) -> treated as a single analysis
    - 2D array shape (n_analyses, m)
    - list/tuple of 1D arrays, each shape (m,)
    """

    if isinstance(p_values, (list, tuple)):
        p_values_by_analysis = [np.asarray(pv, dtype=np.float64) for pv in p_values]
    else:
        p_arr = np.asarray(p_values, dtype=np.float64)
        if p_arr.ndim == 1:
            p_values_by_analysis = [p_arr]
        elif p_arr.ndim == 2:
            p_values_by_analysis = [p_arr[t] for t in range(p_arr.shape[0])]
        else:
            raise ValueError(f"p_values must be 1D, 2D, or a list; got shape {p_arr.shape}")

    n_provided_analyses = len(p_values_by_analysis)
    if n_provided_analyses > n_analyses_planned:
        raise ValueError(
            f"Provided {n_provided_analyses} analyses, but only {n_analyses_planned} were planned"
        )

    for t, pv in enumerate(p_values_by_analysis):
        if pv.shape != (m,):
            raise ValueError(f"Expected {m} p-values at analysis {t}, got shape {pv.shape}")

    return p_values_by_analysis
