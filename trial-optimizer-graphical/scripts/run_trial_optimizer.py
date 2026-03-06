from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _as_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _add_vendor_to_syspath() -> None:
    """Make the skill standalone by preferring the vendored trial_optimizer package.

    If the vendor dir is missing, we fall back to an installed `trial_optimizer`.
    """

    skill_root = Path(__file__).resolve().parents[1]
    vendor_dir = skill_root / "vendor"
    if vendor_dir.exists() and vendor_dir.is_dir():
        sys.path.insert(0, str(vendor_dir))


def _parse_correlation(corr: Union[float, List[List[float]], List[float]]) -> Union[float, np.ndarray]:
    if isinstance(corr, (int, float)):
        return float(corr)
    arr = np.asarray(corr, dtype=float)
    if arr.ndim == 1:
        raise ValueError(
            "correlation must be a scalar (equicorrelation) or an m×m matrix; got 1D array"
        )
    return arr


def _build_objective(*, cfg: Dict[str, Any], m: int):
    from trial_optimizer import GatedSuccess, MarginalRejection, WeightedSuccess

    obj_cfg = cfg.get("objective") or {}
    weights = np.asarray(obj_cfg.get("weights", [1.0] * m), dtype=float)

    success_cfg = obj_cfg.get("success") or {"type": "marginal"}
    success_type = (success_cfg.get("type") or "marginal").lower()

    if success_type == "marginal":
        success_fn = MarginalRejection()
    elif success_type == "gated":
        deps_raw = success_cfg.get("dependencies") or {}
        # JSON keys come in as strings; normalize to int.
        deps: Dict[int, List[int]] = {int(k): [int(i) for i in v] for k, v in deps_raw.items()}
        success_fn = GatedSuccess(dependencies=deps)
    else:
        raise ValueError(f"Unknown objective.success.type: {success_type}")

    return WeightedSuccess(success_fn=success_fn, weights=weights, normalize_weights=True)


def _build_spending_functions(*, cfg: Dict[str, Any], m: int):
    from trial_optimizer import HwangShihDeCani, Linear, OBrienFleming, Pocock

    sf_cfg = cfg.get("spending_function") or {"type": "hsd", "gamma": -4.0}
    sf_type = (sf_cfg.get("type") or "hsd").lower()

    if sf_type in {"hsd", "hwangshihdecani", "hwang-shih-de-cani"}:
        gamma = float(sf_cfg.get("gamma", -4.0))
        sf = HwangShihDeCani(gamma=gamma)
    elif sf_type in {"obf", "obrienfleming", "o'brien-fleming", "obrien-fleming"}:
        sf = OBrienFleming()
    elif sf_type == "pocock":
        sf = Pocock()
    elif sf_type == "linear":
        sf = Linear()
    else:
        raise ValueError(f"Unknown spending_function.type: {sf_type}")

    return [sf] * m


def _power_result_to_dict(power) -> Dict[str, Any]:
    return {
        "marginal_power": [float(x) for x in np.asarray(power.marginal_power).tolist()],
        "disjunctive_power": float(power.disjunctive_power),
        "conjunctive_power": float(power.conjunctive_power),
        "expected_rejections": float(power.expected_rejections),
        "n_simulations": int(power.n_simulations),
        "early_stop_rate": None if power.early_stop_rate is None else float(power.early_stop_rate),
        "custom_power": power.custom_power,
    }


def _plot_training(outdir: Path, result) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(result.loss_history, "b-", alpha=0.7)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("iteration")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.objective_history, "g-", alpha=0.7)
    axes[1].set_title("Objective (reported)")
    axes[1].set_xlabel("iteration")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "training.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_procedure(outdir: Path, proc, labels: Optional[List[str]]) -> Optional[str]:
    try:
        from trial_optimizer import plot_graphical_procedure
    except Exception:
        return None

    try:
        fig = plot_graphical_procedure(
            proc,
            hypothesis_labels=labels,
            title="Optimized",
            show=False,
            save_path=str(outdir / "design_optimized.png"),
        )
        return None if fig is None else str(outdir / "design_optimized.png")
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run trial_optimizer optimization from a JSON config")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for Monte Carlo benchmarking")
    ap.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    args = ap.parse_args()

    cfg_path = _as_path(args.config)
    outdir = _as_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _add_vendor_to_syspath()
    cfg = _load_json(cfg_path)

    m = int(cfg["m"])
    alpha = float(cfg.get("alpha", 0.025))
    effect_sizes = np.asarray(cfg["effect_sizes"], dtype=float)
    correlation = _parse_correlation(cfg.get("correlation", 0.3))
    one_sided = bool(cfg.get("one_sided", True))
    hypothesis_labels = cfg.get("hypothesis_labels")

    objective = _build_objective(cfg=cfg, m=m)

    optimizer_cfg = cfg.get("optimizer") or {}
    n_iterations = int(optimizer_cfg.get("n_iterations", 600))
    batch_size = int(optimizer_cfg.get("batch_size", 2000))
    learning_rate = float(optimizer_cfg.get("learning_rate", 0.05))
    patience = int(optimizer_cfg.get("patience", 200))
    eval_every = int(optimizer_cfg.get("eval_every", 50))
    n_eval_samples = int(optimizer_cfg.get("n_eval_samples", 20000))
    verbose = bool(optimizer_cfg.get("verbose", True))

    init_cfg = cfg.get("procedure_init") or {}
    initial_weights = init_cfg.get("weights")
    initial_transitions = init_cfg.get("transitions")

    from trial_optimizer import HolmProcedure, PowerSimulator

    simulator = PowerSimulator(
        m=m,
        correlation=correlation,
        effect_sizes=effect_sizes,
        one_sided=one_sided,
        seed=args.seed,
    )

    seq_cfg = cfg.get("sequential") or {"enabled": False}
    sequential_enabled = bool(seq_cfg.get("enabled", False))

    if sequential_enabled:
        from trial_optimizer import optimize_sequential_procedure

        n_analyses = int(seq_cfg.get("n_analyses", 2))
        information_fractions = np.asarray(seq_cfg.get("information_fractions"), dtype=float)
        spending_functions = _build_spending_functions(cfg=seq_cfg, m=m)
        optimize_spending = bool(seq_cfg.get("optimize_spending", True))
        initial_gamma = float(seq_cfg.get("initial_gamma", -4.0))

        result = optimize_sequential_procedure(
            m=m,
            effect_sizes=effect_sizes,
            correlation=correlation,
            alpha=alpha,
            objective=objective,
            n_analyses=n_analyses,
            information_fractions=information_fractions,
            spending_function=spending_functions,
            optimize_spending=optimize_spending,
            initial_gamma=initial_gamma,
            n_iterations=n_iterations,
            verbose=verbose,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            eval_every=eval_every,
            n_eval_samples=n_eval_samples,
            initial_weights=initial_weights,
            initial_transitions=initial_transitions,
        )
    else:
        from trial_optimizer import optimize_graphical_procedure

        result = optimize_graphical_procedure(
            m=m,
            effect_sizes=effect_sizes,
            correlation=correlation,
            alpha=alpha,
            objective=objective,
            n_iterations=n_iterations,
            verbose=verbose,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            eval_every=eval_every,
            n_eval_samples=n_eval_samples,
            initial_weights=initial_weights,
            initial_transitions=initial_transitions,
        )

    optimized_proc = result.optimal_procedure

    optimal_gamma = None
    if sequential_enabled:
        try:
            sf0 = optimized_proc.spending_function[0]
            if hasattr(sf0, "gamma"):
                optimal_gamma = float(sf0.gamma)
        except Exception:
            optimal_gamma = None

    benchmark_cfg = cfg.get("benchmark") or {}
    n_bench = int(benchmark_cfg.get("n_simulations", 20000))

    holm = HolmProcedure(m=m, alpha=alpha)
    holm_power = simulator.compute_power(holm, n_simulations=n_bench)
    optimized_power = simulator.compute_power(optimized_proc, n_simulations=n_bench)

    artifacts: Dict[str, Optional[str]] = {
        "result_json": str(outdir / "result.json"),
        "training_plot": None,
        "procedure_plot": None,
    }

    if not args.no_plots:
        artifacts["training_plot"] = _plot_training(outdir, result)
        artifacts["procedure_plot"] = _plot_procedure(outdir, optimized_proc, hypothesis_labels)

    summary = {
        "config_path": str(cfg_path),
        "m": m,
        "alpha": alpha,
        "effect_sizes": effect_sizes.tolist(),
        "correlation": correlation if isinstance(correlation, float) else np.asarray(correlation).tolist(),
        "sequential": {
            "enabled": sequential_enabled,
            **(
                {
                    "n_analyses": int(seq_cfg.get("n_analyses", 2)),
                    "information_fractions": (
                        np.asarray(seq_cfg.get("information_fractions"), dtype=float).tolist()
                    ),
                    "spending_function": seq_cfg.get("spending_function"),
                    "optimize_spending": bool(seq_cfg.get("optimize_spending", True)),
                    "optimal_gamma": optimal_gamma,
                }
                if sequential_enabled
                else {}
            ),
        },
        "optimal_weights": result.optimal_weights.tolist(),
        "optimal_transitions": result.optimal_transitions.tolist(),
        "power": {
            "holm": _power_result_to_dict(holm_power),
            "optimized": _power_result_to_dict(optimized_power),
        },
        "optimization": {
            "n_iterations": int(result.n_iterations),
            "converged": bool(result.converged),
        },
        "artifacts": artifacts,
    }

    _dump_json(outdir / "result.json", summary)

    print("--- trial_optimizer run complete ---")
    print(f"Output: {outdir}")
    print(f"Optimal weights: {np.asarray(result.optimal_weights).round(4)}")
    print("Holm disjunctive power:", f"{holm_power.disjunctive_power:.4f}")
    print("Optimized disjunctive power:", f"{optimized_power.disjunctive_power:.4f}")
    print("Optimized marginal power:", np.asarray(optimized_power.marginal_power).round(4))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
