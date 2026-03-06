"""
Visualization utilities for graphical testing procedures.

Provides functions to visualize:
1. Graphical procedures as directed graphs
2. Training progress
3. Power comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Sequence, Union
import warnings

from .graphical_procedure import GraphicalProcedure
from .optimizer import OptimizationResult


_ProcedureInput = Union[
    GraphicalProcedure,
    Sequence[GraphicalProcedure],
    Sequence[Tuple[str, GraphicalProcedure]],
]


def plot_graphical_procedure(
    procedure: _ProcedureInput,
    hypothesis_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    node_size: int = 6500,
    font_size: int = 8,
    edge_threshold: float = 0.01,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Visualize a graphical procedure as a directed graph (DAG).

    Parameters
    ----------
    procedure : GraphicalProcedure
        The procedure to visualize.
    hypothesis_labels : List[str], optional
        Labels for each hypothesis. Default is H1, H2, ...
    figsize : Tuple[int, int]
        Figure size.
    node_size : int
        Size of nodes.
    font_size : int
        Font size for labels.
    edge_threshold : float
        Minimum transition weight to display.
    title : str, optional
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    try:
        import networkx as nx
    except ImportError:
        warnings.warn(
            "networkx required for graph visualization. Install with: pip install networkx"
        )
        return None

    def _normalize_input(
        p: _ProcedureInput,
    ) -> List[Tuple[Optional[str], GraphicalProcedure]]:
        if isinstance(p, GraphicalProcedure):
            return [(title, p)]

        if not isinstance(p, (list, tuple)) or len(p) == 0:
            raise TypeError(
                "plot_graphical_procedure expects a GraphicalProcedure, a list of GraphicalProcedure, "
                "or a list of (name, GraphicalProcedure)."
            )

        first = p[0]
        if isinstance(first, GraphicalProcedure):
            return [(None, item) for item in p]  # type: ignore[misc]

        if (
            isinstance(first, tuple)
            and len(first) == 2
            and isinstance(first[0], str)
            and isinstance(first[1], GraphicalProcedure)
        ):
            out: List[Tuple[Optional[str], GraphicalProcedure]] = []
            for item in p:  # type: ignore[assignment]
                if not (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], GraphicalProcedure)
                ):
                    raise TypeError("All items must be (name, GraphicalProcedure)")
                out.append((item[0], item[1]))
            return out

        raise TypeError(
            "plot_graphical_procedure expects a GraphicalProcedure, a list of GraphicalProcedure, "
            "or a list of (name, GraphicalProcedure)."
        )

    def _plot_one(
        *,
        proc: GraphicalProcedure,
        ax_: plt.Axes,
        title_: Optional[str],
        hypothesis_labels_: List[str],
    ) -> None:
        m_ = int(proc.m)
        weights_ = np.asarray(proc.weights, dtype=float)
        transitions_ = np.asarray(proc.transitions, dtype=float)

        G = nx.DiGraph()
        for i in range(m_):
            G.add_node(i)

        for i in range(m_):
            for j in range(m_):
                if i != j and transitions_[i, j] > edge_threshold:
                    G.add_edge(i, j, weight=float(transitions_[i, j]))

        pos = nx.circular_layout(G)
        max_weight = float(max(weights_.max(), 0.01))
        node_colors = plt.cm.Blues(weights_ / max_weight * 0.8 + 0.2)

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax_,
            node_color=node_colors,
            node_size=node_size,
            edgecolors="black",
            linewidths=2,
        )

        for i in range(m_):
            alpha_i = float(weights_[i] * proc.alpha)
            label_text = (
                f"{hypothesis_labels_[i]}\nw={weights_[i]:.3f}\nα={alpha_i:.4f}"
                if weights_[i] > 0.001
                else f"{hypothesis_labels_[i]}\nw≈0"
            )
            color_intensity = weights_[i] / max_weight * 0.8 + 0.2
            font_color = "white" if color_intensity > 0.5 else "black"
            x, y = pos[i]
            ax_.text(
                x,
                y,
                label_text,
                fontsize=font_size,
                ha="center",
                va="center",
                color=font_color,
                fontweight="bold",
            )

        edges = list(G.edges(data=True))
        cmap = plt.cm.YlOrRd

        # Prepare edge attributes for drawing
        edge_widths = [2 + 4 * float(d["weight"]) for u, v, d in edges]
        edge_colors = [cmap(0.2 + 0.7 * float(d["weight"])) for u, v, d in edges]

        # Draw all edges at once with curved style
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax_,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.9,
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.3",  # Uniform curvature for all edges
            node_size=node_size,
            min_source_margin=22,
            min_target_margin=22,
        )

        # Draw edge labels - must match the connectionstyle used for edges
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax_,
            font_size=max(font_size - 2, 6),
            font_color="white",
            font_weight="bold",
            label_pos=0.5,  # Position at midpoint of edge
            rotate=False,  # Don't rotate labels (keep horizontal for readability)
            connectionstyle="arc3,rad=0.3",  # Must match edge connectionstyle
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="orange",
                edgecolor="white",
                linewidth=1.5,
                alpha=0.95,
            ),
        )

        ax_.set_title(
            title_ or f"Graphical Procedure (α={proc.alpha})",
            fontsize=14,
            fontweight="bold",
        )
        ax_.axis("off")

    items = _normalize_input(procedure)
    m = int(items[0][1].m)
    for _, proc in items:
        if int(proc.m) != m:
            raise ValueError("All procedures must have matching dimensions")

    if hypothesis_labels is None:
        hypothesis_labels = [f"H{i + 1}" for i in range(m)]

    if ax is not None:
        if len(items) != 1:
            raise ValueError(
                "When providing ax, plot_graphical_procedure supports only a single procedure"
            )
        fig = ax.get_figure()
        _plot_one(
            proc=items[0][1],
            ax_=ax,
            title_=items[0][0] or title,
            hypothesis_labels_=hypothesis_labels,
        )
    else:
        if len(items) == 1:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            _plot_one(
                proc=items[0][1],
                ax_=ax1,
                title_=items[0][0] or title,
                hypothesis_labels_=hypothesis_labels,
            )
        else:
            fig, axes = plt.subplots(1, len(items), figsize=figsize)
            if len(items) == 1:
                axes = [axes]
            for ax_i, (name_i, proc_i) in zip(axes, items):
                _plot_one(
                    proc=proc_i,
                    ax_=ax_i,
                    title_=name_i,
                    hypothesis_labels_=hypothesis_labels,
                )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_training_progress(
    result: OptimizationResult,
    figsize: Tuple[int, int] = (12, 4),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the training progress from an optimization result.

    Parameters
    ----------
    result : OptimizationResult
        The optimization result to visualize.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Overall title.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss history
    axes[0].plot(result.loss_history, "b-", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Power history
    eval_iters = np.linspace(0, len(result.loss_history), len(result.power_history))
    axes[1].plot(eval_iters, result.power_history, "g-o", markersize=3)
    axes[1].axhline(
        y=result.power_history[-1],
        color="r",
        linestyle="--",
        label=f"Final: {result.power_history[-1]:.4f}",
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Power During Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def compare_procedures(
    procedures: List[Tuple[str, GraphicalProcedure]],
    simulator,
    n_simulations: int = 10000,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Compare power of multiple graphical procedures.

    Parameters
    ----------
    procedures : List[Tuple[str, GraphicalProcedure]]
        List of (name, procedure) tuples to compare.
    simulator : PowerSimulator
        Simulator for computing power.
    n_simulations : int
        Number of simulations for power computation.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    names = [name for name, _ in procedures]

    # Compute power for each procedure
    results = []
    for name, proc in procedures:
        power = simulator.compute_power(proc, n_simulations)
        results.append(power)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Marginal powers
    x = np.arange(simulator.m)
    width = 0.8 / len(procedures)

    for i, (name, result) in enumerate(zip(names, results)):
        offset = (i - len(procedures) / 2 + 0.5) * width
        axes[0].bar(x + offset, result.marginal_power, width, label=name, alpha=0.8)

    axes[0].set_xlabel("Hypothesis")
    axes[0].set_ylabel("Marginal Power")
    axes[0].set_title("Marginal Power by Hypothesis")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"H{i + 1}" for i in range(simulator.m)])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Summary metrics
    metrics = ["Disjunctive", "Conjunctive", "Expected"]
    values = {
        name: [r.disjunctive_power, r.conjunctive_power, r.expected_rejections]
        for name, r in zip(names, results)
    }

    x = np.arange(len(metrics))
    for i, name in enumerate(names):
        offset = (i - len(names) / 2 + 0.5) * width
        axes[1].bar(x + offset, values[name], width, label=name, alpha=0.8)

    axes[1].set_xlabel("Metric")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Power Metrics Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_power_surface(
    simulator,
    alpha: float = 0.025,
    weight_resolution: int = 20,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot power surface for 2-hypothesis case as a function of weight allocation.

    Parameters
    ----------
    simulator : PowerSimulator
        Simulator (must have m=2).
    alpha : float
        Significance level.
    weight_resolution : int
        Resolution of the weight grid.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    if simulator.m != 2:
        raise ValueError("Power surface plot only available for m=2")

    # Create weight grid
    w1_grid = np.linspace(0.01, 0.99, weight_resolution)

    # For 2 hypotheses, transitions can be parameterized by g12 and g21
    g_grid = np.linspace(0, 1, weight_resolution)

    # Compute power for each combination
    power_matrix = np.zeros((weight_resolution, weight_resolution))

    for i, w1 in enumerate(w1_grid):
        weights = np.array([w1, 1 - w1])
        for j, g in enumerate(g_grid):
            transitions = np.array([[0, g], [g, 0]])
            try:
                proc = GraphicalProcedure(weights, transitions, alpha)
                result = simulator.compute_power(proc, n_simulations=2000)
                power_matrix[i, j] = result.disjunctive_power
            except:
                power_matrix[i, j] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(
        power_matrix, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="viridis"
    )

    ax.set_xlabel("Transition weight (g₁₂ = g₂₁)")
    ax.set_ylabel("Weight on H₁")
    ax.set_title("Disjunctive Power Surface")

    plt.colorbar(im, ax=ax, label="Power")

    # Mark maximum
    max_idx = np.unravel_index(np.nanargmax(power_matrix), power_matrix.shape)
    ax.plot(
        g_grid[max_idx[1]],
        w1_grid[max_idx[0]],
        "r*",
        markersize=15,
        label=f"Max: {power_matrix[max_idx]:.4f}",
    )
    ax.legend()

    return fig
