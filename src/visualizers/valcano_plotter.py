"""Function to create volcano plot."""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_volcano_plot(
    fold_changes: pd.Series | np.ndarray,
    p_values: pd.Series | np.ndarray,
    fc_threshold: float = 1.5,
    p_threshold: float = 0.05,
    title: str = "Volcano Plot",
    show_thresholds: bool = True,
    point_size: int = 30,
    fig_size: tuple[int, int] = (10, 8),
    label_points: dict[Any, str] | list[Any] | None = None,
    label_offset: tuple[int, int] = (15, 15),
    arrow_props: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Creates a volcano plot to visualize fold changes and p-values.

    Parameters
    ----------
    fold_changes : pd.Series or np.ndarray
        Log2 fold change values for each data point
    p_values : pd.Series or np.ndarray
        P-values corresponding to each fold change value
    fc_threshold : float, default=1.5
        Fold change threshold for significance
    p_threshold : float, default=0.05
        P-value threshold for significance
    title : str, default="Volcano Plot"
        Title for the plot
    show_thresholds : bool, default=True
        Whether to display threshold lines
    point_size : int, default=30
        Size of scatter points
    fig_size : tuple, default=(10, 8)
        Figure size (width, height) in inches
    label_points : dict or list, default=None
        dictionary with indices as keys and labels as values for specific points,
        or list of indices to label with their index value
    label_offset : tuple, default=(15, 15)
        Offset (x, y) in points for label placement
    arrow_props : dict, default=None
        Properties for the arrow connecting points to labels

    Returns
    -------
    fig, ax : matplotlib figure and axis objects

    """
    # Default arrow properties if none provided
    if arrow_props is None:
        arrow_props = {
            "arrowstyle": "->",
            "color": "black",
            "alpha": 0.7,
            "linewidth": 0.8,
            "connectionstyle": "arc3,rad=0.2",
        }

    # Convert p-values to -log10 scale for better visualization
    log_p_values: pd.Series | np.ndarray = -np.log10(p_values)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot points with color coding
    colors: list[str] = _get_point_colors(
        fold_changes, p_values, fc_threshold, p_threshold
    )
    ax.scatter(fold_changes, log_p_values, c=colors, alpha=0.6, s=point_size)

    # Add threshold lines if requested
    if show_thresholds:
        _add_threshold_lines(ax, p_threshold, fc_threshold)

    # Add labels to specific points if provided
    if label_points is not None:
        _add_point_labels(
            ax, fold_changes, log_p_values, label_points, label_offset, arrow_props
        )

    # Set plot styling and legend
    _set_plot_styling(ax, title)

    plt.tight_layout()
    return fig, ax


def _get_point_colors(
    fold_changes: pd.Series | np.ndarray,
    p_values: pd.Series | np.ndarray,
    fc_threshold: float,
    p_threshold: float,
) -> list[str]:
    """
    Determine color for each data point based on significance.

    Parameters
    ----------
    fold_changes : pd.Series or np.ndarray
        Log2 fold change values
    p_values : pd.Series or np.ndarray
        P-values corresponding to fold changes
    fc_threshold : float
        Fold change threshold for significance
    p_threshold : float
        P-value threshold for significance

    Returns
    -------
    colors : list[str]
        list of color strings for each data point

    """
    colors: list[str] = []
    for fc, pval in zip(fold_changes, p_values, strict=False):
        if abs(fc) > fc_threshold and pval < p_threshold:
            colors.append("red" if fc > 0 else "blue")
        else:
            colors.append("gray")
    return colors


def _add_threshold_lines(ax: plt.Axes, p_threshold: float, fc_threshold: float) -> None:
    """
    Add threshold lines to the plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to draw on
    p_threshold : float
        P-value threshold for significance
    fc_threshold : float
        Fold change threshold for significance

    """
    # Horizontal line for p-value threshold
    ax.axhline(y=-np.log10(p_threshold), color="gray", linestyle="--", alpha=0.7)

    # Vertical lines for fold change thresholds
    ax.axvline(x=fc_threshold, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=-fc_threshold, color="gray", linestyle="--", alpha=0.7)


def _add_point_labels(
    ax: plt.Axes,
    fold_changes: pd.Series | np.ndarray,
    log_p_values: pd.Series | np.ndarray,
    label_points: dict[Any, str] | list[Any],
    label_offset: tuple[int, int],
    arrow_props: dict[str, Any],
) -> None:
    """
    Add labels to specified points with connecting arrows.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to draw on
    fold_changes : pd.Series or np.ndarray
        Log2 fold change values
    log_p_values : pd.Series or np.ndarray
        -Log10 transformed p-values
    label_points : dict or list
        dictionary with indices as keys and labels as values,
        or list of indices to label with their index value
    label_offset : tuple
        Offset (x, y) in points for label placement
    arrow_props : dict
        Properties for the arrow connecting points to labels

    """
    if isinstance(label_points, dict):
        for idx, label in label_points.items():
            _add_single_label(
                ax, fold_changes, log_p_values, idx, label, label_offset, arrow_props
            )
    elif isinstance(label_points, list):
        for idx in label_points:
            _add_single_label(
                ax, fold_changes, log_p_values, idx, str(idx), label_offset, arrow_props
            )


def _add_single_label(
    ax: plt.Axes,
    fold_changes: pd.Series | np.ndarray,
    log_p_values: pd.Series | np.ndarray,
    idx: Any,
    label: str,
    label_offset: tuple[int, int],
    arrow_props: dict[str, Any],
) -> None:
    """
    Add a label to a single point.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to draw on
    fold_changes : pd.Series or np.ndarray
        Log2 fold change values
    log_p_values : pd.Series or np.ndarray
        -Log10 transformed p-values
    idx : Any
        Index of the point to label
    label : str
        Text to display as the label
    label_offset : tuple
        Offset (x, y) in points for label placement
    arrow_props : dict
        Properties for the arrow connecting point to label

    """
    # Check if the index exists in the dataset
    if hasattr(fold_changes, "index") and idx in fold_changes.index:
        x, y = fold_changes[idx], log_p_values[idx]
        offset_x, offset_y = label_offset[0] / 72, label_offset[1] / 72

        ax.annotate(
            label,
            xy=(x, y),  # Point position
            xytext=(x + offset_x, y + offset_y),  # Text position
            textcoords="data",
            fontsize=8,
            alpha=0.9,
            arrowprops=arrow_props,
        )


def _set_plot_styling(ax: plt.Axes, title: str) -> None:
    """
    Set axis labels, title, legend and grid.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to style
    title : str
        Plot title

    """
    # Set plot labels and title
    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-Log10 P-value")
    ax.set_title(title)

    # Add a legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Upregulated",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Downregulated",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Not significant",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Add grid for better readability
    ax.grid(visible=True, linestyle="--", alpha=0.3)


# Example usage
if __name__ == "__main__":
    # Create a dictionary to map gene IDs to their data
    gene_data: dict[str, dict[str, float]] = {
        "BRCA1": {"fold_change": 2.5, "p_value": 0.001},
        "TP53": {"fold_change": -3.1, "p_value": 0.0005},
        "KRAS": {"fold_change": 1.8, "p_value": 0.04},
        "EGFR": {"fold_change": 4.2, "p_value": 0.0001},
        "PTEN": {"fold_change": -2.7, "p_value": 0.003},
        "MYC": {"fold_change": 1.2, "p_value": 0.08},
        "CDK4": {"fold_change": 0.7, "p_value": 0.12},
        "RB1": {"fold_change": -1.9, "p_value": 0.01},
        "AKT1": {"fold_change": 0.4, "p_value": 0.3},
        "CDKN2A": {"fold_change": -3.4, "p_value": 0.0008},
    }

    # Create Pandas Series with gene IDs as indices
    fold_changes: pd.Series = pd.Series(
        {gene: data["fold_change"] for gene, data in gene_data.items()}
    )
    p_values: pd.Series = pd.Series(
        {gene: data["p_value"] for gene, data in gene_data.items()}
    )

    # Create a dictionary for labeling specific genes of interest
    genes_to_label: dict[str, str] = {
        "BRCA1": "BRCA1",
        "TP53": "TP53",
        "EGFR": "EGFR",
        "PTEN": "PTEN",
    }

    # Create the volcano plot
    fig, ax = create_volcano_plot(
        fold_changes,
        p_values,
        fc_threshold=1.5,
        p_threshold=0.05,
        title="Differential Gene Expression Analysis",
        label_points=genes_to_label,
    )

    plt.show()
