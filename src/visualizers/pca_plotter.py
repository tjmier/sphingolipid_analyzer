"""Module for creating PCA plots from lipidomics data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Dict, List, Union


def create_pca_plot(
    data: Union[pd.DataFrame, np.ndarray],
    sample_labels: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
    group_colors: Optional[Dict[str, str]] = None,
    n_components: int = 2,
    fig_size: Tuple[int, int] = (10, 8),
    title: str = "PCA of Lipidomics Data",
    point_size: int = 70,
    plot_loadings: bool = False,
    confidence_ellipse: bool = True,
    random_state: int = 42,
) -> Tuple[plt.Figure, plt.Axes, PCA, np.ndarray]:
    """
    Create a Principal Component Analysis (PCA) plot from lipidomics data.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input data where columns are samples and rows are lipid analytes.
        If a DataFrame, the index should contain analyte names.
    sample_labels : list of str, optional
        Labels for each sample (column). If None and data is a DataFrame,
        column names will be used.
    group_labels : list of str, optional
        Group assignment for each sample, used for coloring points.
        Must match the order of columns in data.
    group_colors : dict, optional
        Mapping of group names to colors.
    n_components : int, default=2
        Number of principal components to calculate.
    fig_size : tuple, default=(10, 8)
        Figure size (width, height) in inches.
    title : str, default="PCA of Lipidomics Data"
        Plot title.
    point_size : int, default=70
        Size of scatter points.
    confidence_ellipse : bool, default=True
        If True, draw confidence ellipses around groups.
    plot_loadings : bool, default=False
        If True, plot the top loading vectors on the PCA.
    confidence_ellipse : bool, default=True
        If True, draw confidence ellipses around groups.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the PCA plot.
    ax : matplotlib.axes.Axes
        The matplotlib axes object for the PCA plot.
    pca : sklearn.decomposition.PCA
        The fitted PCA model.
    transformed_data : np.ndarray
        The PCA-transformed data.
    """
    # Convert data to DataFrame if it's a numpy array
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(
            data,
            index=[f"Analyte_{i}" for i in range(data.shape[0])],
            columns=[f"Sample_{i}" for i in range(data.shape[1])],
        )

    # Use column names as sample labels if not provided
    if sample_labels is None:
        sample_labels = data.columns.tolist()

    # Preprocessing
    processed_data = _clr_transform(data)

    # Perform PCA
    pca_result, transformed_data = _perform_pca(
        processed_data, n_components, random_state
    )

    # Create plot
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot PCA results
    scatter_plot = _plot_pca_results(
        ax,
        transformed_data,
        sample_labels,
        group_labels,
        group_colors,
        point_size,
        confidence_ellipse,
    )

    # Add loadings if requested
    if plot_loadings and isinstance(data, pd.DataFrame):
        _add_loadings_to_plot(ax, pca_result, processed_data.index)

    # Style the plot
    _style_pca_plot(ax, pca_result, title)

    # Add legend if groups are provided
    if group_labels is not None:
        _add_legend_to_plot(ax, scatter_plot)

    plt.tight_layout()
    return fig, ax, pca_result, transformed_data


def _clr_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply center log ratio (CLR) transformation columnwise to the input array.

    Parameters
    ----------
    data : pd.DataFrame
        Input data where columns are samples and rows are lipid analytes.

    Returns
    -------
    pd.DataFrame
        Preprocessed data.

    """
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()

    # Handle NaN values if present
    if processed_data.isna().any().any():
        processed_data = processed_data.fillna(0)

    # Handle zeros for log transformation if needed
    if (processed_data == 0).any().any():
        min_nonzero = processed_data[processed_data > 0].min().min()
        min_value = max(min_nonzero / 10, 1e-10)
        processed_data = processed_data.replace(0, min_value)

    # Apply CLR transformation
    # Log transform
    log_data = np.log(processed_data)

    # Calculate geometric mean of each sample (column)
    geom_means = log_data.mean(axis=0)

    # CLR transformation: log(x_i) - mean(log(x))
    processed_data = log_data.sub(geom_means, axis=1)

    return processed_data

def _perform_pca(
    data: pd.DataFrame, n_components: int = 2, random_state: int = 42
) -> Tuple[PCA, np.ndarray]:
    """
    Perform PCA on the preprocessed data.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data.
    n_components : int, default=2
        Number of principal components to calculate.
    random_state : int, default=42
        Random seed for PCA.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        The fitted PCA model.
    transformed_data : np.ndarray
        The PCA-transformed data.
    """
    # Ensure data is suitable for PCA
    data_array = np.asarray(data.T)  # Transpose to have samples as rows

    # Choose appropriate number of components
    actual_n_components = min(n_components, min(data_array.shape))
    if actual_n_components < n_components:
        print(
            f"Note: Using {actual_n_components} components (requested {n_components})."
        )

    # Perform PCA
    pca = PCA(n_components=actual_n_components, random_state=random_state)
    transformed_data = pca.fit_transform(data_array)

    # If we only got one component, add a zero column for 2D plotting
    if transformed_data.shape[1] == 1:
        padding = np.zeros((transformed_data.shape[0], 1))
        transformed_data = np.hstack((transformed_data, padding))

    return pca, transformed_data


def _plot_pca_results(
    ax: plt.Axes,
    transformed_data: np.ndarray,
    sample_labels: List[str],
    group_labels: Optional[List[str]] = None,
    group_colors: Optional[Dict[str, str]] = None,
    point_size: int = 70,
    confidence_ellipse: bool = True,
) -> PathCollection:
    """
    Plot PCA results with sample points.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on.
    transformed_data : np.ndarray
        PCA-transformed data.
    sample_labels : list of str
        Labels for each sample.
    group_labels : list of str, optional
        Group assignment for each sample.
    group_colors : dict, optional
        Mapping of group names to colors.
    point_size : int, default=70
        Size of scatter points.

    Returns
    -------
    scatter : matplotlib.collections.PathCollection
        The scatter plot object.
    """
    # If no groups provided, plot all points with same color
    if group_labels is None:
        scatter = ax.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            s=point_size,
            alpha=0.7,
            edgecolors="k",
        )

        # Add sample labels as annotations
        for i, label in enumerate(sample_labels):
            ax.annotate(
                label,
                (transformed_data[i, 0], transformed_data[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        return scatter

    # Get unique groups and create colors
    unique_groups = sorted(set(group_labels))
    if group_colors is None:
        cmap = plt.get_cmap("tab10", len(unique_groups))
        group_colors = {group: cmap(i) for i, group in enumerate(unique_groups)}

    # Create numeric values for coloring
    group_to_int = {group: i for i, group in enumerate(unique_groups)}
    c_values = [group_to_int[group] for group in group_labels]

    # Create scatter plot
    scatter = ax.scatter(
        transformed_data[:, 0],
        transformed_data[:, 1],
        c=c_values,
        cmap=plt.get_cmap("tab10", len(unique_groups)),
        s=point_size,
        alpha=0.7,
        edgecolors="k",
    )

    # Add sample labels
    for i, label in enumerate(sample_labels):
        ax.annotate(
            label,
            (transformed_data[i, 0], transformed_data[i, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    # Add confidence ellipses if requested and if groups are provided
    if confidence_ellipse and group_labels is not None:
        _add_confidence_ellipses(
            ax, transformed_data, group_labels, unique_groups, group_colors
        )

    return scatter


def _add_loadings_to_plot(
    ax: plt.Axes,
    pca: PCA,
    feature_names: pd.Index,
    top_n: int = 3,
) -> None:
    """
    Add the top feature loadings to the PCA plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on.
    pca : sklearn.decomposition.PCA
        The fitted PCA model.
    feature_names : pd.Index
        Names of the features (analytes).
    top_n : int, default=5
        Number of top loadings to display for each component.

    """
    # Skip if we don't have enough components
    if pca.components_.shape[0] < 2:
        return

    # Get loadings for first two components
    loadings = pca.components_.T[:, :2]

    # Scale the loadings for visualization
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    scale = min(x_range, y_range) / loadings.max() * 0.5

    # Get top loadings for each component
    top_x_idx = np.argsort(np.abs(loadings[:, 0]))[-top_n:]
    top_y_idx = np.argsort(np.abs(loadings[:, 1]))[-top_n:]
    top_idx = np.union1d(top_x_idx, top_y_idx)

    # Plot loadings
    for i in top_idx:
        ax.arrow(
            0,
            0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="r",
            alpha=0.5,
            width=0.002,
            head_width=0.05,
        )
        ax.text(
            loadings[i, 0] * scale*1.1,
            loadings[i, 1] * scale*1.1,
            feature_names[i],
            color="r",
            ha="center",
            va="center",
            fontsize=8,
        )


def _style_pca_plot(ax: plt.Axes, pca: PCA, title: str) -> None:
    """
    Style the PCA plot with labels and grid.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to style.
    pca : sklearn.decomposition.PCA
        The fitted PCA model (for explained variance).
    title : str
        Plot title.
    """
    # Get explained variance if available
    explained_variance = pca.explained_variance_ratio_ * 100

    # Set axis labels
    if len(explained_variance) >= 1:
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.1f}%)")
    else:
        ax.set_xlabel("PC1")

    if len(explained_variance) >= 2:
        ax.set_ylabel(f"PC2 ({explained_variance[1]:.1f}%)")
    else:
        ax.set_ylabel("PC2")

    # Set title
    ax.set_title(title)

    # Add grid and origin lines
    ax.grid(visible=True, linestyle="--", alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)


def _add_legend_to_plot(ax: plt.Axes, scatter: PathCollection) -> None:
    """
    Add a legend to the PCA plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add the legend to.
    scatter : matplotlib.collections.PathCollection
        The scatter plot object.
    """
    handles, labels = scatter.legend_elements(prop="colors")
    ax.legend(handles, labels, title="Groups", loc="best")


def _add_confidence_ellipses(
    ax: plt.Axes,
    transformed_data: np.ndarray,
    group_labels: List[str],
    unique_groups: List[str],
    group_colors: Dict[str, str],
) -> None:
    """
    Add confidence ellipses around groups in the PCA plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    transformed_data : np.ndarray
        PCA-transformed data.
    group_labels : list of str
        Group assignment for each sample.
    unique_groups : list of str
        List of unique group names.
    group_colors : dict
        Mapping of group names to colors.

    """
    for group in unique_groups:
        # Get indices of samples in this group
        indices = [i for i, g in enumerate(group_labels) if g == group]

        # Skip if fewer than 3 samples (need at least 3 for covariance)
        if len(indices) < 3:
            continue

        # Get coordinates for this group
        x = transformed_data[indices, 0]
        y = transformed_data[indices, 1]

        # Calculate mean and covariance
        mean_x, mean_y = np.mean(x), np.mean(y)

        try:
            cov = np.cov(x, y)

            # Calculate eigenvalues and eigenvectors
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)

            # Sort eigenvalues and eigenvectors
            idx = lambda_.argsort()[::-1]
            lambda_ = lambda_[idx]
            v = v[:, idx]

            # Angle of the ellipse
            theta = np.degrees(np.arctan2(v[1, 0], v[0, 0]))

            # Draw the ellipse (95% confidence)
            ellipse = plt.matplotlib.patches.Ellipse(
                xy=(mean_x, mean_y),
                width=lambda_[0] * 3.92,  # 95% confidence (1.96*2)
                height=lambda_[1] * 3.92,
                angle=theta,
                alpha=0.2,
                color=group_colors[group],
                fill=True,
            )
            ax.add_patch(ellipse)
        except (ValueError, np.linalg.LinAlgError):
            # Skip if there's an error calculating the ellipse
            continue


# Example usage with more distinct group separation
if __name__ == "__main__":
    # Generate synthetic lipidomics data with better group separation
    np.random.seed(42)

    # Number of lipid analytes and samples
    n_analytes = 100
    n_samples = 20

    # Create base data
    data = np.random.random((n_analytes, n_samples)) * 0.1

    # Create two distinct groups (samples columns 0-9 and 10-19)
    group_labels = ["Control"] * (n_samples // 2) + ["Treatment"] * (n_samples // 2)

    # Make group separation more clear by creating specific patterns
    # Add strong upregulation of certain lipids in Treatment group
    treatment_markers = np.random.choice(range(n_analytes), 15, replace=False)
    for i in range(n_samples // 2, n_samples):
        data[treatment_markers, i] += (
            np.random.random(len(treatment_markers)) * 0.8 + 0.2
        )

    # Add strong upregulation of different lipids in Control group
    control_markers = np.random.choice(
        [i for i in range(n_analytes) if i not in treatment_markers], 15, replace=False
    )
    for i in range(0, n_samples // 2):
        data[control_markers, i] += np.random.random(len(control_markers)) * 0.8 + 0.2

    # Add some zeros to simulate missing values
    data[np.random.rand(*data.shape) < 0.1] = 0

    # Convert to DataFrame
    lipid_names = [f"Lipid_{i}" for i in range(n_analytes)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]
    df = pd.DataFrame(data, index=lipid_names, columns=sample_names)

    # Create PCA plot
    fig, ax, pca_model, transformed_data = create_pca_plot(
        df,
        sample_labels=sample_names,
        group_labels=group_labels,
        title="PCA of Synthetic Lipidomics Data",
        plot_loadings=True,
        confidence_ellipse=True,
    )

    plt.show()