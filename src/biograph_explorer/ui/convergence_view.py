"""Convergent nodes table view for Streamlit UI.

Features:
- Sortable/filterable table of convergent nodes
- Gene frequency filtering slider
- Category filtering (Gene, Protein, etc.)
- Centrality metrics display
- Click to highlight in visualization
- Export to CSV

Phase 2 Status: Stub created
TODO: Implement Streamlit table components
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd


def render_convergence_view(
    convergent_nodes: List[Dict[str, Any]],
    min_gene_frequency: Optional[int] = None,
) -> Optional[str]:
    """Render convergent nodes table with filtering.

    Args:
        convergent_nodes: List of convergent node dicts
        min_gene_frequency: Optional minimum gene frequency filter

    Returns:
        Selected node ID if user clicks a row, else None

    Example:
        >>> selected_node = render_convergence_view(results.convergent_nodes)
        >>> if selected_node:
        ...     # Highlight in visualization
        ...     pass

    TODO: Implement Streamlit table with filtering
    """
    raise NotImplementedError("TODO: Implement convergence view table")


def create_convergent_nodes_dataframe(
    convergent_nodes: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Create DataFrame from convergent nodes for display.

    Args:
        convergent_nodes: List of convergent node dicts

    Returns:
        Formatted pandas DataFrame

    TODO: Implement DataFrame creation
    """
    raise NotImplementedError("TODO: Implement DataFrame creation")


def render_gene_frequency_filter(
    max_frequency: int,
) -> int:
    """Render gene frequency filter slider.

    Args:
        max_frequency: Maximum gene frequency in dataset

    Returns:
        Selected minimum gene frequency

    TODO: Implement slider filter
    """
    raise NotImplementedError("TODO: Implement frequency filter slider")


def render_category_filter(
    available_categories: List[str],
) -> List[str]:
    """Render category multiselect filter.

    Args:
        available_categories: List of available node categories

    Returns:
        List of selected categories

    TODO: Implement multiselect filter
    """
    raise NotImplementedError("TODO: Implement category filter")


def render_export_csv_button(df: pd.DataFrame, filename: str = "convergent_nodes.csv") -> None:
    """Render button to export table as CSV.

    Args:
        df: DataFrame to export
        filename: Output filename

    TODO: Implement CSV export button
    """
    raise NotImplementedError("TODO: Implement CSV export button")


def apply_filters(
    df: pd.DataFrame,
    min_gene_frequency: Optional[int] = None,
    selected_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply filters to convergent nodes DataFrame.

    Args:
        df: Input DataFrame
        min_gene_frequency: Minimum gene frequency threshold
        selected_categories: List of categories to include

    Returns:
        Filtered DataFrame

    TODO: Implement filtering logic
    """
    raise NotImplementedError("TODO: Implement filtering logic")
