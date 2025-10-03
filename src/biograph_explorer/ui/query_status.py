"""Query progress and status tracking UI.

Features:
- Real-time progress bar for TRAPI queries
- API success/failure status table
- Gene normalization status
- Estimated time remaining
- Error reporting with helpful messages

Phase 2 Status: Stub created
TODO: Implement Streamlit progress components
"""

from typing import Dict, List, Optional
import streamlit as st


def render_query_status(
    stage: str,
    progress: float,
    status_message: str,
    details: Optional[Dict[str, any]] = None,
) -> None:
    """Render query progress status.

    Args:
        stage: Current stage (normalizing, querying, building_graph, clustering)
        progress: Progress value 0.0-1.0
        status_message: Status message to display
        details: Optional details dict (API counts, errors, etc.)

    Example:
        >>> render_query_status(
        ...     stage="querying",
        ...     progress=0.6,
        ...     status_message="Querying 15 Translator APIs...",
        ...     details={"apis_queried": 9, "apis_succeeded": 5}
        ... )

    TODO: Implement Streamlit progress components
    """
    raise NotImplementedError("TODO: Implement query status rendering")


def render_normalization_status(
    total_genes: int,
    normalized_genes: int,
    failed_genes: Optional[List[str]] = None,
) -> None:
    """Render gene normalization status.

    Args:
        total_genes: Total number of genes
        normalized_genes: Number successfully normalized
        failed_genes: Optional list of failed gene symbols

    TODO: Implement normalization status display
    """
    raise NotImplementedError("TODO: Implement normalization status")


def render_api_status_table(
    api_statuses: List[Dict[str, any]],
) -> None:
    """Render table of API query statuses.

    Args:
        api_statuses: List of dicts with api_name, status, edge_count, error_msg

    TODO: Implement API status table
    """
    raise NotImplementedError("TODO: Implement API status table")


def render_error_message(
    error: Exception,
    recovery_suggestions: Optional[List[str]] = None,
) -> None:
    """Render error message with recovery suggestions.

    Args:
        error: Exception that occurred
        recovery_suggestions: Optional list of suggestions for user

    TODO: Implement error message display
    """
    raise NotImplementedError("TODO: Implement error message rendering")


def create_progress_placeholder() -> any:
    """Create Streamlit placeholder for progress updates.

    Returns:
        Streamlit placeholder object

    TODO: Implement progress placeholder creation
    """
    raise NotImplementedError("TODO: Implement progress placeholder")
