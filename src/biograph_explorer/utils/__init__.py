"""Utility modules for BioGraph Explorer.

Provides:
- Input validation for gene lists and disease CURIEs
- Data formatters for display
- Persistence (pickle NetworkX graphs, JSON caching)

Phase 2 Status: Stubs created
"""

from .validators import validate_gene_list, validate_disease_curie, ValidationError
from .formatters import format_node_label, format_edge_label, format_clustering_results
from .persistence import save_graph, load_graph, save_session, load_session

__all__ = [
    "validate_gene_list",
    "validate_disease_curie",
    "ValidationError",
    "format_node_label",
    "format_edge_label",
    "format_clustering_results",
    "save_graph",
    "load_graph",
    "save_session",
    "load_session",
]
