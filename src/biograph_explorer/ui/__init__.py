"""Streamlit UI modules for BioGraph Explorer.

Provides:
- Input panel for gene/disease entry
- Query status and progress tracking
- Results overview dashboard
- Convergent nodes table view
- PyVis network visualization
- RAG chat interface (Phase 3)

Phase 2 Status: Stubs created, Streamlit integration pending
"""

from .network_viz import (
    render_network_visualization,
    export_visualization_html,
    sample_graph_for_visualization,
    get_node_details,
    create_clustered_graph,
)

# Stub modules (Phase 2 - not yet implemented)
# from .input_panel import render_input_panel
# from .query_status import render_query_status
# from .results_overview import render_results_overview
# from .convergence_view import render_convergence_view

# Phase 3
# from .rag_chat import render_rag_chat

__all__ = [
    "render_network_visualization",
    "export_visualization_html",
    "sample_graph_for_visualization",
    "get_node_details",
    "create_clustered_graph",
]
