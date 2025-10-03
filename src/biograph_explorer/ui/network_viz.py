"""PyVis network visualization for Streamlit UI.

Features:
- Interactive PyVis graph rendering
- Physics-based layouts (forceAtlas2, barnesHut)
- Node sizing by centrality or gene frequency
- Node coloring by category
- Hover tooltips with node properties
- Clickable nodes showing source/provenance details
- Export to standalone HTML

Phase 2 Status: Implemented
"""

from typing import Optional, List, Dict, Any, Tuple
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pathlib import Path
import logging

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Color scheme for node categories
CATEGORY_COLORS = {
    "Gene": "#E74C3C",  # Red
    "Disease": "#9B59B6",  # Purple
    "Protein": "#4ECDC4",  # Cyan
    "ChemicalEntity": "#F39C12",  # Yellow/Orange
    "BiologicalProcess": "#2ECC71",  # Green
    "Cluster": "#16A085",  # Teal (for cluster meta-nodes)
    "Other": "#3498DB",  # Blue
}

# Shape scheme for node categories
CATEGORY_SHAPES = {
    "Gene": "box",  # Rectangle
    "Disease": "diamond",
    "Protein": "ellipse",
    "ChemicalEntity": "ellipse",
    "BiologicalProcess": "ellipse",
    "Cluster": "hexagon",  # Hexagon for clusters
    "Other": "dot",
}


def render_network_visualization(
    graph: nx.DiGraph,
    query_genes: List[str],
    sizing_metric: str = "gene_frequency",
    highlight_nodes: Optional[List[str]] = None,
    max_nodes: int = 200,
    height: str = "750px",
    freeze_layout: bool = True,
) -> Optional[str]:
    """Render interactive PyVis network visualization.

    Args:
        graph: NetworkX DiGraph to visualize
        query_genes: List of query gene node IDs (highlighted)
        sizing_metric: Node sizing metric (gene_frequency, pagerank, betweenness, degree)
        highlight_nodes: Optional list of node IDs to highlight (for citations)
        max_nodes: Maximum nodes to display (warn if exceeded)
        height: Height of visualization in CSS units
        freeze_layout: If True, disable physics after stabilization

    Returns:
        HTML string of PyVis graph, or None if error

    Example:
        >>> html = render_network_visualization(graph, ["NCBIGene:1803"])
        >>> components.html(html, height=750)
    """
    if not PYVIS_AVAILABLE:
        logger.error("PyVis not installed")
        return None

    if graph.number_of_nodes() == 0:
        logger.warning("Empty graph - nothing to visualize")
        return None

    try:
        # Sample graph if too large
        if graph.number_of_nodes() > max_nodes:
            logger.info(f"Graph has {graph.number_of_nodes()} nodes - sampling to {max_nodes}")
            graph = sample_graph_for_visualization(graph, query_genes, max_edges=max_nodes)

        # Create PyVis network
        net = create_pyvis_graph(
            graph,
            query_genes,
            sizing_metric=sizing_metric,
            highlight_nodes=highlight_nodes or [],
        )

        # Configure physics
        configure_physics(net, graph.number_of_nodes(), freeze_after_stabilization=freeze_layout)

        # Generate HTML
        html = net.generate_html()

        return html

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_pyvis_graph(
    graph: nx.DiGraph,
    query_genes: List[str],
    sizing_metric: str = "gene_frequency",
    highlight_nodes: Optional[List[str]] = None,
) -> "Network":
    """Create PyVis Network from NetworkX graph.

    Args:
        graph: NetworkX DiGraph
        query_genes: Query gene node IDs
        sizing_metric: Metric for node sizing
        highlight_nodes: Nodes to highlight

    Returns:
        PyVis Network object
    """
    if not PYVIS_AVAILABLE:
        raise ImportError("PyVis not installed. Run: pip install pyvis")

    highlight_nodes = highlight_nodes or []

    # Create PyVis network
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#0E1117",  # Match Streamlit dark theme
        font_color="#FAFAFA",
    )

    # Normalize sizing metric values for consistent node sizes
    metric_values = {}
    if sizing_metric in ["pagerank", "betweenness", "degree"]:
        for node in graph.nodes():
            metric_values[node] = graph.nodes[node].get(sizing_metric, 0)
    elif sizing_metric == "gene_frequency":
        for node in graph.nodes():
            metric_values[node] = graph.nodes[node].get("gene_frequency", 0)
    else:
        # Default to degree
        metric_values = dict(graph.degree())

    # Normalize to 0-1 range
    if metric_values:
        max_val = max(metric_values.values()) if max(metric_values.values()) > 0 else 1
        normalized_metrics = {k: v / max_val for k, v in metric_values.items()}
    else:
        normalized_metrics = {node: 0.5 for node in graph.nodes()}

    # Add nodes with styling
    for node in graph.nodes():
        node_attrs = graph.nodes[node]
        is_query_gene = node_attrs.get("is_query_gene", False)
        is_highlighted = node in highlight_nodes
        category = node_attrs.get("category", "Other")

        # Get style properties
        style = style_node(node, graph, is_query_gene, is_highlighted, normalized_metrics[node], category)

        # Create tooltip
        tooltip = create_node_tooltip(node, graph)

        # Use original symbol for query genes, otherwise use label
        original_symbol = node_attrs.get("original_symbol", "")
        display_label = original_symbol if original_symbol else node_attrs.get("label", node)

        # Add node to PyVis network
        net.add_node(
            node,
            label=display_label,
            title=tooltip,
            color=style["color"],
            size=style["size"],
            shape=style["shape"],
            borderWidth=style["borderWidth"],
            borderWidthSelected=style["borderWidthSelected"],
        )

    # Add edges
    for source, target, edge_attrs in graph.edges(data=True):
        predicate = edge_attrs.get("predicate", "")
        predicate_label = predicate.replace("biolink:", "")

        net.add_edge(
            source,
            target,
            title=predicate_label,
            label=predicate_label if len(predicate_label) < 30 else "",
            color="#666666",
            width=1.5,
            arrows="to",
        )

    return net


def configure_physics(net: "Network", num_nodes: int, freeze_after_stabilization: bool = True) -> None:
    """Configure physics settings based on graph size.

    Args:
        net: PyVis Network object
        num_nodes: Number of nodes in graph
        freeze_after_stabilization: If True, disable physics after layout stabilizes
    """
    import json

    # Stabilize with physics, then optionally freeze to prevent excessive movement
    physics_enabled = not freeze_after_stabilization

    physics_config = {
        "physics": {
            "enabled": physics_enabled,
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 150,  # Increased from 100
                "springConstant": 0.05,  # Reduced from 0.08
                "damping": 0.9,  # Increased from 0.4 for stability
                "avoidOverlap": 1.0,  # Increased from 0.5
            },
            "maxVelocity": 30,  # Reduced from 50
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": True,
                "iterations": 200,  # Increased from 150
                "updateInterval": 25,
                "fit": True,
            },
            "adaptiveTimestep": True,
        },
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "hover": True,
        },
    }

    net.set_options(json.dumps(physics_config))


def style_node(
    node_id: str,
    graph: nx.DiGraph,
    is_query_gene: bool,
    is_highlighted: bool,
    metric_normalized: float,
    category: str,
) -> Dict[str, Any]:
    """Generate PyVis styling for a node.

    Args:
        node_id: Node identifier
        graph: NetworkX graph containing node
        is_query_gene: Whether this is a query gene
        is_highlighted: Whether to highlight (for citations)
        metric_normalized: Normalized metric value (0-1)
        category: Node category (Gene, Disease, etc.)

    Returns:
        Dictionary with PyVis node properties (color, size, shape, etc.)
    """
    # Base size: 15-45px based on metric (from PROJECT_PLAN.md)
    size = 15 + (metric_normalized * 30)

    # Query genes get larger minimum size
    if is_query_gene:
        size = max(size, 25)

    # Highlighted nodes get even larger
    if is_highlighted:
        size = size * 1.3

    # Color by category
    color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["Other"])

    # Query genes get brighter color
    if is_query_gene:
        color = CATEGORY_COLORS.get(category, "#E74C3C")  # Ensure query genes are visible

    # Shape by category
    shape = CATEGORY_SHAPES.get(category, "dot")

    # Border width
    borderWidth = 4 if is_query_gene else 2
    borderWidthSelected = 6 if is_query_gene else 4

    return {
        "color": color,
        "size": size,
        "shape": shape,
        "borderWidth": borderWidth,
        "borderWidthSelected": borderWidthSelected,
    }


def create_node_tooltip(node_id: str, graph: nx.DiGraph) -> str:
    """Create plain text tooltip for node hover.

    PyVis doesn't render HTML in tooltips properly, so we use plain text.

    Args:
        node_id: Node identifier
        graph: NetworkX graph containing node

    Returns:
        Plain text string for tooltip
    """
    attrs = graph.nodes[node_id]

    # Extract attributes
    label = attrs.get("label", node_id)
    original_symbol = attrs.get("original_symbol", "")
    category = attrs.get("category", "Unknown")
    is_query = attrs.get("is_query_gene", False)

    # Check if this is a cluster meta-node
    if category == "Cluster":
        # Special tooltip for cluster nodes
        size = attrs.get("size", 0)
        density = attrs.get("density", 0)
        query_gene_count = attrs.get("query_gene_count", 0)
        top_nodes = attrs.get("top_nodes", [])

        lines = [
            label,
            "━" * 30,
            f"Type: Community Cluster",
            f"Size: {size} nodes",
            f"Density: {density:.3f}",
            f"Query Genes: {query_gene_count}",
        ]

        if top_nodes:
            lines.append("━" * 30)
            lines.append("Top Nodes:")
            for node_label in top_nodes[:3]:
                if node_label:
                    lines.append(f"  • {node_label[:30]}")

        return "\n".join(lines)
    else:
        # Regular node tooltip
        gene_freq = attrs.get("gene_frequency", 0)
        pagerank = attrs.get("pagerank", 0)
        betweenness = attrs.get("betweenness", 0)
        degree = graph.degree(node_id)

        # Build plain text tooltip
        lines = [
            label if not original_symbol else f"{original_symbol} ({label})",
            "━" * 30,
            f"ID: {node_id}",
            f"Category: {category}",
        ]

        if is_query:
            lines.append("✓ Query Gene")

        lines.extend([
            "━" * 30,
            f"Gene Frequency: {gene_freq}",
            f"PageRank: {pagerank:.4f}",
            f"Betweenness: {betweenness:.1f}",
            f"Degree: {degree}",
            "━" * 30,
            "Click node for detailed info",
        ])

        return "\n".join(lines)


def export_visualization_html(
    graph: nx.DiGraph,
    query_genes: List[str],
    output_path: Path,
) -> Path:
    """Export standalone HTML visualization.

    Args:
        graph: NetworkX graph to visualize
        query_genes: Query gene node IDs
        output_path: Output file path

    Returns:
        Path to exported HTML file
    """
    html = render_network_visualization(graph, query_genes)

    if html:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Exported visualization to {output_path}")
        return output_path
    else:
        raise ValueError("Failed to generate visualization HTML")


def sample_graph_for_visualization(
    graph: nx.DiGraph,
    query_genes: List[str],
    max_edges: int = 50,
) -> nx.DiGraph:
    """Sample graph edges for visualization while preserving query genes.

    Uses strategy from notebook cell 14:
    - Guarantee ≥2 edges per query gene
    - Fill remaining budget with high-degree edges

    Args:
        graph: Full NetworkX graph
        query_genes: Query gene node IDs
        max_edges: Maximum edges to include

    Returns:
        Sampled subgraph
    """
    if graph.number_of_edges() <= max_edges:
        return graph

    logger.info(f"Sampling graph: {graph.number_of_edges()} edges → {max_edges} edges")

    # Get node degrees for prioritization
    degrees = dict(graph.degree())

    # STEP 1: Guarantee edges for each query gene
    required_edges = []
    genes_in_graph = [g for g in query_genes if g in graph.nodes()]

    logger.info(f"Found {len(genes_in_graph)}/{len(query_genes)} query genes in graph")

    for gene in genes_in_graph:
        # Get all edges involving this gene
        gene_edges = list(graph.in_edges(gene, data=True)) + list(graph.out_edges(gene, data=True))

        if gene_edges:
            # Score edges by neighbor degree (prefer high-connectivity neighbors)
            scored_edges = []
            for src, tgt, data in gene_edges:
                neighbor = tgt if src == gene else src
                score = degrees.get(neighbor, 0)
                scored_edges.append((score, (src, tgt, data)))

            # Take top 2 edges for this gene
            scored_edges.sort(reverse=True, key=lambda x: x[0])
            required_edges.extend([edge for _, edge in scored_edges[:min(2, len(scored_edges))]])

    logger.info(f"Selected {len(required_edges)} edges covering query genes")

    # STEP 2: Fill remaining budget with high-degree edges
    remaining_budget = max_edges - len(required_edges)

    if remaining_budget > 0:
        # Get all other edges
        required_edge_set = set((src, tgt) for src, tgt, _ in required_edges)
        other_edges = [
            (src, tgt, data)
            for src, tgt, data in graph.edges(data=True)
            if (src, tgt) not in required_edge_set
        ]

        # Score edges by total endpoint degree
        scored_other = [
            (degrees.get(src, 0) + degrees.get(tgt, 0), (src, tgt, data))
            for src, tgt, data in other_edges
        ]
        scored_other.sort(reverse=True, key=lambda x: x[0])

        # Add top edges to fill budget
        additional_edges = [edge for _, edge in scored_other[:remaining_budget]]
        sampled_edges = required_edges + additional_edges
    else:
        sampled_edges = required_edges[:max_edges]

    # Build sampled subgraph
    sampled_graph = nx.DiGraph()

    for src, tgt, data in sampled_edges:
        # Copy edge with all attributes
        sampled_graph.add_edge(src, tgt, **data)

    # Copy node attributes
    for node in sampled_graph.nodes():
        if node in graph.nodes():
            sampled_graph.nodes[node].update(graph.nodes[node])

    logger.info(
        f"Sampled graph: {sampled_graph.number_of_nodes()} nodes, {sampled_graph.number_of_edges()} edges"
    )

    # Verify query gene coverage
    genes_in_sample = sum(1 for g in genes_in_graph if g in sampled_graph.nodes())
    logger.info(f"Query genes in sample: {genes_in_sample}/{len(genes_in_graph)}")

    return sampled_graph


def get_node_details(node_id: str, graph: nx.DiGraph) -> Dict[str, Any]:
    """Extract detailed information about a node.

    Args:
        node_id: Node identifier (CURIE)
        graph: NetworkX graph containing the node

    Returns:
        Dictionary with node details including edges and sources
    """
    if node_id not in graph.nodes():
        return {"error": f"Node {node_id} not found in graph"}

    attrs = graph.nodes[node_id]

    # Get edges
    in_edges = list(graph.in_edges(node_id, data=True))
    out_edges = list(graph.out_edges(node_id, data=True))

    # Group edges by predicate
    edges_by_predicate = {}
    all_sources = set()

    for src, tgt, data in in_edges + out_edges:
        predicate = data.get("predicate", "unknown").replace("biolink:", "")
        sources = data.get("knowledge_source", [])

        if predicate not in edges_by_predicate:
            edges_by_predicate[predicate] = []

        edge_info = {
            "direction": "incoming" if tgt == node_id else "outgoing",
            "source" if tgt == node_id else "target": src if tgt == node_id else tgt,
            "source_label" if tgt == node_id else "target_label": graph.nodes[src if tgt == node_id else tgt].get(
                "label", src if tgt == node_id else tgt
            ),
            "sources": sources,
        }
        edges_by_predicate[predicate].append(edge_info)

        # Collect all sources
        all_sources.update(sources)

    return {
        "node_id": node_id,
        "label": attrs.get("label", node_id),
        "original_symbol": attrs.get("original_symbol", ""),
        "category": attrs.get("category", "Unknown"),
        "is_query_gene": attrs.get("is_query_gene", False),
        "metrics": {
            "gene_frequency": attrs.get("gene_frequency", 0),
            "pagerank": attrs.get("pagerank", 0),
            "betweenness": attrs.get("betweenness", 0),
            "degree": graph.degree(node_id),
            "in_degree": graph.in_degree(node_id),
            "out_degree": graph.out_degree(node_id),
        },
        "edges_by_predicate": edges_by_predicate,
        "total_edges": len(in_edges) + len(out_edges),
        "knowledge_sources": list(all_sources),
    }


def create_clustered_graph(
    graph: nx.DiGraph, clustering_results, query_genes: List[str]
) -> nx.DiGraph:
    """Create a meta-graph where nodes represent clusters.

    Args:
        graph: Original NetworkX graph
        clustering_results: ClusteringResults from clustering_engine
        query_genes: List of query gene node IDs

    Returns:
        Meta-graph with cluster nodes
    """
    meta_graph = nx.DiGraph()

    # Create a node for each community
    for community in clustering_results.communities:
        cluster_id = f"cluster_{community.community_id}"

        # Get cluster statistics
        cluster_nodes = community.nodes
        subgraph = graph.subgraph(cluster_nodes)

        # Count query genes in cluster
        query_genes_in_cluster = [n for n in cluster_nodes if n in query_genes]

        # Get top nodes by PageRank
        top_nodes = community.top_nodes[:5] if community.top_nodes else []
        top_labels = [node_info.get("label", "") for node_info in top_nodes]

        # Add meta-node
        meta_graph.add_node(
            cluster_id,
            label=f"Cluster {community.community_id}\n({len(cluster_nodes)} nodes)",
            original_symbol="",  # No original symbol for clusters
            category="Cluster",  # Special category for clusters
            is_query_gene=False,
            size=len(cluster_nodes),
            density=community.density,
            top_nodes=top_labels,
            query_gene_count=len(query_genes_in_cluster),
            node_ids=cluster_nodes,  # Store original node IDs
            # Metrics for visualization
            gene_frequency=len(query_genes_in_cluster),
            pagerank=0,
            betweenness=0,
        )

    # Add edges between clusters
    community_map = {}  # node_id → community_id
    for community in clustering_results.communities:
        for node in community.nodes:
            community_map[node] = community.community_id

    # Count inter-cluster edges
    inter_cluster_edges = {}  # (comm1, comm2) → count

    for src, tgt in graph.edges():
        src_comm = community_map.get(src)
        tgt_comm = community_map.get(tgt)

        if src_comm is not None and tgt_comm is not None and src_comm != tgt_comm:
            edge_key = tuple(sorted([src_comm, tgt_comm]))
            inter_cluster_edges[edge_key] = inter_cluster_edges.get(edge_key, 0) + 1

    # Add inter-cluster edges to meta-graph
    for (comm1, comm2), count in inter_cluster_edges.items():
        meta_graph.add_edge(
            f"cluster_{comm1}",
            f"cluster_{comm2}",
            weight=count,
            label=f"{count} edges",
        )

    return meta_graph
