"""Graph clustering and centrality analysis engine.

Implements:
- Louvain community detection
- Centrality metrics (PageRank, betweenness, degree, closeness)
- Convergent node identification (gene_frequency filtering)
- Graph statistics (density, modularity, clustering coefficient)

Phase 2 Status: Implemented
"""

from typing import Dict, List, Any, Optional
import networkx as nx
from pydantic import BaseModel, Field
import logging

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class CommunityInfo(BaseModel):
    """Information about a detected community."""

    community_id: int = Field(description="Community identifier")
    nodes: List[str] = Field(description="Node IDs in this community")
    size: int = Field(description="Number of nodes")
    density: float = Field(description="Internal edge density")
    top_nodes: List[Dict[str, Any]] = Field(default_factory=list, description="High-centrality nodes")


class ClusteringResults(BaseModel):
    """Results from clustering analysis."""

    communities: List[CommunityInfo] = Field(description="Detected communities")
    num_communities: int = Field(description="Total number of communities")
    modularity: float = Field(description="Graph modularity score")
    convergent_nodes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Nodes with high gene frequency"
    )
    centrality_scores: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Centrality metrics by node"
    )
    graph_stats: Dict[str, Any] = Field(default_factory=dict, description="Overall graph statistics")


class ClusteringEngine:
    """Engine for graph clustering and analysis.

    Example:
        >>> engine = ClusteringEngine(algorithm="louvain")
        >>> results = engine.analyze_graph(graph, query_genes=["NCBIGene:1803"])
        >>> print(f"Detected {results.num_communities} communities")
        >>> print(f"Top convergent node: {results.convergent_nodes[0]}")
    """

    def __init__(
        self,
        algorithm: str = "louvain",
        convergence_threshold: int = 2,
        centrality_metrics: Optional[List[str]] = None,
    ):
        """Initialize clustering engine.

        Args:
            algorithm: Community detection algorithm (default: "louvain")
            convergence_threshold: Min gene frequency for convergent nodes
            centrality_metrics: List of centrality metrics to compute
        """
        self.algorithm = algorithm
        self.convergence_threshold = convergence_threshold
        self.centrality_metrics = centrality_metrics or ["pagerank", "betweenness", "degree"]

        if algorithm == "louvain" and not LOUVAIN_AVAILABLE:
            raise ImportError("python-louvain not installed. Run: poetry add python-louvain")

    def analyze_graph(
        self,
        graph: nx.DiGraph,
        query_genes: List[str],
    ) -> ClusteringResults:
        """Perform complete clustering analysis on graph.

        Args:
            graph: NetworkX DiGraph to analyze
            query_genes: List of query gene node IDs

        Returns:
            ClusteringResults with communities, centrality, convergence
        """
        logger.info(f"Analyzing graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Compute centrality metrics
        centrality_scores = self.compute_centrality(graph, self.centrality_metrics)

        # Detect communities
        community_partition = self.detect_communities(graph)
        modularity = self._compute_modularity(graph.to_undirected(), community_partition)

        # Build community info
        communities = self._build_community_info(graph, community_partition, centrality_scores)

        # Identify convergent nodes (if gene_frequency attribute exists)
        convergent_nodes = []
        if graph.nodes and "gene_frequency" in next(iter(graph.nodes(data=True)))[1]:
            convergent_nodes = self.identify_convergent_nodes(graph, query_genes, self.convergence_threshold)

        # Compute graph statistics
        graph_stats = self.compute_graph_statistics(graph)

        logger.info(f"Analysis complete: {len(communities)} communities, modularity={modularity:.3f}")

        return ClusteringResults(
            communities=communities,
            num_communities=len(communities),
            modularity=modularity,
            convergent_nodes=convergent_nodes,
            centrality_scores=centrality_scores,
            graph_stats=graph_stats,
        )

    def detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """Detect communities using Louvain algorithm.

        Args:
            graph: NetworkX graph (converted to undirected if needed)

        Returns:
            Dictionary mapping node ID to community ID
        """
        # Convert to undirected for community detection
        if isinstance(graph, nx.DiGraph):
            undirected = graph.to_undirected()
        else:
            undirected = graph

        logger.info("Detecting communities using Louvain algorithm...")

        # Run Louvain
        partition = community_louvain.best_partition(undirected)

        logger.info(f"Detected {len(set(partition.values()))} communities")

        return partition

    def compute_centrality(
        self,
        graph: nx.DiGraph,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute centrality metrics for all nodes.

        Args:
            graph: NetworkX DiGraph
            metrics: List of metrics to compute (pagerank, betweenness, degree, closeness)

        Returns:
            Nested dict: {node_id: {metric_name: score}}
        """
        if metrics is None:
            metrics = self.centrality_metrics

        logger.info(f"Computing centrality metrics: {metrics}")

        # Initialize results
        centrality = {node: {} for node in graph.nodes()}

        # Compute each metric
        if "pagerank" in metrics:
            pr = nx.pagerank(graph)
            for node, score in pr.items():
                centrality[node]["pagerank"] = score

        if "betweenness" in metrics:
            bc = nx.betweenness_centrality(graph)
            for node, score in bc.items():
                centrality[node]["betweenness"] = score

        if "degree" in metrics:
            for node, degree in graph.degree():
                centrality[node]["degree"] = float(degree)

        if "closeness" in metrics:
            # Closeness requires connected graph
            if nx.is_weakly_connected(graph):
                cc = nx.closeness_centrality(graph)
                for node, score in cc.items():
                    centrality[node]["closeness"] = score
            else:
                logger.warning("Graph not connected, skipping closeness centrality")

        return centrality

    def identify_convergent_nodes(
        self,
        graph: nx.DiGraph,
        query_genes: List[str],
        threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Identify convergent nodes (high gene frequency).

        Args:
            graph: NetworkX DiGraph with gene_frequency node attribute
            query_genes: List of query gene node IDs
            threshold: Min gene frequency (default: use self.convergence_threshold)

        Returns:
            List of dicts with node_id, label, gene_frequency, categories
        """
        if threshold is None:
            threshold = self.convergence_threshold

        logger.info(f"Identifying convergent nodes (threshold >= {threshold})...")

        convergent = []
        for node in graph.nodes():
            freq = graph.nodes[node].get("gene_frequency", 0)
            if freq >= threshold:
                convergent.append(
                    {
                        "node_id": node,
                        "label": graph.nodes[node].get("label", node),
                        "gene_frequency": freq,
                        "category": graph.nodes[node].get("category", "Unknown"),
                    }
                )

        # Sort by gene frequency descending
        convergent.sort(key=lambda x: x["gene_frequency"], reverse=True)

        logger.info(f"Found {len(convergent)} convergent nodes")

        return convergent

    def compute_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute overall graph statistics.

        Returns dict with:
            - num_nodes, num_edges
            - density
            - diameter (if connected)
            - average_clustering_coefficient
            - num_connected_components

        Args:
            graph: NetworkX DiGraph

        Returns:
            Dictionary of graph statistics
        """
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
        }

        # Check connectivity
        if nx.is_weakly_connected(graph):
            stats["is_connected"] = True
            stats["num_components"] = 1
            # Diameter (expensive for large graphs)
            if graph.number_of_nodes() < 500:
                try:
                    stats["diameter"] = nx.diameter(graph.to_undirected())
                except:
                    stats["diameter"] = None
        else:
            stats["is_connected"] = False
            stats["num_components"] = nx.number_weakly_connected_components(graph)

        # Clustering coefficient
        undirected = graph.to_undirected()
        stats["avg_clustering_coefficient"] = nx.average_clustering(undirected)

        return stats

    def _compute_modularity(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
    ) -> float:
        """Compute modularity score for community structure.

        Args:
            graph: NetworkX graph
            communities: Node to community mapping

        Returns:
            Modularity score (higher = better community structure)
        """
        return community_louvain.modularity(communities, graph)

    def _build_community_info(
        self,
        graph: nx.DiGraph,
        partition: Dict[str, int],
        centrality_scores: Dict[str, Dict[str, float]],
    ) -> List[CommunityInfo]:
        """Build CommunityInfo objects from partition.

        Args:
            graph: NetworkX graph
            partition: Node to community mapping
            centrality_scores: Centrality scores for ranking nodes

        Returns:
            List of CommunityInfo objects
        """
        # Group nodes by community
        communities_dict = {}
        for node, comm_id in partition.items():
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(node)

        # Build CommunityInfo for each
        communities = []
        for comm_id, nodes in communities_dict.items():
            # Calculate internal density
            subgraph = graph.subgraph(nodes)
            density = nx.density(subgraph) if len(nodes) > 1 else 0.0

            # Get top nodes by PageRank
            top_nodes = []
            if "pagerank" in centrality_scores.get(nodes[0], {}):
                node_scores = [(n, centrality_scores[n].get("pagerank", 0)) for n in nodes]
                node_scores.sort(key=lambda x: x[1], reverse=True)
                top_nodes = [
                    {
                        "node_id": n,
                        "label": graph.nodes[n].get("label", n),
                        "pagerank": score,
                    }
                    for n, score in node_scores[:5]
                ]

            communities.append(
                CommunityInfo(
                    community_id=comm_id,
                    nodes=nodes,
                    size=len(nodes),
                    density=density,
                    top_nodes=top_nodes,
                )
            )

        # Sort by size descending
        communities.sort(key=lambda x: x.size, reverse=True)

        return communities
