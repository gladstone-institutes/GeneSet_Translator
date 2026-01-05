"""Publication analysis utilities for BioGraph Explorer."""
import re
from typing import Dict, List, Any, Optional
import networkx as nx

# Regex patterns for publication IDs
PMID_PATTERN = re.compile(r'PMID:?\s*(\d+)', re.IGNORECASE)
PMC_PATTERN = re.compile(r'(?:PMC|PMCID):?\s*(PMC)?(\d+)', re.IGNORECASE)


def get_publication_frequency(graph: nx.DiGraph) -> Dict[str, int]:
    """Count publication occurrences across all edges.

    Args:
        graph: NetworkX DiGraph with edge 'publications' attributes

    Returns:
        Dictionary mapping normalized publication ID to edge count
    """
    pub_counts: Dict[str, int] = {}
    for u, v, data in graph.edges(data=True):
        pubs = data.get('publications', [])
        if pubs:
            for pub in pubs:
                normalized = normalize_publication_id(pub)
                if normalized:
                    pub_counts[normalized] = pub_counts.get(normalized, 0) + 1
    return pub_counts


def normalize_publication_id(pub_id: str) -> Optional[str]:
    """Normalize publication ID format for consistent grouping.

    Args:
        pub_id: Raw publication ID string

    Returns:
        Normalized publication ID or None if invalid
    """
    if not pub_id or not isinstance(pub_id, str):
        return None
    pub_id = pub_id.strip()

    if not pub_id:
        return None

    # Already normalized PMID format
    if pub_id.upper().startswith('PMID:'):
        return pub_id.upper()

    # PMC/PMCID formats - normalize to PMC:XXXXXXX
    if pub_id.upper().startswith(('PMC:', 'PMCID:')):
        match = PMC_PATTERN.match(pub_id)
        if match:
            return f"PMC:{match.group(2)}"
        return pub_id.upper()

    # URLs - keep as-is
    if pub_id.startswith(('http://', 'https://')):
        return pub_id

    return pub_id


def format_publication_display(pub_id: str) -> str:
    """Format publication ID for UI display.

    Args:
        pub_id: Normalized publication ID

    Returns:
        Human-readable display string
    """
    if not pub_id:
        return "Unknown"

    if pub_id.startswith('PMID:'):
        return f"PMID {pub_id[5:]}"
    if pub_id.startswith('PMC:'):
        return f"PMC {pub_id[4:]}"
    if pub_id.startswith(('http://', 'https://')):
        # Extract filename or last path segment
        parts = pub_id.rstrip('/').split('/')
        return parts[-1][:30] if parts else pub_id[:30]

    return pub_id[:40]


def _extract_publications_from_attributes(attributes: List[Dict[str, Any]]) -> List[str]:
    """Extract publications using the same logic as GraphBuilder._extract_publications_robust().

    This mirrors the extraction logic to validate what SHOULD be extracted.

    Args:
        attributes: List of TRAPI attribute dictionaries

    Returns:
        List of publication IDs that should be extracted
    """
    pubs = []

    for attr in attributes:
        # Pattern 1 & 2: Top-level publications
        if attr.get('attribute_type_id') == 'biolink:publications':
            value = attr.get('value', [])
            if isinstance(value, list):
                pubs.extend(value)
            elif value:
                pubs.append(value)

        # Pattern 3: Nested in has_supporting_study_result
        if attr.get('attribute_type_id') == 'biolink:has_supporting_study_result':
            for nested in attr.get('attributes', []):
                if nested.get('attribute_type_id') == 'biolink:publications':
                    value = nested.get('value', [])
                    if isinstance(value, list):
                        pubs.extend(value)
                    elif value:
                        pubs.append(value)

    return list(set(filter(None, pubs)))


def validate_publication_extraction(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect publication loss due to edge collisions in DiGraph.

    This function analyzes raw TRAPI edges to identify where publications exist
    in the source data. When multiple edges exist between the same node pair,
    DiGraph keeps only the last edge, causing publication loss.

    Args:
        edges: List of raw TRAPI edge dictionaries

    Returns:
        Validation report dictionary with:
        - total_edges: Total raw TRAPI edges
        - edges_with_publications: Edges that have extractable publications
        - unique_node_pairs: Number of unique (subject, object) pairs
        - edges_lost_to_collisions: Edges lost due to DiGraph overwriting
        - publications_at_risk: Publications on edges that may be lost
        - sample_collisions: Sample edges involved in collisions
        - has_issues: Boolean indicating if edge collisions occurred
    """
    total_edges = len(edges)
    edges_with_pubs = 0

    # Track edges by node pair to detect collisions
    node_pair_edges: Dict[tuple, List[Dict[str, Any]]] = {}

    for edge in edges:
        attributes = edge.get('attributes', [])
        extractable_pubs = _extract_publications_from_attributes(attributes)

        if extractable_pubs:
            edges_with_pubs += 1

        # Group by node pair
        pair = (edge.get('subject'), edge.get('object'))
        if pair not in node_pair_edges:
            node_pair_edges[pair] = []
        node_pair_edges[pair].append({
            'predicate': edge.get('predicate'),
            'publications': extractable_pubs,
            'subject': edge.get('subject'),
            'object': edge.get('object'),
        })

    # Analyze collisions
    unique_pairs = len(node_pair_edges)
    collision_pairs = {k: v for k, v in node_pair_edges.items() if len(v) > 1}
    edges_lost = sum(len(v) - 1 for v in collision_pairs.values())

    # Find publications at risk (on non-final edges in collision groups)
    publications_at_risk = 0
    sample_collisions = []

    for pair, edge_list in collision_pairs.items():
        # Publications from all but the last edge are at risk
        for edge_info in edge_list[:-1]:
            publications_at_risk += len(edge_info['publications'])

        # Collect samples
        if len(sample_collisions) < 5:
            pubs_by_edge = [
                f"{e['predicate']}: {len(e['publications'])} pubs"
                for e in edge_list if e['publications']
            ]
            if pubs_by_edge:  # Only include if there are publications involved
                sample_collisions.append({
                    'subject': edge_list[0]['subject'],
                    'object': edge_list[0]['object'],
                    'edge_count': len(edge_list),
                    'predicates': [e['predicate'] for e in edge_list],
                    'pubs_by_edge': pubs_by_edge,
                    'kept_predicate': edge_list[-1]['predicate'],
                    'kept_pubs': edge_list[-1]['publications'][:3],
                })

    return {
        'total_edges': total_edges,
        'edges_with_publications': edges_with_pubs,
        'unique_node_pairs': unique_pairs,
        'edges_lost_to_collisions': edges_lost,
        'collision_pairs_count': len(collision_pairs),
        'publications_at_risk': publications_at_risk,
        'sample_collisions': sample_collisions,
        'has_issues': publications_at_risk > 0,
    }
