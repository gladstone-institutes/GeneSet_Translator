"""LLM-assisted category summaries with citation graphs using Claude Haiku 4.

This module generates verifiable category-specific summaries with XML-based
citation extraction. LLM identifies relevant nodes by CURIE, and the system
extracts all edge/publication/sentence data directly from the graph.
Uses token-aware sampling to optimize costs while maintaining analytical depth.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import networkx as nx
from pydantic import BaseModel, Field

from ..utils.biolink_predicates import get_predicate_info

logger = logging.getLogger(__name__)


class CitationGraph(BaseModel):
    """Citation graph linking claims to nodes/edges/publications."""

    citation_id: int = Field(description="Unique citation ID")
    claim: str = Field(description="The specific claim being cited")
    node_ids: List[str] = Field(default_factory=list, description="Node CURIEs from LLM")
    edge_ids: List[str] = Field(default_factory=list, description="Edge IDs extracted from graph")
    context_node_ids: List[str] = Field(default_factory=list, description="Additional context nodes")
    publication_ids: List[str] = Field(default_factory=list, description="PMIDs extracted from graph")
    sentences: List[str] = Field(default_factory=list, description="Supporting text extracted from graph")
    confidence: str = Field(default="medium", description="Citation confidence: low/medium/high")


class SummaryData(BaseModel):
    """Complete summary with text and citation graphs."""

    category: str = Field(description="Node category (e.g., Protein, BiologicalProcess)")
    summary_text: str = Field(description="Summary text with [Citation N] markers")
    citations: List[CitationGraph] = Field(default_factory=list, description="Citation graphs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Tokens, sampling strategy, etc.")


class LLMSummarizer:
    """Generate citation-based category summaries using Claude Haiku 4."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        cache_dir: Path = Path("data/cache")
    ):
        """Initialize LLM summarizer.

        Args:
            model: Claude model to use
            cache_dir: Directory for caching summaries and citations

        Note:
            Requires ANTHROPIC_API_KEY environment variable to be set.
            The Anthropic SDK will automatically read it from the environment.
        """
        try:
            from anthropic import Anthropic
            from dotenv import load_dotenv

            load_dotenv()
            # Anthropic SDK automatically reads from ANTHROPIC_API_KEY env var
            self.client = Anthropic()
            logger.info(f"Anthropic client initialized successfully")
        except ImportError:
            raise ImportError("anthropic package required. Install with: poetry add anthropic")

        self.model = model
        self.summary_cache_dir = cache_dir / "summaries"
        self.citation_cache_dir = cache_dir / "citation_graphs"
        self.summary_cache_dir.mkdir(parents=True, exist_ok=True)
        self.citation_cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_category_summary(
        self,
        graph: nx.MultiDiGraph,
        category: str,
        query_genes: List[str],
        disease_curie: str,
        infores_metadata: Optional[Dict] = None
    ) -> SummaryData:
        """Generate citation-based summary for a specific category using JSON context.

        Args:
            graph: Full knowledge graph
            category: Node category to summarize (e.g., "Protein", "BiologicalProcess")
            query_genes: List of input gene CURIEs
            disease_curie: Target disease CURIE
            infores_metadata: Optional knowledge source metadata

        Returns:
            SummaryData with summary text and citations
        """
        # Check cache first
        cache_key = self._generate_cache_key(query_genes, disease_curie, category, graph)
        cached_summary = self._load_from_cache(cache_key, category)
        if cached_summary:
            logger.info(f"Using cached summary for {category}")
            cached_summary.metadata['from_cache'] = True
            return cached_summary

        # Filter to category nodes
        category_nodes = [
            node for node, data in graph.nodes(data=True)
            if data.get('category') == category
        ]

        if not category_nodes:
            logger.warning(f"No nodes found for category {category}")
            return SummaryData(
                category=category,
                summary_text=f"No {category} nodes found in the knowledge graph.",
                citations=[],
                metadata={'error': 'no_nodes'}
            )

        logger.info(f"Generating summary for {category}: {len(category_nodes)} nodes")

        # Sample top intermediates by query gene connectivity (matches visualization approach)
        max_nodes = 20  # Top 20 most connected category intermediates
        sampled_subgraph = self._sample_nodes_by_query_gene_connections(
            graph,
            category_nodes,
            query_genes,
            max_nodes,
            disease_curie=disease_curie
        )

        # Prepare JSON context
        context = self._prepare_json_context(
            sampled_subgraph,
            category_nodes,
            query_genes,
            disease_curie,
            category,
            infores_metadata
        )

        # Log token usage
        token_count = self._estimate_json_tokens(context)
        logger.info(f"JSON context for {category}: {token_count} tokens, "
                   f"{len(context['nodes'])} nodes, {len(context['edges'])} edges")

        # Generate with citations
        summary_text, citations = self._generate_with_citations(context, category, graph)

        # Create summary data
        summary_data = SummaryData(
            category=category,
            summary_text=summary_text,
            citations=citations,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'nodes_total': len(category_nodes),
                'nodes_sampled': sampled_subgraph.number_of_nodes(),
                'edges_sampled': sampled_subgraph.number_of_edges(),
                'max_nodes': max_nodes,
                'sampling_strategy': 'query_gene_connectivity',
                'token_count': token_count,
                'model': self.model,
                'format_version': 'json_v3',
                'from_cache': False
            }
        )

        # Cache the result
        self._save_to_cache(cache_key, category, summary_data)

        return summary_data

    def _estimate_json_tokens(self, context: Dict[str, Any]) -> int:
        """Estimate token count for JSON context using tiktoken.

        Args:
            context: JSON context dictionary

        Returns:
            Estimated token count
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            context_str = json.dumps(context, indent=2)
            return len(encoding.encode(context_str))
        except ImportError:
            # Fallback: rough estimate of 4 chars per token
            context_str = json.dumps(context, indent=2)
            return len(context_str) // 4

    def _generate_cache_key(
        self,
        query_genes: List[str],
        disease_curie: str,
        category: str,
        graph: nx.MultiDiGraph
    ) -> str:
        """Generate hash-based cache key including model and format version."""
        format_version = "json_v2"  # Increment when changing JSON structure
        key_str = f"{','.join(sorted(query_genes))}|{disease_curie}|{category}|{graph.number_of_edges()}|{self.model}|{format_version}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str, category: str) -> Optional[SummaryData]:
        """Load summary from cache if available and fresh."""
        summary_file = self.summary_cache_dir / cache_key / f"summary_{category}.json"

        if summary_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(summary_file.stat().st_mtime)
            if cache_age < timedelta(days=30):
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    return SummaryData(**data)

        return None

    def _save_to_cache(self, cache_key: str, category: str, summary_data: SummaryData):
        """Save summary to cache."""
        cache_subdir = self.summary_cache_dir / cache_key
        cache_subdir.mkdir(parents=True, exist_ok=True)

        summary_file = cache_subdir / f"summary_{category}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data.model_dump(), f, indent=2)

        # Also save citations separately for citation viewer
        citation_subdir = self.citation_cache_dir / cache_key
        citation_subdir.mkdir(parents=True, exist_ok=True)

        citation_file = citation_subdir / f"citations_{category}.json"
        with open(citation_file, 'w') as f:
            json.dump([c.model_dump() for c in summary_data.citations], f, indent=2)

        logger.info(f"Cached summary and citations for {category}")


    def _sample_nodes_by_query_gene_connections(
        self,
        graph: nx.MultiDiGraph,
        category_nodes: List[str],
        query_genes: List[str],
        max_nodes: int,
        disease_curie: Optional[str] = None
    ) -> nx.MultiDiGraph:
        """Sample category nodes by query gene connectivity (matches visualization approach).

        This sampling strategy mirrors the "Top Intermediates" approach used in the
        Network tab visualization, ensuring consistency between what users see and
        what the LLM summarizes.

        Strategy:
            1. Score each category node by number of direct edges to/from query genes
            2. Take top-N category nodes by score
            3. Always include query genes and disease node
            4. Return subgraph with selected nodes and ALL their connecting edges

        Args:
            graph: Full knowledge graph
            category_nodes: Nodes in the target category
            query_genes: Query gene CURIEs
            max_nodes: Maximum number of category nodes to include
            disease_curie: Target disease CURIE (always included if present)

        Returns:
            Subgraph with top category nodes by query gene connectivity
        """
        query_genes_set = set(query_genes)
        min_gene_frequency = 2  # Minimum gene_frequency to include in LLM context

        # Score each category node by query gene connections
        # Filter: only include nodes with gene_frequency >= 2 (convergent nodes)
        node_scores = {}
        filtered_count = 0
        for node in category_nodes:
            if node not in graph.nodes():
                continue
            node_data = graph.nodes[node]
            gene_freq = node_data.get('gene_frequency', 0)

            # Skip nodes with gene_frequency < 2 (not convergent)
            if gene_freq < min_gene_frequency:
                filtered_count += 1
                continue

            score = 0
            for gene in query_genes_set:
                if gene in graph.nodes():
                    # Count edges in both directions
                    if graph.has_edge(gene, node):
                        score += 1
                    if graph.has_edge(node, gene):
                        score += 1
            node_scores[node] = score

        logger.info(f"Filtered {filtered_count} nodes with gene_frequency < {min_gene_frequency}")

        # Sort by score (descending) and take top-N
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]

        # Build node set: top category nodes + query genes (exempt from filter) + disease
        nodes_to_include = set(node for node, _ in top_nodes)
        # Query genes are always included regardless of gene_frequency
        nodes_to_include.update(g for g in query_genes if g in graph.nodes())

        # Always include disease node
        if disease_curie and disease_curie in graph.nodes():
            nodes_to_include.add(disease_curie)

        # Create subgraph with all edges between selected nodes
        subgraph = graph.subgraph(nodes_to_include).copy()

        # Log sampling statistics
        top_score = top_nodes[0][1] if top_nodes else 0
        min_score = top_nodes[-1][1] if top_nodes else 0
        logger.info(
            f"Sampled {len(top_nodes)} category nodes from {len(node_scores)} total "
            f"(max: {max_nodes}, score range: {min_score}-{top_score})"
        )
        logger.info(
            f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges"
        )

        return subgraph


    def _prepare_json_context(
        self,
        subgraph: nx.MultiDiGraph,
        all_category_nodes: List[str],
        query_genes: List[str],
        disease_curie: str,
        category: str,
        infores_metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Prepare structured JSON context with complete edge information.

        Args:
            subgraph: Sampled subgraph with importance-scored edges
            all_category_nodes: All nodes in category (for stats)
            query_genes: Input gene CURIEs
            disease_curie: Target disease
            category: The category being summarized (e.g., "BiologicalProcess")
            infores_metadata: Knowledge source metadata

        Returns:
            Structured JSON context dictionary
        """
        # Query context
        query_context = {
            "query_genes": query_genes,
            "disease_curie": disease_curie,
            "category": category,  # Use the category parameter directly
            "total_category_nodes": len(all_category_nodes),
            "sampled_nodes": subgraph.number_of_nodes()
        }

        # Nodes
        nodes = []
        for node_id, data in subgraph.nodes(data=True):
            node_obj = {
                "curie": node_id,
                "label": data.get('label', node_id),
                "category": data.get('category', 'Unknown'),
                "gene_frequency": data.get('gene_frequency', 0)
            }

            # Mark disease node explicitly
            if node_id == disease_curie:
                node_obj['is_disease'] = True

            # Optional attributes
            if 'is_query_gene' in data:
                node_obj['is_query_gene'] = data['is_query_gene']
            if 'pagerank' in data:
                node_obj['pagerank'] = round(data['pagerank'], 4)
            if 'betweenness' in data:
                node_obj['betweenness'] = round(data['betweenness'], 4)

            # HPA annotation fields (skip GO annotations per user request)
            annotation_features = data.get('annotation_features', {})
            if annotation_features:
                # Tissue specificity
                if 'hpa_tissue_specificity' in annotation_features:
                    node_obj['tissue_specificity'] = annotation_features['hpa_tissue_specificity']
                if 'hpa_top_tissues' in annotation_features:
                    node_obj['top_tissues'] = annotation_features['hpa_top_tissues']

                # Cell type specificity
                if 'hpa_cell_type_specificity' in annotation_features:
                    node_obj['cell_type_specificity'] = annotation_features['hpa_cell_type_specificity']
                if 'hpa_top_cell_types' in annotation_features:
                    node_obj['top_cell_types'] = annotation_features['hpa_top_cell_types']

                # Immune cell specificity
                if 'hpa_immune_cell_specificity' in annotation_features:
                    node_obj['immune_cell_specificity'] = annotation_features['hpa_immune_cell_specificity']
                if 'hpa_top_immune_cells' in annotation_features:
                    node_obj['top_immune_cells'] = annotation_features['hpa_top_immune_cells']

                # Disease involvement and protein class
                if 'hpa_disease_involvement' in annotation_features:
                    node_obj['disease_involvement'] = annotation_features['hpa_disease_involvement']
                if 'hpa_protein_class' in annotation_features:
                    node_obj['protein_class'] = annotation_features['hpa_protein_class']

            nodes.append(node_obj)

        # Edges
        edges = []
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            # Format: subject→predicate→object (semantic triple order)
            predicate_for_id = key.rsplit('_', 1)[0] if '_' in key else key
            edge_id = f"{u}→{predicate_for_id}→{v}"

            # Extract sources
            sources = data.get('sources', [])
            source_list = []
            for s in sources:
                if isinstance(s, dict):
                    source_list.append({
                        "resource_id": s.get('resource_id', ''),
                        "resource_role": s.get('resource_role', '')
                    })
                else:
                    source_list.append({"resource_id": str(s)})

            # Extract publications (no truncation, but limit display to 20 for readability)
            publications = data.get('publications', [])
            pub_list = publications[:20] if len(publications) > 20 else publications

            # Extract supporting text
            supporting_text = data.get('sentences', [])
            text_list = supporting_text[:5] if len(supporting_text) > 5 else supporting_text

            # Get predicate definition from Biolink model
            predicate = data.get('predicate', 'related_to')
            predicate_name = predicate.replace('biolink:', '') if predicate else 'related_to'
            predicate_info = get_predicate_info(predicate_name)
            predicate_definition = ""
            predicate_inverse = ""
            if predicate_info:
                predicate_definition = predicate_info.get('description', '')
                predicate_inverse = predicate_info.get('inverse', '')

            edge_obj = {
                "edge_id": edge_id,
                "subject": u,
                "object": v,
                "predicate": predicate,
                "predicate_definition": predicate_definition,
                "predicate_inverse": predicate_inverse,
                "sources": source_list,
                "publications": pub_list,
                "supporting_text": text_list,
                "confidence_scores": data.get('confidence_scores', {}),
                "importance_score": round(data.get('importance_score', 0.0), 2),
                "publication_count": data.get('publication_count', len(publications)),
                "connects_to_query_genes": data.get('connects_to_query_genes', [])
            }

            edges.append(edge_obj)

        # Knowledge sources summary
        knowledge_sources = {}
        if infores_metadata:
            summary = infores_metadata.get('summary', {})
            knowledge_sources = {
                "total_sources": summary.get('total_sources', 0),
                "top_sources": [
                    {
                        "name": src['name'],
                        "resource_id": src.get('id', ''),
                        "edge_count": src['edge_count']
                    }
                    for src in summary.get('top_sources', [])[:5]
                ]
            }

        return {
            "query_context": query_context,
            "nodes": nodes,
            "edges": edges,
            "knowledge_sources": knowledge_sources
        }

    def _generate_with_citations(
        self,
        context: Dict[str, Any],
        category: str,
        graph: nx.MultiDiGraph
    ) -> Tuple[str, List[CitationGraph]]:
        """Generate summary with XML-based citation extraction.

        The LLM outputs citations with node CURIEs in XML format, and the system
        extracts all edge/publication/sentence data directly from the graph.

        Args:
            context: Prepared JSON context dictionary
            category: Node category
            graph: Full graph for citation extraction

        Returns:
            Tuple of (summary_text, citations)
        """
        # Build the XML-structured prompt
        system_prompt = self._build_system_prompt(category)

        try:
            # Convert context to JSON string
            context_json_str = json.dumps(context, indent=2)

            # Single API call with XML-structured prompt
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,  # Increased to ensure room for citations + summary
                temperature=0.1,  # Low temperature for consistent output
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": context_json_str
                }]
            )

            # Extract response text
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            # Parse XML citations and extract graph data
            # Pass query context for complete pathway extraction
            query_genes = context.get('query_context', {}).get('query_genes', [])
            disease_curie = context.get('query_context', {}).get('disease_curie')
            citations = self._parse_xml_citations(
                response_text, graph, query_genes=query_genes, disease_curie=disease_curie
            )

            # Extract summary from <summary> tags (with robust fallbacks)
            summary_text = self._extract_summary(response_text, category=category)

            if not summary_text:
                logger.warning(f"No summary found in response for {category}")
                summary_text = f"Summary extraction failed for {category}. Check data/cache/llm_debug/ for raw response."

            logger.info(f"Generated summary for {category}: {len(citations)} citations, {len(summary_text)} chars")
            return summary_text, citations

        except Exception as e:
            logger.error(f"LLM generation failed for {category}: {e}")
            return f"Failed to generate summary for {category}: {e}", []

    def _build_system_prompt(self, category: str) -> str:
        """Build the XML-structured system prompt for category summaries."""
        return f"""You are analyzing a biomedical knowledge graph to identify how intermediate nodes connect query genes to a disease of interest. Your goal is to extract well-supported biological insights and present them with proper citations.

The category of intermediate nodes you are analyzing is: {category}

The user will provide the knowledge graph data in JSON format containing:
- **query_context**: Information about query genes, the disease (disease_curie), and sampling statistics
- **nodes**: Node objects with attributes including curie, label, categories, is_disease (true for the disease node), is_query_gene (true for query genes), and HPA (Human Protein Atlas) tissue/cell specificity data
- **edges**: Edge objects with:
  - subject, object: Node CURIEs
  - predicate, predicate_definition: Relationship type and its semantic definition
  - publications: List of PMIDs supporting the edge
  - **supporting_text**: Sentences extracted from literature describing the relationship (VERY IMPORTANT for understanding mechanistic context)
  - confidence_scores: Evidence quality metrics
  - importance_score: Combined score weighing publications and query gene connections
- **knowledge_sources**: Summary of the data sources used

**IMPORTANT**: The disease node (marked with is_disease: true) and its connecting edges are included in the graph. When creating citations that trace pathways to the disease, ALWAYS include the disease CURIE in your node_curies list to ensure full traceability.

You will complete this task in TWO TURNS within a single response:

## TURN 1: SAVE CITATIONS

First, output citations for every factual claim you will make.

Focus your citations on:
1. **Convergent nodes**: Intermediate {category} nodes that connect to multiple query genes (these are the most important)
2. **Disease connections**: Edges connecting intermediate nodes to the disease - include the disease CURIE in citations to show the complete pathway
3. **Well-supported edges**: Connections with multiple publications, high confidence scores, and rich supporting_text
4. **HPA expression context**: Tissue-specific or cell-type-specific expression patterns from the HPA data
5. **Mechanistic insights**: Use the predicate_definition AND supporting_text sentences to explain the biological relationship

**IMPORTANT**: The supporting_text field contains sentences extracted from publications that describe the relationship. Use these to understand and convey the mechanistic nature of connections.

### CRITICAL CITATION REQUIREMENTS:

- Each citation must support ONE specific claim - NEVER combine multiple citations like [Citation 1, 2]
- Each citation should center on the connections between query genes and ONE intermediate node
- When multiple edges connect the same nodes, aggregate them by predicate type
- **ALWAYS include the disease CURIE** in your node_curies list to ensure complete pathway traceability
- Always include HPA tissue/cell type context when available (e.g., "Protein X is highly expressed in T cells according to HPA data")
- Use the predicate_definition to explain the semantic meaning of relationships
- Set confidence level based on publication count: "high" (>5 publications), "medium" (2-5 publications), "low" (<2 publications)

### Citation Format:

For each citation, output:
<save_citation_graph>
<citation_id>[number]</citation_id>
<claim>[The specific biological claim this citation supports]</claim>
<node_curies>["CURIE1", "CURIE2", ...]</node_curies>
<confidence>[high/medium/low based on publication count]</confidence>
</save_citation_graph>

Number your citations sequentially starting from 1.

IMPORTANT: Only include node CURIEs (e.g., "NCBIGene:6776", "UniProtKB:Q00987") in the node_curies array. The system will automatically extract all edges and publications connecting these nodes from the knowledge graph.

## TURN 2: GENERATE SUMMARY

After all citations, write a **300-500 word** summary that synthesizes the key findings.

Your summary should:
- Focus on convergent pathways that link multiple query genes to the disease through {category} intermediates
- Include specific tissue or cell type context from HPA data when it provides biological insight
- Explain the mechanistic nature of relationships (using information from predicate definitions and supporting_text)
- Use [Citation N] markers after each factual claim, with exactly ONE citation per claim
- Be written in clear, scientifically accurate prose
- Prioritize the most well-supported and biologically meaningful connections

Format your summary inside <summary> tags.

### CRITICAL - CITATION FORMAT IN SUMMARY:

**WRONG:**
- "DPP4 regulates both CXCL10 and CCL2 [Citation 2, 3]"
- "These genes converge on inflammation [Citations 1-4]"
- "Multiple pathways are involved [Citation 14, 17]"

**CORRECT:**
- "DPP4 regulates CXCL10, a key chemokine elevated in COVID-19 [Citation 2]. DPP4 also regulates CCL2, which recruits monocytes [Citation 3]."

If you need to discuss multiple related findings, write a separate sentence for each claim with its own single citation.

### Example:

<save_citation_graph>
<citation_id>1</citation_id>
<claim>BRCA1 and TP53 both interact with MDM2, a key regulator in the p53 pathway that shows high expression in lymphoid tissues per HPA</claim>
<node_curies>["NCBIGene:672", "NCBIGene:7157", "UniProtKB:Q00987", "MONDO:0007254"]</node_curies>
<confidence>high</confidence>
</save_citation_graph>

<summary>
Analysis of the knowledge graph reveals convergent pathways linking BRCA1 and TP53 to breast cancer through key protein intermediates. Both genes interact with MDM2, a critical regulator in the p53 pathway that shows high expression in lymphoid tissues according to HPA data [Citation 1].
</summary>

### Final Reminders:

- **LIMIT: Create a MAXIMUM of 10 citations** - Focus on the most important and well-supported findings
- Complete ALL citations FIRST, then write the summary
- **ONE citation per claim** - if you write [Citation X, Y] you have made a critical error
- Include the disease CURIE in every citation for complete pathway traceability
- Include HPA tissue/cell context explicitly in citations when available
- Use predicate_definition and supporting_text to explain relationship semantics
- Ensure every factual claim in the summary has exactly one [Citation N] marker

Begin now."""

    def _flatten_to_strings(self, data: Any) -> List[str]:
        """Flatten nested lists/structures and extract only string elements.

        Handles cases where LLM returns nested arrays like [["CURIE1", "CURIE2"]]
        or mixed types like ["CURIE1", 123, ["nested"]].

        Args:
            data: Parsed JSON data (could be list, string, or nested structure)

        Returns:
            Flat list of strings
        """
        result = []
        if isinstance(data, str):
            result.append(data)
        elif isinstance(data, list):
            for item in data:
                result.extend(self._flatten_to_strings(item))
        # Ignore non-string, non-list items (ints, dicts, None, etc.)
        return result

    def _parse_xml_citations(
        self,
        response_text: str,
        graph: nx.MultiDiGraph,
        query_genes: Optional[List[str]] = None,
        disease_curie: Optional[str] = None
    ) -> List[CitationGraph]:
        """Parse XML citations and extract full graph data for cited nodes.

        Args:
            response_text: LLM response containing XML citation blocks
            graph: Full graph for extracting edge/publication data
            query_genes: Query gene CURIEs for complete pathway extraction
            disease_curie: Disease CURIE for complete pathway extraction

        Returns:
            List of CitationGraph objects with extracted graph data
        """
        citations = []
        pattern = r'<save_citation_graph>(.*?)</save_citation_graph>'

        for match in re.finditer(pattern, response_text, re.DOTALL):
            block = match.group(1)

            # Extract citation_id
            citation_id_match = re.search(r'<citation_id>(\d+)</citation_id>', block)
            if not citation_id_match:
                logger.warning("Citation block missing citation_id, skipping")
                continue
            citation_id = int(citation_id_match.group(1))

            # Extract claim
            claim_match = re.search(r'<claim>(.*?)</claim>', block, re.DOTALL)
            claim = claim_match.group(1).strip() if claim_match else ""

            # Extract confidence
            confidence_match = re.search(r'<confidence>(.*?)</confidence>', block)
            confidence = confidence_match.group(1).strip() if confidence_match else "medium"

            # Extract node CURIEs from LLM output
            node_curies = []
            node_curies_match = re.search(r'<node_curies>(.*?)</node_curies>', block, re.DOTALL)
            if node_curies_match:
                try:
                    parsed = json.loads(node_curies_match.group(1).strip())
                    # Flatten and filter to only strings (handles nested lists, non-string items)
                    node_curies = self._flatten_to_strings(parsed)
                except json.JSONDecodeError:
                    logger.warning(f"Citation {citation_id}: Failed to parse node_curies JSON")
                    # Try to extract CURIEs using regex as fallback
                    curie_pattern = r'"([^"]+)"'
                    node_curies = re.findall(curie_pattern, node_curies_match.group(1))

            # Extract ALL edges, publications, and sentences from graph for cited nodes
            # Include query genes and disease for complete pathway traceability
            edge_ids, publication_ids, sentences, avg_importance = self._extract_edges_for_nodes(
                graph, node_curies, query_genes=query_genes, disease_curie=disease_curie
            )

            # Auto-adjust confidence based on extracted evidence
            # Override LLM confidence if evidence strongly disagrees
            pub_count = len(publication_ids)
            if avg_importance > 7.0 and pub_count > 5:
                adjusted_confidence = "high"
            elif avg_importance > 4.0 or pub_count > 2:
                adjusted_confidence = "medium"
            elif pub_count < 2:
                adjusted_confidence = "low"
            else:
                adjusted_confidence = confidence  # Keep LLM's assessment

            citations.append(CitationGraph(
                citation_id=citation_id,
                claim=claim,
                node_ids=node_curies,
                edge_ids=edge_ids,
                publication_ids=publication_ids,
                sentences=sentences,
                confidence=adjusted_confidence
            ))

        logger.info(f"Parsed {len(citations)} citations from XML response")
        return citations

    def _extract_edges_for_nodes(
        self,
        graph: nx.MultiDiGraph,
        node_curies: List[str],
        query_genes: Optional[List[str]] = None,
        disease_curie: Optional[str] = None
    ) -> Tuple[List[str], List[str], List[str], float]:
        """Extract all edges, publications, and sentences connecting the cited nodes.

        Automatically includes query genes and disease for complete pathway traceability.
        This ensures citation graphs show the full path from query genes to disease.

        Args:
            graph: Full knowledge graph
            node_curies: List of node CURIEs from the citation
            query_genes: Query gene CURIEs to include for complete pathways
            disease_curie: Disease CURIE to include for complete pathways

        Returns:
            Tuple of (edge_ids, publication_ids, sentences, avg_importance_score)
        """
        node_set = set(node_curies)

        # Auto-include disease and query genes for complete pathway traceability
        if disease_curie and disease_curie in graph.nodes():
            node_set.add(disease_curie)
        if query_genes:
            node_set.update(g for g in query_genes if g in graph.nodes())

        edge_ids = []
        all_publications = []
        all_sentences = []
        importance_scores = []

        # Find all edges where BOTH endpoints are in the cited node set
        for u, v, key, data in graph.edges(keys=True, data=True):
            if u in node_set and v in node_set:
                # Format: subject→predicate→object (semantic triple order)
                predicate_for_id = key.rsplit('_', 1)[0] if '_' in key else key
                edge_id = f"{u}→{predicate_for_id}→{v}"
                edge_ids.append(edge_id)

                # Collect publications from this edge
                pubs = data.get('publications', [])
                all_publications.extend(pubs)

                # Collect sentences/supporting_text from this edge
                sentences = data.get('sentences', [])
                if sentences:
                    all_sentences.extend(sentences)

                # Track importance scores for confidence adjustment
                importance = data.get('importance_score', 0.0)
                importance_scores.append(importance)

        # Deduplicate publications (these should be strings like "PMID:12345")
        unique_publications = list(set(p for p in all_publications if isinstance(p, str)))

        # Deduplicate sentences - handle potential non-string items
        seen_sentences = set()
        unique_sentences = []
        for s in all_sentences:
            if isinstance(s, str) and s not in seen_sentences:
                seen_sentences.add(s)
                unique_sentences.append(s)

        # Calculate average importance
        avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0.0

        return edge_ids, unique_publications, unique_sentences, avg_importance

    def _extract_summary(self, response_text: str, category: str = "") -> str:
        """Extract summary from LLM response using multiple strategies.

        Tries extraction patterns in order of priority:
        1. Exact <summary>...</summary> tags
        2. Case-insensitive summary tags
        3. Summary tags with attributes
        4. Markdown "## Summary" or "**Summary:**" sections
        5. Everything after the last </save_citation_graph> tag
        6. Paragraphs containing [Citation N] markers

        Args:
            response_text: Full LLM response text
            category: Category name for logging

        Returns:
            Extracted summary text (never empty - falls back to heuristic extraction)
        """
        # Log raw response for debugging
        self._log_raw_response(response_text, category)

        # Strategy 1: Exact <summary>...</summary> tags
        match = re.search(r'<summary>(.*?)</summary>', response_text, re.DOTALL)
        if match:
            logger.info(f"Summary extracted using Strategy 1 (exact tags)")
            return match.group(1).strip()

        # Strategy 2: Case-insensitive summary tags
        match = re.search(r'<summary>(.*?)</summary>', response_text, re.DOTALL | re.IGNORECASE)
        if match:
            logger.info(f"Summary extracted using Strategy 2 (case-insensitive)")
            return match.group(1).strip()

        # Strategy 3: Summary tags with attributes (e.g., <summary type="...">)
        match = re.search(r'<summary[^>]*>(.*?)</summary>', response_text, re.DOTALL | re.IGNORECASE)
        if match:
            logger.info(f"Summary extracted using Strategy 3 (tags with attributes)")
            return match.group(1).strip()

        # Strategy 4: Markdown section headers
        # Look for "## Summary", "### Summary", "**Summary:**", etc.
        patterns = [
            r'##\s*Summary\s*\n(.*?)(?=\n##|\n\*\*|$)',  # ## Summary header
            r'\*\*Summary:?\*\*\s*\n?(.*?)(?=\n\*\*|\n##|$)',  # **Summary:** bold
            r'Summary:\s*\n(.*?)(?=\n##|\n\*\*|$)',  # Plain "Summary:" label
        ]
        for i, pattern in enumerate(patterns, start=1):
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                if len(text) > 100:  # Minimum viable summary length
                    logger.info(f"Summary extracted using Strategy 4.{i} (markdown section)")
                    return text

        # Strategy 5: Extract everything after the last citation block
        # This assumes the summary comes after all citations
        last_citation_end = response_text.rfind('</save_citation_graph>')
        if last_citation_end != -1:
            after_citations = response_text[last_citation_end + len('</save_citation_graph>'):].strip()
            # Clean up markdown/XML artifacts
            after_citations = re.sub(r'^#+\s*TURN\s*2.*?\n', '', after_citations, flags=re.IGNORECASE)
            after_citations = re.sub(r'^#+\s*GENERATE\s*SUMMARY.*?\n', '', after_citations, flags=re.IGNORECASE)
            after_citations = after_citations.strip()
            if len(after_citations) > 100:
                logger.info(f"Summary extracted using Strategy 5 (after last citation)")
                return after_citations

        # Strategy 6: If nothing else works, try to find the longest paragraph
        # that contains citation markers [Citation N]
        paragraphs = response_text.split('\n\n')
        citation_paragraphs = [p for p in paragraphs if re.search(r'\[Citation \d+\]', p)]
        if citation_paragraphs:
            # Join paragraphs with citations
            summary = '\n\n'.join(citation_paragraphs)
            logger.info(f"Summary extracted using Strategy 6 (citation paragraphs)")
            return summary.strip()

        logger.error(f"All extraction strategies failed for {category}")
        return ""

    def _log_raw_response(self, response_text: str, category: str) -> None:
        """Log raw LLM response to file for debugging.

        Args:
            response_text: Full LLM response text
            category: Category name for filename
        """
        try:
            from pathlib import Path
            from datetime import datetime as dt

            log_dir = Path("data/cache/llm_debug")
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"raw_response_{category}_{timestamp}.txt"

            with open(log_file, 'w') as f:
                f.write(f"Category: {category}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Response length: {len(response_text)} chars\n")
                f.write("=" * 80 + "\n")
                f.write(response_text)

            logger.info(f"Raw LLM response logged to {log_file}")
        except Exception as e:
            logger.warning(f"Failed to log raw response: {e}")
