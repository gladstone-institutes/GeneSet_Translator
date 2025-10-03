"""Tests for graph builder.

Tests:
- Graph construction from TRAPI edges
- Node attribute annotation
- Gene frequency calculation
- Subgraph extraction
- Node name lookup

Phase 2 Status: Stub created
TODO: Implement tests
"""

import pytest
import networkx as nx
from biograph_explorer.core import GraphBuilder, KnowledgeGraph


class TestGraphBuilder:
    """Test suite for GraphBuilder."""

    @pytest.fixture
    def builder(self):
        """Create test GraphBuilder instance."""
        pytest.skip("TODO: Implement test fixture")

    @pytest.fixture
    def sample_edges(self):
        """Sample TRAPI edges for testing."""
        pytest.skip("TODO: Implement test fixture")

    @pytest.fixture
    def query_genes(self):
        """Sample query gene CURIEs."""
        return ["NCBIGene:1803", "NCBIGene:6776", "NCBIGene:10410"]

    def test_build_from_trapi_edges(self, builder, sample_edges, query_genes):
        """Test building NetworkX graph from TRAPI edges."""
        pytest.skip("TODO: Implement test")

    def test_node_attributes(self, builder, sample_edges, query_genes):
        """Test that nodes have correct attributes (label, category, is_query_gene)."""
        pytest.skip("TODO: Implement test")

    def test_edge_attributes(self, builder, sample_edges, query_genes):
        """Test that edges preserve TRAPI attributes (predicate, etc.)."""
        pytest.skip("TODO: Implement test")

    def test_gene_frequency_calculation(self, builder, query_genes):
        """Test gene frequency (convergence) calculation."""
        pytest.skip("TODO: Implement test")

    def test_extract_subgraph_1hop(self, builder):
        """Test 1-hop subgraph extraction."""
        pytest.skip("TODO: Implement test")

    def test_extract_subgraph_2hop(self, builder):
        """Test 2-hop subgraph extraction."""
        pytest.skip("TODO: Implement test")

    def test_node_name_lookup(self, builder):
        """Test node name lookup via TCT."""
        pytest.skip("TODO: Implement test")

    def test_empty_edges(self, builder, query_genes):
        """Test handling of empty edge list."""
        pytest.skip("TODO: Implement test")


class TestKnowledgeGraph:
    """Test suite for KnowledgeGraph model."""

    def test_knowledge_graph_serialization(self):
        """Test KnowledgeGraph metadata serialization."""
        pytest.skip("TODO: Implement test")

    def test_node_category_counts(self):
        """Test node category counting."""
        pytest.skip("TODO: Implement test")
