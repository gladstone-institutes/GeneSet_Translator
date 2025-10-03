"""Tests for TRAPI client.

Tests:
- Gene normalization
- TRAPI query construction
- API response parsing
- Caching behavior
- Error handling

Phase 2 Status: Stub created
TODO: Implement tests
"""

import pytest
from pathlib import Path
from biograph_explorer.core import TRAPIClient, TRAPIResponse


class TestTRAPIClient:
    """Test suite for TRAPIClient."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with temporary cache directory."""
        pytest.skip("TODO: Implement test fixture")

    def test_gene_normalization(self, client):
        """Test gene symbol normalization to CURIEs."""
        pytest.skip("TODO: Implement test")

    def test_normalize_invalid_genes(self, client):
        """Test handling of invalid gene symbols."""
        pytest.skip("TODO: Implement test")

    def test_query_gene_neighborhood(self, client):
        """Test gene neighborhood query construction and execution."""
        pytest.skip("TODO: Implement test")

    def test_query_with_caching(self, client, tmp_path):
        """Test that responses are cached and reused."""
        pytest.skip("TODO: Implement test")

    def test_parallel_queries(self, client):
        """Test parallel API querying with max_workers."""
        pytest.skip("TODO: Implement test")

    def test_api_error_handling(self, client):
        """Test graceful handling of API failures."""
        pytest.skip("TODO: Implement test")

    def test_empty_response_handling(self, client):
        """Test handling of empty API responses."""
        pytest.skip("TODO: Implement test")

    def test_progress_callback(self, client):
        """Test progress callback invocation during queries."""
        pytest.skip("TODO: Implement test")


class TestTRAPIResponse:
    """Test suite for TRAPIResponse model."""

    def test_response_serialization(self):
        """Test TRAPIResponse serialization to JSON."""
        pytest.skip("TODO: Implement test")

    def test_response_validation(self):
        """Test TRAPIResponse validation with invalid data."""
        pytest.skip("TODO: Implement test")
