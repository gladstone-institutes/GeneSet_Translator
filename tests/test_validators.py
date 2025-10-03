"""Tests for input validators.

Tests:
- Gene list validation
- Disease CURIE validation
- CURIE format validation
- Parameter validation

Phase 2 Status: Stub created
TODO: Implement tests
"""

import pytest
from biograph_explorer.utils import validate_gene_list, validate_disease_curie, ValidationError


class TestGeneListValidation:
    """Test suite for gene list validation."""

    def test_validate_valid_gene_list(self):
        """Test validation of valid gene list."""
        pytest.skip("TODO: Implement test")

    def test_validate_empty_gene_list(self):
        """Test that empty gene list raises ValidationError."""
        pytest.skip("TODO: Implement test")

    def test_validate_too_many_genes(self):
        """Test that gene list exceeding max_genes raises ValidationError."""
        pytest.skip("TODO: Implement test")

    def test_gene_list_trimming(self):
        """Test that gene symbols are trimmed and cleaned."""
        pytest.skip("TODO: Implement test")

    def test_gene_list_uppercasing(self):
        """Test that gene symbols are converted to uppercase."""
        pytest.skip("TODO: Implement test")

    def test_duplicate_genes(self):
        """Test handling of duplicate gene symbols."""
        pytest.skip("TODO: Implement test")


class TestDiseaseCURIEValidation:
    """Test suite for disease CURIE validation."""

    def test_validate_valid_mondo_curie(self):
        """Test validation of valid MONDO CURIE."""
        pytest.skip("TODO: Implement test")

    def test_validate_valid_doid_curie(self):
        """Test validation of valid DOID CURIE."""
        pytest.skip("TODO: Implement test")

    def test_validate_invalid_curie_format(self):
        """Test that invalid CURIE format raises ValidationError."""
        pytest.skip("TODO: Implement test")

    def test_validate_empty_curie(self):
        """Test that empty string raises ValidationError."""
        pytest.skip("TODO: Implement test")


class TestCURIEValidation:
    """Test suite for generic CURIE validation."""

    def test_valid_curie_formats(self):
        """Test various valid CURIE formats."""
        valid_curies = [
            "NCBIGene:1803",
            "MONDO:0004975",
            "DOID:10652",
            "HGNC:1234",
            "UniProtKB:P12345",
        ]
        pytest.skip("TODO: Implement test")

    def test_invalid_curie_formats(self):
        """Test various invalid CURIE formats."""
        invalid_curies = [
            "not-a-curie",
            ":12345",
            "PREFIX:",
            "12345",
            "",
        ]
        pytest.skip("TODO: Implement test")


class TestParameterValidation:
    """Test suite for parameter validation."""

    def test_convergence_threshold_validation(self):
        """Test convergence threshold validation."""
        pytest.skip("TODO: Implement test")

    def test_negative_threshold(self):
        """Test that negative threshold raises ValidationError."""
        pytest.skip("TODO: Implement test")

    def test_excessive_threshold(self):
        """Test that excessively high threshold raises ValidationError."""
        pytest.skip("TODO: Implement test")
