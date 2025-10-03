"""Gene/disease input panel for Streamlit UI.

Features:
- Text area for gene list input (comma or newline separated)
- Disease CURIE input with validation
- Pre-defined example datasets (COVID-19, Alzheimer's)
- Input validation with helpful error messages
- Gene normalization preview

Phase 2 Status: Stub created
TODO: Implement Streamlit components
"""

from typing import Dict, List, Optional, Tuple
import streamlit as st


def render_input_panel() -> Tuple[Optional[List[str]], Optional[str]]:
    """Render gene/disease input panel.

    Returns:
        Tuple of (gene_list, disease_curie) or (None, None) if incomplete

    Example:
        >>> genes, disease = render_input_panel()
        >>> if genes and disease:
        ...     # Proceed with query
        ...     pass

    TODO: Implement Streamlit input components
    """
    raise NotImplementedError("TODO: Implement Streamlit input panel")


def render_gene_input() -> Optional[List[str]]:
    """Render gene list input component.

    Returns:
        List of gene symbols or None if invalid

    TODO: Implement gene input text area
    """
    raise NotImplementedError("TODO: Implement gene input")


def render_disease_input() -> Optional[str]:
    """Render disease CURIE input component.

    Returns:
        Disease CURIE or None if invalid

    TODO: Implement disease input field
    """
    raise NotImplementedError("TODO: Implement disease input")


def render_example_datasets() -> Optional[Dict[str, any]]:
    """Render example dataset selector.

    Returns:
        Dictionary with gene_list and disease_curie, or None

    TODO: Implement example dataset dropdown
    """
    # Example datasets
    EXAMPLES = {
        "COVID-19 (10 genes)": {
            "genes": ["CD6", "IFITM3", "IFITM2", "STAT5A", "KLRG1", "DPP4", "IL32", "PIK3AP1", "FYN", "IL4R"],
            "disease": "MONDO:0100096",
        },
        "Alzheimer's (15 genes)": {
            "genes": [
                "APOE",
                "APP",
                "PSEN1",
                "PSEN2",
                "MAPT",
                "TREM2",
                "CLU",
                "CR1",
                "BIN1",
                "PICALM",
                "CD33",
                "MS4A6A",
                "ABCA7",
                "SORL1",
                "BACE1",
            ],
            "disease": "MONDO:0004975",
        },
    }
    raise NotImplementedError("TODO: Implement example dataset selector")


def validate_gene_input(gene_text: str) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """Validate and parse gene input text.

    Args:
        gene_text: Raw text input from user

    Returns:
        Tuple of (is_valid, gene_list, error_message)

    TODO: Implement gene input validation
    """
    raise NotImplementedError("TODO: Implement gene input validation")


def validate_disease_input(disease_curie: str) -> Tuple[bool, Optional[str]]:
    """Validate disease CURIE input.

    Args:
        disease_curie: Disease CURIE string

    Returns:
        Tuple of (is_valid, error_message)

    TODO: Implement disease input validation
    """
    raise NotImplementedError("TODO: Implement disease input validation")
