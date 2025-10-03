# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioGraph Explorer is a Streamlit application for multi-gene TRAPI query integration with NetworkX clustering and LLM-assisted exploration using PyVis visualization. The project explores biomedical knowledge graphs by querying the NCATS Translator system to find convergent pathways and therapeutic targets.

**Current Status**: Phase 1 (Foundation & POC) completed. The codebase currently consists of a working Jupyter notebook prototype (`notebooks/multi_gene_pathfinder.ipynb`). Phase 2 will involve refactoring into a modular Python package with Streamlit UI.

## Development Commands

```bash
# Add a package
poetry add TCT
```

### Environment Setup
```bash
# Create and activate conda environment
conda create -n biograph_explorer python=3.10
conda activate biograph_explorer

# Install with Poetry
poetry install

# Install TCT library (required for TRAPI queries)
pip install TCT

# Install visualization dependencies
pip install ipycytoscape streamlit pyvis
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_biograph_explorer.py

# Run with coverage
pytest --cov=src/biograph_explorer
```

### Documentation
```bash
# Build documentation (ReadTheDocs)
cd docs
make html
```

## Architecture Overview

The project follows a planned architecture (see [PROJECT_PLAN.md](PROJECT_PLAN.md)) with these core components:

### Planned Module Structure
```
biograph_explorer/
├── core/
│   ├── trapi_client.py          # TRAPI query & caching
│   ├── graph_builder.py         # TRAPI → NetworkX conversion
│   ├── clustering_engine.py     # NetworkX analysis (centrality, communities)
│   └── rag_system.py            # Claude + NetworkX RAG with PyVis
├── ui/
│   ├── input_panel.py           # Gene/disease input
│   ├── network_viz.py           # PyVis rendering
│   └── rag_chat.py              # Visual RAG chat interface
├── utils/
│   ├── validators.py            # Input validation
│   ├── formatters.py            # Data formatting
│   └── persistence.py           # Pickle NetworkX graphs
└── data/
    ├── cache/                   # TRAPI response cache
    ├── sessions/                # Pickled NetworkX graphs
    └── exports/                 # HTML/PNG exports
```

### Current Implementation (Phase 1)

All functionality is currently in `notebooks/multi_gene_pathfinder.ipynb`:
- **Gene Normalization**: Uses TCT's `name_resolver.batch_lookup()` to normalize HUGO symbols → CURIEs
- **TRAPI Querying**: Neighborhood discovery pattern (empty target list) to find all connections
- **API Integration**: Parallel queries to 15 Translator APIs with graceful degradation
- **Visualization**: ipycytoscape graph with 3-tier color coding (DE genes, disease, other nodes)
- **Data Persistence**: JSON storage of raw TRAPI responses in `data/raw/`

### Key Technical Patterns

**TRAPI Query Strategy**:
- Use empty target list `[]` for neighborhood discovery (finds all connections, not just direct paths)
- Batch query genes in parallel with `translator_query.parallel_api_query()`
- Accept partial success (6/15 APIs succeeding is normal and sufficient)
- Cache all responses to avoid re-querying

**Graph Construction**:
- Build NetworkX DiGraph from TRAPI edge data
- Use CURIEs as node IDs for analysis
- Add rich node attributes: labels, categories (Gene/Disease/Other), is_query_gene flag
- Preserve edge attributes: predicates, qualifiers, publications, knowledge_source

**Visualization Sampling**:
- Guarantee ≥2 edges per query gene to prevent missing nodes
- Sample high-degree edges to fill budget
- Use circle layout (fast, user can rearrange)
- Color: Red (DE genes), Purple (disease), Blue (other)

## Data Files

### Input
- Test gene lists defined in notebooks (e.g., 10 COVID-19 DE genes from PMC11255397)
- Expected format: List of HUGO gene symbols

### Output
- `data/raw/tct_results_YYYYMMDD_HHMMSS.json`: Full TRAPI query results
- `data/processed/`: (Future) Processed graphs and clustering results
- `data/exports/`: (Future) HTML/PNG visualizations

## Test Case: Alzheimer's Disease

The primary validation dataset uses 15 Alzheimer's genes (APOE, APP, PSEN1, PSEN2, MAPT, TREM2, etc.) queried against MONDO:0004975.

**Expected Results**:
- Convergent nodes: BACE1, amyloid-β, cholesterol pathway proteins
- 3-4 communities: amyloid processing, lipid metabolism, neuroinflammation
- Top targets: BACE1, APOE, TREM2

**Validation**: Results should match known biology (80%+ accuracy)

## Phase 2 Goals (Next Steps - See CURRENT_PROGRESS.md)

1. **Graph Clustering**: Louvain community detection, identify hub nodes
2. **LLM Integration**: Claude Haiku 4 for cluster summarization
3. **Streamlit UI**: Input panel, results dashboard, interactive PyVis visualization
4. **RAG System**: 3-layer context strategy with citation validation

## Dependencies

Core dependencies (see `pyproject.toml`):
- Python ^3.10
- TCT (Translator Clinical Tools) - for TRAPI queries
- NetworkX - graph analysis
- python-louvain - community detection
- PyVis - interactive visualization
- Streamlit - web UI (planned)
- Anthropic - Claude API (planned)

## Common Development Patterns

### Running the Notebook
```bash
jupyter notebook notebooks/multi_gene_pathfinder.ipynb
```

### Querying New Gene Sets
1. Define gene list (HUGO symbols)
2. Normalize with `name_resolver.batch_lookup()`
3. Query with empty target list for neighborhood discovery
4. Sample edges ensuring all query genes are represented
5. Visualize with ipycytoscape

### Handling API Errors
- "NoneType is not iterable" = API returned no data (normal)
- Expect 40-60% API success rate
- Use `max_workers=5` to avoid network saturation
- All responses are cached for retry

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| 15-gene TRAPI batch | <5 min | With caching/parallelization |
| Clustering analysis | <30 sec | NetworkX in-memory |
| PyVis render (<100 nodes) | <3 sec | Physics stabilization |
| LLM response | <10 sec | Claude Haiku |

## Model Selection

**Claude Haiku 4** chosen for RAG system:
- Task is information extraction, not complex reasoning
- Graph analysis done by NetworkX
- Tool use for structured citations
- Cost: ~$0.02 per session (5 questions × 12K input + 1K output)

## Important Notes

- This is early-stage research code transitioning from notebook to package
- The `src/biograph_explorer/` directory is currently empty (Phase 1 uses notebooks)
- Refer to `PROJECT_PLAN.md` for full architecture and implementation roadmap
- Refer to `CURRENT_PROGRESS.md` for development status and next tasks
