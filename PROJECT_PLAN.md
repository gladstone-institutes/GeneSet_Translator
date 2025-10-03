# BioGraph Explorer: Project Plan

**Goal**: Streamlit application for multi-gene TRAPI query integration with NetworkX clustering and LLM-assisted exploration using PyVis visualization

**Current Status**: Phase 1 (Foundation & POC) ✅ COMPLETED | Phase 2 (Clustering) 🎯 IN PROGRESS

**Timeline**: 4 weeks MVP
**Team**: 1-2 developers
**Stack**: Python, Streamlit, NetworkX, PyVis, Claude API, TRAPI

**Working Prototype**: `notebooks/multi_gene_pathfinder.ipynb` (end-to-end TRAPI query → visualization)

---

## Architecture

```
biograph_explorer/
├── config/
│   ├── settings.py              # Configuration management
│   └── logging_config.py
├── core/
│   ├── trapi_client.py          # TRAPI query & caching
│   ├── graph_builder.py         # TRAPI → NetworkX conversion
│   ├── clustering_engine.py     # NetworkX analysis (centrality, communities)
│   └── rag_system.py            # Claude + NetworkX RAG with PyVis
├── utils/
│   ├── validators.py            # Input validation
│   ├── formatters.py            # Data formatting
│   └── persistence.py           # Pickle NetworkX graphs
├── ui/
│   ├── input_panel.py           # Gene/disease input
│   ├── query_status.py          # Progress tracking
│   ├── results_overview.py      # Summary dashboard
│   ├── convergence_view.py      # Convergent nodes table
│   ├── network_viz.py           # PyVis rendering
│   └── rag_chat.py              # Visual RAG chat interface
├── data/
│   ├── cache/                   # TRAPI response cache
│   ├── sessions/                # Pickled NetworkX graphs
│   └── exports/                 # HTML/PNG exports
└── tests/
    ├── test_trapi_client.py
    ├── test_clustering.py
    └── fixtures/
        └── alzheimers_test_case.json
```

---

### Test Gene List (15 genes)
| Gene Symbol | Gene ID | Known Role | Expected Finding |
|-------------|---------|------------|------------------|
| APOE | HGNC:613 | Lipid transport | Central hub node |
| APP | HGNC:620 | Amyloid precursor | Converges on BACE1 |
| PSEN1 | HGNC:9508 | γ-secretase component | Converges on Aβ production |
| PSEN2 | HGNC:9509 | γ-secretase component | Converges on Aβ production |
| MAPT | HGNC:6893 | Tau protein | Separate cluster from amyloid |
| TREM2 | HGNC:17761 | Microglial receptor | Inflammation pathway |
| CLU | HGNC:2095 | Clusterin/apoJ | Converges with APOE on lipid |
| CR1 | HGNC:2328 | Complement receptor | Inflammation cluster |
| BIN1 | HGNC:1052 | Endocytosis | Converges on APP trafficking |
| PICALM | HGNC:8301 | Clathrin assembly | Converges on endocytosis |
| CD33 | HGNC:1659 | Myeloid receptor | Inflammation pathway |
| MS4A6A | HGNC:13378 | Membrane protein | Immune cluster |
| ABCA7 | HGNC:29 | ATP-binding cassette | Lipid metabolism with APOE |
| SORL1 | HGNC:11140 | Sorting receptor | APP trafficking |
| BACE1 | HGNC:933 | β-secretase | Central convergence point |

**Disease**: Alzheimer's Disease (MONDO:0004975)

### Expected Outcomes (Validation Criteria)

1. **Convergent nodes**: BACE1, β-amyloid, cholesterol metabolism proteins
2. **Communities**: 
   - Cluster 1: Amyloid processing (APP, PSEN1/2, BACE1, BIN1)
   - Cluster 2: Lipid metabolism (APOE, CLU, ABCA7)
   - Cluster 3: Neuroinflammation (TREM2, CD33, CR1, MS4A6A)
3. **Top targets**: BACE1, APOE, TREM2 (all have drugs in development)
4. **Regulatory circuits**: APOE-lipid homeostasis feedback loop

## Core Components

### 1. TRAPI Client (`core/trapi_client.py`)
- Batch query genes to disease
- Rate limiting (10 queries/sec)
- Response caching
- Retry logic with exponential backoff
- Progress callbacks

### 2. Graph Builder (`core/graph_builder.py`)
- Convert TRAPI responses → NetworkX MultiDiGraph
- Track source gene → node mappings
- Calculate gene_frequency (convergence metric)
- Pickle/unpickle graphs
- Extract k-hop subgraphs

### 3. Clustering Engine (`core/clustering_engine.py`)
**Algorithms**:
- Centrality: PageRank, betweenness, degree, closeness
- Convergent nodes: Filter by gene_frequency ≥ threshold
- Communities: Louvain algorithm (python-louvain)
- Graph statistics: density, diameter, clustering coefficient

**Output**: ClusteringResults with ranked convergent nodes and community structure

### 4. RAG System (`core/rag_system.py`)
**Context Strategy** (3-layer):
- Layer 1: Graph statistics + top convergent nodes (2K tokens)
- Layer 2: Question-relevant subgraph (3-8K tokens)
- Layer 3: Detailed node properties (1-2K tokens per node)

**Citation Mechanism**:
- Claude Haiku 4 with tool use (structured output)
- Tool: `cite_graph_evidence` returns node_ids, metric_name, metric_value
- Validation: Check citations against actual graph
- Visualization: Extract subgraph → render with PyVis

### 5. PyVis Visualization (`ui/network_viz.py`)
**Features**:
- Physics-based layout (forceAtlas2)
- Node sizing by gene_frequency or centrality
- Node coloring by biolink category
- Hover tooltips with full properties
- Interactive drag/zoom/pan
- Export to standalone HTML

**Configuration**:
- <30 nodes: forceAtlas2, 100 iterations
- 30-100 nodes: barnesHut, 150 iterations  
- \>100 nodes: Show warning, reduce node size
- \>200 nodes: Auto-sample or warn user

**Citation Display**:
- Cited nodes: Red/enlarged
- Context nodes: Teal/standard size
- Edge width by confidence score
- Embedded in Streamlit via `st.components.v1.html()`

---

## Test Case: Alzheimer's Disease (15 genes)

**Genes**: APOE, APP, PSEN1, PSEN2, MAPT, TREM2, CLU, CR1, BIN1, PICALM, CD33, MS4A6A, ABCA7, SORL1, BACE1  
**Disease**: MONDO:0004975

**Expected Results**:
- Convergent nodes: BACE1, amyloid-β, cholesterol pathway proteins
- Communities: 3-4 clusters (amyloid processing, lipid metabolism, neuroinflammation)
- Top targets: BACE1, APOE, TREM2

**Validation**: Results match known biology (80%+ accuracy)

---

## Data Persistence Strategy

### Critical Checkpoint Points

| Stage | Checkpoint | File Location | Purpose |
|-------|-----------|---------------|---------|
| Input | Gene list validated | `sessions/{id}/input_config.json` | Resume if interrupted |
| TRAPI | Individual responses | `cache/trapi_{gene}_{disease}.json` | Avoid re-querying |
| Clustering | All results | `sessions/{id}/clustering_results.json` | Preserve clustering results for previous queries |
| RAG | Conversation | `sessions/{id}/conversation.json` | Preserve chat history |
| Export | Final reports | `exports/{id}_report_{timestamp}.pdf` | Deliverable |

## Development Phases

### Phase 1: Foundation & POC ✅ COMPLETED
**Status**: All functionality working in `notebooks/multi_gene_pathfinder.ipynb`

**Accomplishments**:
- ✅ Gene normalization using TCT's `name_resolver.batch_lookup()`
- ✅ TRAPI integration with neighborhood discovery pattern (empty target list)
- ✅ Parallel API querying (15 Translator APIs, 6/15 success rate = normal)
- ✅ NetworkX graph construction from TRAPI responses
- ✅ Rich node attributes (labels, categories, is_query_gene flags)
- ✅ Interactive ipycytoscape visualization with 3-tier color coding
- ✅ Smart edge sampling guaranteeing all query genes appear
- ✅ JSON persistence of raw TRAPI results in `data/raw/`
- ✅ Robust error handling and graceful API degradation

**Test Results**:
- Successfully normalized 10/10 COVID-19 DE genes
- Retrieved 1,537 edges (812 unique after dedup) from knowledge graphs
- Built graph with 476 nodes, 723 edges, single connected component

**Current Implementation**: Single notebook architecture with complete end-to-end workflow

---

### Phase 2: Graph Processing & Clustering 🎯 IN PROGRESS
**Goal**: Convert notebook prototype to modular package with clustering capabilities

**Tasks**:
1. **Refactor to Package Structure**
   - [ ] Create `core/trapi_client.py` from notebook cells
   - [ ] Create `core/graph_builder.py` with NetworkX construction
   - [ ] Create `utils/persistence.py` for caching/serialization
   - [ ] Move visualization logic to `ui/network_viz.py`

2. **Implement Clustering**
   - [ ] Add Louvain community detection (`core/clustering_engine.py`)
   - [ ] Compute centrality metrics (PageRank, betweenness, degree)
   - [ ] Identify hub nodes within clusters
   - [ ] Add cluster statistics (size, density, modularity)

3. **Basic Streamlit UI**
   - [ ] Gene input panel with validation
   - [ ] Query status/progress tracking
   - [ ] Results overview dashboard
   - [ ] Cluster visualization with PyVis

**Expected Outcomes**:
- Modular codebase following planned architecture
- 3-4 detected communities for Alzheimer's test case
- Basic Streamlit app for running queries

---

### Phase 3: RAG + Advanced Visualization 📋 PLANNED
**Goal**: LLM-assisted exploration with Claude API integration

**Tasks**:
1. **RAG System**
   - [ ] Implement 3-layer context strategy
   - [ ] Claude Haiku 4 integration with tool use
   - [ ] Citation validation logic
   - [ ] Subgraph extraction for citations

2. **Enhanced Visualization**
   - [ ] PyVis renderer with physics-based layouts
   - [ ] Node sizing by convergence metrics
   - [ ] Hover tooltips with full node properties
   - [ ] Citation highlighting (cited nodes enlarged)

3. **Interactive Chat UI**
   - [ ] Streamlit chat interface
   - [ ] Click citation → expand PyVis subgraph
   - [ ] Export to standalone HTML

---

### Phase 4: UI Polish & Production 📋 PLANNED
**Goal**: Production-ready application with full test coverage

**Tasks**:
1. **Complete UI**
   - [ ] Convergent nodes table with sorting/filtering
   - [ ] Session management (save/load graphs)
   - [ ] Export to HTML, PNG, PDF

2. **Testing & Documentation**
   - [ ] Test suite with >80% coverage
   - [ ] User guide and examples
   - [ ] API documentation
   - [ ] Alzheimer's test case validation

3. **Deployment**
   - [ ] Streamlit Cloud deployment
   - [ ] ReadTheDocs documentation
   - [ ] PyPI package release

---

## Dependencies

```
streamlit==1.32.0
anthropic==0.21.3
networkx==3.2.1
python-louvain==0.16
pyvis==0.3.2
matplotlib==3.8.3
pydantic==2.6.3
requests==2.31.0
pandas==2.2.1
pytest==8.1.1
```

---

## RAG Citation Workflow

1. **User asks**: "Why is BACE1 a high-priority target?"
2. **Context retrieval**: Extract BACE1 node + 1-hop neighbors + metrics
3. **LLM generates**: Answer with structured citations via tool use
4. **System validates**: Check node IDs exist, metrics match
5. **User clicks citation**: PyVis graph expands showing cited subgraph
6. **User explores**: Hover nodes, drag to reposition, click to expand
7. **Export**: Download standalone HTML or screenshot

---

## PyVis Configuration

**Physics settings**:
```python
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "springLength": 100
    },
    "solver": "forceAtlas2Based",
    "stabilization": {"iterations": 150}
  }
}
```

**Node styling**:
- Size: `base_size + (gene_frequency * 20)`
- Color: Biolink category mapping (Gene=#FF6B6B, Protein=#4ECDC4, etc.)
- Border: 2px standard, 4px for cited nodes

**Tooltips**:
```
{node.name}
ID: {node.id}
Gene frequency: {gene_frequency}
Betweenness: {betweenness:.1f}
PageRank: {pagerank:.4f}
Categories: {categories}
```

---

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| 15-gene TRAPI batch | <5 min | With caching/parallelization |
| Clustering analysis | <30 sec | NetworkX in-memory |
| PyVis render (<100 nodes) | <3 sec | Physics stabilization |
| PyVis render (100-200 nodes) | <8 sec | May show warning |
| LLM response | <10 sec | Claude Haiku |
| Page load | <2 sec | Any view |

---

## Model Selection: Claude Haiku 4

**Why Haiku**:
- Task is information extraction, not complex reasoning
- Graph analysis already done by NetworkX
- Tool use for structured citations
- Cost: $0.02 per session (5 questions × 12K input + 1K output)

---

## Success Metrics

**Phase 1 (Completed)**:
- ✅ Process 10 genes in <5 minutes (achieved: 1,537 raw edges, ~2 min)
- ✅ Build NetworkX graph from TRAPI responses (476 nodes, 723 edges)
- ✅ Render interactive visualization (ipycytoscape, 50-edge sampling)
- ✅ Gene normalization with 100% success rate (10/10 COVID genes)
- ✅ Graceful API degradation (40% success rate = sufficient)

**Phase 2 (In Progress)**:
- [ ] Detect 3-4 communities in Alzheimer's test case
- [ ] Identify convergent nodes (BACE1, Aβ expected)
- [ ] Modular package structure following architecture plan
- [ ] Basic Streamlit UI for running queries

**Phase 3 (Planned)**:
- [ ] RAG system with Claude Haiku 4
- [ ] Citation accuracy >90% (metrics match graph)
- [ ] PyVis visualization with physics-based layouts
- [ ] Export to standalone HTML

**Phase 4 (Planned)**:
- [ ] Test suite with >80% coverage
- [ ] Non-expert can run analysis in <10 clicks
- [ ] Complete documentation (README, user guide, API docs)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| TRAPI downtime | Cache all responses, retry logic |
| PyVis performance (>200 nodes) | Sample nodes or show warning |
| Citation parsing errors | Use structured output (tool use) |
| Browser compatibility | Test Chrome/Firefox/Safari |
| Physics doesn't stabilize | Max iteration limit (150) |

---

## Future Enhancements (Post-MVP)

- Multi-disease comparison
- Augmenting the TRAPI response graphs with addtional data sources (WikiPathways, PFOCR, Human Proteing Atlas)

---

## Deliverables


1. Working Streamlit app
2. Alzheimer's test case validated
3. Documentation (README, user guide)
4. Example session exports (HTML graphs)
5. Test suite with >80% coverage

