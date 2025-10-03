"""BioGraph Explorer - Streamlit Application

Main Streamlit app for multi-gene TRAPI query integration with NetworkX clustering.
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

from biograph_explorer.core import TRAPIClient, GraphBuilder, ClusteringEngine
from biograph_explorer.utils import validate_gene_list, validate_disease_curie, ValidationError

# Page config
st.set_page_config(
    page_title="BioGraph Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: bold;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'query_genes' not in st.session_state:
    st.session_state.query_genes = []
if 'response' not in st.session_state:
    st.session_state.response = None
if 'selected_node' not in st.session_state:
    st.session_state.selected_node = None

# Title
st.title("🧬 BioGraph Explorer")
st.markdown("Multi-gene TRAPI query integration with NetworkX clustering and visualization")

# Sidebar: Input Configuration
st.sidebar.header("📥 Input Configuration")

# Input method selection
input_method = st.sidebar.radio(
    "Input Method",
    ["Example Dataset", "Upload CSV", "Manual Entry"],
    index=0
)

genes = []
disease_curie = "MONDO:0004975"  # Default: Alzheimer's

if input_method == "Example Dataset":
    dataset_choice = st.sidebar.selectbox(
        "Select Example",
        ["Alzheimer's Disease (15 genes)", "COVID-19 (10 genes)"]
    )
    
    if dataset_choice == "Alzheimer's Disease (15 genes)":
        csv_path = "data/test_genes/alzheimers_genes.csv"
        disease_curie = "MONDO:0004975"
    else:
        csv_path = "data/test_genes/covid19_genes.csv"
        disease_curie = "MONDO:0100096"
    
    try:
        df = pd.read_csv(csv_path)
        genes = df['gene_symbol'].tolist()
        st.sidebar.success(f"✓ Loaded {len(genes)} genes from example dataset")
        with st.sidebar.expander("View genes"):
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"Error loading example: {e}")

elif input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Gene List CSV",
        type=["csv"],
        help="CSV file with 'gene_symbol' column"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'gene_symbol' in df.columns:
                genes = df['gene_symbol'].tolist()
                st.sidebar.success(f"✓ Loaded {len(genes)} genes from CSV")
                with st.sidebar.expander("View genes"):
                    st.dataframe(df, use_container_width=True)
            else:
                st.sidebar.error("CSV must have 'gene_symbol' column")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

else:  # Manual Entry
    gene_text = st.sidebar.text_area(
        "Gene Symbols",
        height=150,
        help="Enter gene symbols, one per line or comma-separated",
        placeholder="APOE\nAPP\nPSEN1\n..."
    )
    
    if gene_text:
        # Parse input
        genes = [g.strip().upper() for g in gene_text.replace(',', '\n').split('\n') if g.strip()]
        st.sidebar.info(f"📝 {len(genes)} genes entered")

# Disease CURIE input
disease_curie = st.sidebar.text_input(
    "Disease CURIE",
    value=disease_curie,
    help="Optional disease CURIE (e.g., MONDO:0004975 for Alzheimer's)"
)

# Intermediate entity type selector
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Query Configuration")
st.sidebar.markdown("**Find paths:** Gene → **[Intermediate]** → Disease")

intermediate_types = st.sidebar.multiselect(
    "Intermediate Entity Types",
    options=[
        "Any (all connections)",
        "Protein",
        "ChemicalEntity (Drugs/Metabolites)",
        "PhenotypicFeature",
        "Pathway",
        "BiologicalProcess",
        "Gene",
        "AnatomicalEntity",
    ],
    default=["Any (all connections)"],
    help="Select intermediate entity types to filter connections. Choose 'Any' for broad exploration."
)

st.sidebar.markdown("---")

# Query button
run_query = st.sidebar.button(
    "🚀 Run Query",
    type="primary",
    use_container_width=True,
    disabled=len(genes) == 0
)

if len(genes) > 0:
    st.sidebar.markdown(f"**Ready to query:** {len(genes)} genes")

# Main area
if not run_query and not st.session_state.graph:
    # Welcome screen
    st.info("👈 Configure your query in the sidebar and click **Run Query** to start")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 What it does")
        st.markdown("""
        - **Normalize genes** using TCT name resolver
        - **Query TRAPI** APIs for knowledge graph edges
        - **Build graph** with NetworkX
        - **Detect communities** using Louvain algorithm
        - **Visualize** with interactive PyVis graphs
        """)
    
    with col2:
        st.subheader("📊 Example Datasets")
        st.markdown("""
        **Alzheimer's Disease** (15 genes)
        - APOE, APP, PSEN1, PSEN2, MAPT, TREM2, CLU, CR1, BIN1, PICALM, CD33, MS4A6A, ABCA7, SORL1, BACE1
        
        **COVID-19** (10 genes)
        - CD6, IFITM3, IFITM2, STAT5A, KLRG1, DPP4, IL32, PIK3AP1, FYN, IL4R
        """)

# Execute query
if run_query:
    try:
        # Validate input
        validated_genes = validate_gene_list(genes, min_genes=1, max_genes=50)
        
        if disease_curie:
            disease_curie = validate_disease_curie(disease_curie)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize clients
        status_text.text("Initializing TRAPI client...")
        progress_bar.progress(10)
        
        client = TRAPIClient(cache_dir=Path("data/cache"), max_workers=5)
        builder = GraphBuilder()
        engine = ClusteringEngine()

        # Convert intermediate type selections to biolink categories
        intermediate_categories = None
        if intermediate_types and "Any (all connections)" not in intermediate_types:
            # Map UI labels to biolink categories
            type_mapping = {
                "Protein": "biolink:Protein",
                "ChemicalEntity (Drugs/Metabolites)": "biolink:ChemicalEntity",
                "PhenotypicFeature": "biolink:PhenotypicFeature",
                "Pathway": "biolink:Pathway",
                "BiologicalProcess": "biolink:BiologicalProcess",
                "Gene": "biolink:Gene",
                "AnatomicalEntity": "biolink:AnatomicalEntity",
            }
            intermediate_categories = [type_mapping[t] for t in intermediate_types if t in type_mapping]

        # Step 2: Query TRAPI
        status_text.text(f"Querying Translator APIs for {len(validated_genes)} genes...")
        progress_bar.progress(20)

        def progress_callback(msg):
            status_text.text(msg)

        response = client.query_gene_neighborhood(
            validated_genes,
            disease_curie=disease_curie,
            intermediate_categories=intermediate_categories,
            progress_callback=progress_callback
        )
        
        progress_bar.progress(60)
        st.success(f"✓ Found {len(response.edges)} edges from {response.apis_succeeded}/{response.apis_queried} APIs")
        
        # Step 3: Build graph
        status_text.text("Building knowledge graph...")
        progress_bar.progress(70)

        # Get curie_to_symbol mapping from response metadata
        curie_to_symbol = response.metadata.get("curie_to_symbol", {})

        kg = builder.build_from_trapi_edges(response.edges, response.input_genes, curie_to_symbol)
        
        # Calculate gene frequency
        gene_freq = builder.calculate_gene_frequency(kg.graph, response.input_genes)
        for node, freq in gene_freq.items():
            kg.graph.nodes[node]['gene_frequency'] = freq
        
        progress_bar.progress(80)
        
        # Step 4: Cluster
        status_text.text("Detecting communities...")
        results = engine.analyze_graph(kg.graph, response.input_genes)
        
        progress_bar.progress(90)
        
        # Save to session state
        st.session_state.graph = kg.graph
        st.session_state.clustering_results = results
        st.session_state.query_genes = response.input_genes
        st.session_state.response = response
        
        progress_bar.progress(100)
        status_text.text("✓ Analysis complete!")
        
    except ValidationError as e:
        st.error(f"Validation error: {e}")
    except Exception as e:
        st.error(f"Error during query: {e}")
        import traceback
        st.code(traceback.format_exc())

# Display results
if st.session_state.graph:
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🕸️ Network", "🔍 Communities"])
    
    with tab1:
        st.header("Analysis Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", st.session_state.graph.number_of_nodes())
        
        with col2:
            st.metric("Edges", st.session_state.graph.number_of_edges())
        
        with col3:
            st.metric("Communities", st.session_state.clustering_results.num_communities)
        
        with col4:
            import networkx as nx
            st.metric("Density", f"{nx.density(st.session_state.graph):.4f}")
        
        # Graph statistics
        st.subheader("Graph Statistics")
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.json(st.session_state.clustering_results.graph_stats)
        
        with stats_col2:
            st.metric("Modularity", f"{st.session_state.clustering_results.modularity:.3f}")
            st.caption("Higher modularity indicates better community structure")
        
        # Node categories
        st.subheader("Node Categories")
        category_df = pd.DataFrame([
            {"Category": cat, "Count": count}
            for cat, count in sorted(
                st.session_state.response.metadata.get("node_categories", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )
        ])
        if not category_df.empty:
            st.dataframe(category_df, use_container_width=True)
    
    with tab2:
        st.header("Knowledge Graph Visualization")

        # Visualization controls
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])

        with col1:
            sizing_metric = st.selectbox(
                "Node Size By",
                ["gene_frequency", "pagerank", "betweenness", "degree"],
                index=0,
                help="Metric used to determine node size"
            )

        with col2:
            max_nodes = st.slider(
                "Max Nodes",
                min_value=50,
                max_value=200,
                value=100,
                step=10,
                help="Maximum number of nodes to display (graph will be sampled if larger)"
            )

        with col3:
            freeze_layout = st.checkbox(
                "Freeze Layout",
                value=True,
                help="Disable physics after stabilization to prevent excessive movement"
            )

        with col4:
            cluster_view = st.checkbox(
                "Cluster View",
                value=False,
                help="Group nodes by community (shows meta-graph)"
            )

        with col5:
            export_html = st.button("📥 Export", help="Save visualization as standalone HTML file")

        if export_html:
            from biograph_explorer.ui.network_viz import export_visualization_html
            output_path = Path("data/exports") / f"network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            try:
                result_path = export_visualization_html(
                    st.session_state.graph,
                    st.session_state.query_genes,
                    output_path
                )
                st.success(f"✓ Exported to {result_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")

        # Two-column layout: visualization + node details
        viz_col, details_col = st.columns([7, 3])

        with viz_col:
            # Node inspection input
            inspect_node_id = st.text_input(
                "🔍 Inspect Node (paste CURIE from tooltip)",
                value=st.session_state.selected_node or "",
                help="Paste a node ID (e.g., NCBIGene:1803) to view details",
                key="node_inspect_input"
            )

            if inspect_node_id and inspect_node_id != st.session_state.selected_node:
                st.session_state.selected_node = inspect_node_id

            # Render PyVis visualization
            from biograph_explorer.ui.network_viz import render_network_visualization, create_clustered_graph

            # Determine which graph to visualize
            if cluster_view and st.session_state.clustering_results:
                # Create cluster meta-graph
                display_graph = create_clustered_graph(
                    st.session_state.graph,
                    st.session_state.clustering_results,
                    st.session_state.query_genes
                )
                st.info(f"ℹ️ Showing {display_graph.number_of_nodes()} clusters (from {st.session_state.graph.number_of_nodes()} nodes)")
            else:
                display_graph = st.session_state.graph

            with st.spinner("Rendering interactive network..."):
                html = render_network_visualization(
                    display_graph,
                    st.session_state.query_genes,
                    sizing_metric=sizing_metric,
                    max_nodes=max_nodes,
                    freeze_layout=freeze_layout,
                )

            if html:
                components.html(html, height=700, scrolling=True)

                st.caption("""
                **💡 How to explore:**
                - **Drag** nodes to rearrange • **Scroll** to zoom
                - **Hover** for details • **Copy node ID** from tooltip and paste above to inspect
                """)
            else:
                st.error("Failed to render visualization")

        with details_col:
            st.subheader("📌 Node Details")

            if st.session_state.selected_node:
                from biograph_explorer.ui.network_viz import get_node_details

                node_details = get_node_details(st.session_state.selected_node, st.session_state.graph)

                if "error" in node_details:
                    st.error(node_details["error"])
                else:
                    # Display node information
                    if node_details["original_symbol"]:
                        st.markdown(f"**{node_details['original_symbol']}**")
                        st.caption(node_details["label"])
                    else:
                        st.markdown(f"**{node_details['label']}**")

                    st.caption(f"`{node_details['node_id']}`")

                    if node_details["is_query_gene"]:
                        st.markdown("🔴 **Query Gene**")

                    st.markdown(f"**Category:** {node_details['category']}")

                    # Metrics
                    st.markdown("**📊 Metrics**")
                    metrics = node_details["metrics"]
                    st.markdown(f"- Gene Frequency: **{metrics['gene_frequency']}**")
                    st.markdown(f"- PageRank: **{metrics['pagerank']:.4f}**")
                    st.markdown(f"- Betweenness: **{metrics['betweenness']:.1f}**")
                    st.markdown(f"- Degree: **{metrics['degree']}** ({metrics['in_degree']} in, {metrics['out_degree']} out)")

                    # Edges
                    st.markdown(f"**🔗 Edges ({node_details['total_edges']})**")
                    for predicate, edges in sorted(node_details["edges_by_predicate"].items()):
                        with st.expander(f"{predicate.replace('_', ' ').title()} ({len(edges)})", expanded=len(node_details["edges_by_predicate"]) == 1):
                            for edge in edges[:5]:  # Limit to 5 per predicate
                                direction_icon = "←" if edge["direction"] == "incoming" else "→"
                                target_key = "source_label" if edge["direction"] == "incoming" else "target_label"
                                st.markdown(f"{direction_icon} {edge[target_key][:40]}")
                            if len(edges) > 5:
                                st.caption(f"... and {len(edges) - 5} more")

                    # Sources
                    if node_details["knowledge_sources"]:
                        st.markdown(f"**📚 Knowledge Sources ({len(node_details['knowledge_sources'])})**")
                        for source in node_details["knowledge_sources"][:10]:
                            if source:  # Some sources may be empty
                                st.markdown(f"- {source}")
                        if len(node_details["knowledge_sources"]) > 10:
                            st.caption(f"... and {len(node_details['knowledge_sources']) - 10} more")
                    else:
                        st.caption("_No source information available_")
            else:
                st.info("👆 Paste a node ID above to view detailed information")
    
    with tab3:
        st.header("Community Detection Results")
        
        st.write(f"Detected **{st.session_state.clustering_results.num_communities} communities** using Louvain algorithm")
        st.write(f"Modularity: **{st.session_state.clustering_results.modularity:.3f}**")
        
        # Display each community
        for comm in st.session_state.clustering_results.communities[:10]:  # Limit to top 10
            with st.expander(f"Community {comm.community_id} ({comm.size} nodes, density={comm.density:.3f})"):
                if comm.top_nodes:
                    st.write("**Top nodes by PageRank:**")
                    for node_info in comm.top_nodes:
                        st.write(f"- {node_info['label']} (PageRank: {node_info['pagerank']:.4f})")
                else:
                    st.write(f"Nodes: {', '.join(comm.nodes[:10])}")
                    if len(comm.nodes) > 10:
                        st.caption(f"... and {len(comm.nodes) - 10} more")

# Footer
st.divider()
st.markdown("**BioGraph Explorer** | Built with Streamlit, NetworkX, TCT, and PyVis")
