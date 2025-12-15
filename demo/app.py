"""Interactive Streamlit demo for citation network analysis."""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.data.citation_datasets import CitationDataset
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.utils.device import get_device


def load_model_and_data(model_name: str, dataset_name: str) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    """Load trained model and data."""
    # Load dataset
    dataset = CitationDataset(
        dataset_name=dataset_name,
        root="data",
        normalize_features=True,
        add_self_loops=True,
    )
    
    data = dataset.get_data()
    data_info = dataset.get_info()
    
    # Initialize model
    if model_name == "GCN":
        model = GCN(
            input_dim=data_info["num_features"],
            hidden_dim=64,
            output_dim=data_info["num_classes"],
            num_layers=2,
            dropout=0.5,
            activation="relu",
            batch_norm=True,
        )
    elif model_name == "GraphSAGE":
        model = GraphSAGE(
            input_dim=data_info["num_features"],
            hidden_dim=64,
            output_dim=data_info["num_classes"],
            num_layers=2,
            dropout=0.5,
            activation="relu",
            batch_norm=True,
            aggregator="mean",
        )
    elif model_name == "GAT":
        model = GAT(
            input_dim=data_info["num_features"],
            hidden_dim=64,
            output_dim=data_info["num_classes"],
            num_layers=2,
            num_heads=8,
            dropout=0.5,
            activation="elu",
            batch_norm=True,
            attention_dropout=0.0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load trained weights if available
    checkpoint_path = f"assets/models/{model_name.lower()}/best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    else:
        st.warning(f"No trained model found at {checkpoint_path}. Using random weights.")
    
    return model, data, data_info


def create_network_plot(data: torch.Tensor, node_ids: List[int] = None, max_nodes: int = 100) -> go.Figure:
    """Create interactive network visualization."""
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # Sample nodes if too many
    if len(G.nodes()) > max_nodes:
        if node_ids is None:
            node_ids = list(G.nodes())[:max_nodes]
        G = G.subgraph(node_ids)
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Node {node}<br>Class: {data.y[node].item()}")
        node_colors.append(data.y[node].item())
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=10,
            color=node_colors,
            colorscale="Viridis",
            line=dict(width=2, color="white"),
        ),
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Citation Network Visualization",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Interactive citation network",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(color="#888", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor="white",
                    ))
    
    return fig


def create_embeddings_plot(model: torch.nn.Module, data: torch.Tensor, method: str = "tsne") -> go.Figure:
    """Create embeddings visualization."""
    with torch.no_grad():
        embeddings = model.get_embeddings(data.x, data.edge_index)
    
    embeddings_np = embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()
    
    # Dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
    elif method.lower() == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings_np)
    
    # Create scatter plot
    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=labels,
        title=f"Node Embeddings ({method.upper()})",
        labels={"x": f"{method.upper()} 1", "y": f"{method.upper()} 2"},
        color_continuous_scale="Viridis",
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(plot_bgcolor="white")
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Citation Network Analysis",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 Citation Network Analysis with Graph Neural Networks")
    st.markdown("Interactive exploration of citation networks using GCN, GraphSAGE, and GAT models.")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["cora", "citeseer", "pubmed", "synthetic"],
        index=0,
    )
    
    model_name = st.sidebar.selectbox(
        "Model",
        ["GCN", "GraphSAGE", "GAT"],
        index=0,
    )
    
    # Load model and data
    try:
        model, data, data_info = load_model_and_data(model_name, dataset_name)
        
        st.sidebar.success(f"Loaded {model_name} on {dataset_name}")
        st.sidebar.metric("Nodes", data_info["num_nodes"])
        st.sidebar.metric("Edges", data_info["num_edges"])
        st.sidebar.metric("Features", data_info["num_features"])
        st.sidebar.metric("Classes", data_info["num_classes"])
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🕸️ Network", "🧠 Embeddings", "🔍 Node Analysis"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            stats_df = pd.DataFrame([
                {"Metric": "Nodes", "Value": data_info["num_nodes"]},
                {"Metric": "Edges", "Value": data_info["num_edges"]},
                {"Metric": "Features", "Value": data_info["num_features"]},
                {"Metric": "Classes", "Value": data_info["num_classes"]},
                {"Metric": "Train Nodes", "Value": data_info["train_nodes"]},
                {"Metric": "Val Nodes", "Value": data_info["val_nodes"]},
                {"Metric": "Test Nodes", "Value": data_info["test_nodes"]},
            ])
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.subheader("Class Distribution")
            class_counts = torch.bincount(data.y).numpy()
            class_df = pd.DataFrame({
                "Class": range(len(class_counts)),
                "Count": class_counts,
            })
            
            fig = px.bar(class_df, x="Class", y="Count", title="Node Class Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Network Visualization")
        
        max_nodes = st.slider("Max nodes to display", 10, 500, 100)
        
        # Create network plot
        fig = create_network_plot(data, max_nodes=max_nodes)
        st.plotly_chart(fig, use_container_width=True)
        
        # Network statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Density", f"{data.num_edges / (data.num_nodes * (data.num_nodes - 1) / 2):.4f}")
        
        with col2:
            degrees = torch.bincount(data.edge_index[0])
            st.metric("Avg Degree", f"{degrees.float().mean():.2f}")
        
        with col3:
            st.metric("Max Degree", f"{degrees.max().item()}")
    
    with tab3:
        st.header("Node Embeddings")
        
        method = st.selectbox("Dimensionality Reduction", ["tsne", "pca"])
        
        # Create embeddings plot
        fig = create_embeddings_plot(model, data, method)
        st.plotly_chart(fig, use_container_width=True)
        
        # Embedding statistics
        with torch.no_grad():
            embeddings = model.get_embeddings(data.x, data.edge_index)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Embedding Dim", embeddings.shape[1])
        
        with col2:
            st.metric("Mean Norm", f"{embeddings.norm(dim=1).mean():.4f}")
        
        with col3:
            st.metric("Std Norm", f"{embeddings.norm(dim=1).std():.4f}")
    
    with tab4:
        st.header("Node Analysis")
        
        # Node selection
        node_id = st.number_input("Node ID", 0, data.num_nodes - 1, 0)
        
        if st.button("Analyze Node"):
            with torch.no_grad():
                # Get predictions
                out = model(data.x, data.edge_index)
                pred = out[node_id].argmax().item()
                confidence = torch.softmax(out[node_id], dim=0).max().item()
                
                # Get neighbors
                edge_mask = (data.edge_index[0] == node_id) | (data.edge_index[1] == node_id)
                neighbor_edges = data.edge_index[:, edge_mask]
                neighbors = neighbor_edges[1 - (neighbor_edges[0] == node_id).long()]
                
                # Get embeddings
                embeddings = model.get_embeddings(data.x, data.edge_index)
                node_embedding = embeddings[node_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Node Information")
                st.write(f"**Node ID:** {node_id}")
                st.write(f"**True Class:** {data.y[node_id].item()}")
                st.write(f"**Predicted Class:** {pred}")
                st.write(f"**Confidence:** {confidence:.4f}")
                st.write(f"**Degree:** {len(neighbors)}")
                
                # Feature statistics
                features = data.x[node_id]
                st.write(f"**Feature Mean:** {features.mean():.4f}")
                st.write(f"**Feature Std:** {features.std():.4f}")
            
            with col2:
                st.subheader("Neighbors")
                if len(neighbors) > 0:
                    neighbor_df = pd.DataFrame({
                        "Neighbor": neighbors.numpy(),
                        "Class": data.y[neighbors].numpy(),
                    })
                    st.dataframe(neighbor_df, use_container_width=True)
                else:
                    st.write("No neighbors found")
            
            # Prediction probabilities
            st.subheader("Prediction Probabilities")
            probs = torch.softmax(out[node_id], dim=0)
            prob_df = pd.DataFrame({
                "Class": range(len(probs)),
                "Probability": probs.numpy(),
            })
            
            fig = px.bar(prob_df, x="Class", y="Probability", title="Class Probabilities")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
