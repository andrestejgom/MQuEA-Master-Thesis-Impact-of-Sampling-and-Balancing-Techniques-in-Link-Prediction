# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 16:04:51 2025

@author: Andrés
"""

import os
import pickle
import json
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path


# NETWORK CONFIGURATION
NETWORK_PARAMS_CONFIG = { 
    "erdos_renyi": {
        "param_names": ["n", "p"],
        "param_parsers": [int, float],
        "dirname_template": "synthetic_networks/erdos_renyi/er_n{n}_p{p:.3f}",
        "description_template": "Erdos-Renyi random graph with {n} nodes and edge probability {p}",
        "metadata_params": ["n", "p"],
        "generator_function": lambda n, p: nx.erdos_renyi_graph(n=n, p=p),
        "generation_params": ["n", "p"]
    },
    "watts_strogatz": {
        "param_names": ["n", "k", "p"],
        "param_parsers": [int, int, float],
        "dirname_template": "synthetic_networks/watts_strogatz/ws_n{n}_k{k}_p{p:.3f}",
        "description_template": "Watts-Strogatz small-world graph with {n} nodes, {k} neighbors, rewiring probability {p}",
        "metadata_params": ["n", "k", "p"],
        "generator_function": lambda n, k, p: nx.watts_strogatz_graph(n=n, k=k, p=p),
        "generation_params": ["n", "k", "p"]
    },
    "barabasi_albert": {
        "param_names": ["n", "m"],
        "param_parsers": [int, int],
        "dirname_template": "synthetic_networks/barabasi_albert/ba_n{n}_m{m}",
        "description_template": "Barabasi-Albert scale-free graph with {n} nodes, {m} edges per new node",
        "metadata_params": ["n", "m"],
        "generator_function": lambda n, m: nx.barabasi_albert_graph(n=n, m=m),
        "generation_params": ["n", "m"]
    },
    "stochastic_block_model": {
        "param_names": ["sizes", "p_intra", "p_inter"],
        "param_parsers": [
            lambda x: [int(i) for i in x.split('|')],
            float,
            float
        ],
        "dirname_template": "synthetic_networks/stochastic_block_model/sbm_s{sizes_str}_pi{p_intra:.4f}_pe{p_inter:.5f}",
        "description_template": "Stochastic block model with communities of sizes {sizes}, intra-community probability {p_intra}, inter-community probability {p_inter}",
        "metadata_params": ["sizes", "p_intra", "p_inter"],
        "special_formatting": {
            "sizes_str": lambda sizes, **kwargs: "_".join(map(str, sizes))
        },
        "generator_function": lambda sizes, p_intra, p_inter: nx.stochastic_block_model(sizes=sizes, p=create_prob_matrix(sizes, p_intra, p_inter)),
        "generation_params": ["sizes", "p_intra", "p_inter"]
    },
    "powerlaw_cluster": {
        "param_names": ["n", "m", "p"],
        "param_parsers": [int, int, float],
        "dirname_template": "synthetic_networks/powerlaw_cluster/pc_n{n}_m{m}_p{p:.3f}",
        "description_template": "Powerlaw cluster graph with {n} nodes, {m} edges per node, triangle probability {p}",
        "metadata_params": ["n", "m", "p"],
        "generator_function": lambda n, m, p: nx.powerlaw_cluster_graph(n=n, m=m, p=p),
        "generation_params": ["n", "m", "p"]
    }
}

# PREDEFINED NETWORK CONFIGURATIONS FOR SPECIFIC DENSITIES
NETWORK_CONFIGURATIONS = [
    # Erdos-Renyi networks (3 networks)
    ("erdos_renyi", 500, 0.015),     # 1.5% density
    ("erdos_renyi", 500, 0.010),     # 1.0% density
    ("erdos_renyi", 500, 0.005),     # 0.5% density
    
    # Watts-Strogatz networks (3 networks) - approx due to integer k constraint
    ("watts_strogatz", 500, 8, 0.1),  # 1.6% density (closest to 1.5%)
    ("watts_strogatz", 500, 6, 0.1),  # 1.2% density (closest to 1.0%)
    ("watts_strogatz", 500, 2, 0.1),  # 0.4% density (closest to 0.5%)
    
    # Barabasi-Albert networks (3 networks) - approx due to integer m constraint
    ("barabasi_albert", 500, 4),      # 1.6% density (closest to 1.5%)
    ("barabasi_albert", 500, 2),      # 0.8% density (closest to 1.0%)
    ("barabasi_albert", 500, 1),      # 0.4% density (closest to 0.5%)
    
    # Powerlaw Cluster networks (3 networks) - approx due to integer m constraint
    ("powerlaw_cluster", 500, 4, 0.2),  # 1.6% density (closest to 1.5%)
    ("powerlaw_cluster", 500, 2, 0.2),  # 0.8% density (closest to 1.0%)
    ("powerlaw_cluster", 500, 1, 0.2),  # 0.4% density (closest to 0.5%)
    
    # Stochastic Block Model networks (3 networks)
    ("stochastic_block_model", "166|167|167", 0.0375, 0.00375),  # 1.5% density
    ("stochastic_block_model", "166|167|167", 0.0250, 0.00250),  # 1.0% density
    ("stochastic_block_model", "166|167|167", 0.0125, 0.00125),  # 0.5% density
]

########################### AUXILIARY FUNCTIONS ###############################

def create_prob_matrix(sizes, p_intra, p_inter):
    """
    Create probability matrix for stochastic block model.
    
    Parameters:
    -----------
    sizes : list
        List of community sizes
    p_intra : float
        Intra-community connection probability (diagonal elements)
    p_inter : float
        Inter-community connection probability (off-diagonal elements)
        
    Returns:
    --------
    numpy.ndarray
        Probability matrix for SBM generation
    """
    k = len(sizes)
    P = np.full((k, k), p_inter)
    np.fill_diagonal(P, p_intra)
    return P

def parse_network_config(config_tuple):
    """
    Parse network configuration tuple into network_type and parameters.
    
    Parameters:
    -----------
    config_tuple : tuple
        Configuration tuple from NETWORK_CONFIGURATIONS
        
    Returns:
    --------
    tuple
        (network_type, parsed_parameters)
    """
    network_type = config_tuple[0]
    raw_params = config_tuple[1:]
    
    if network_type not in NETWORK_PARAMS_CONFIG:
        raise ValueError(f"Unknown network type: {network_type}")
    
    config = NETWORK_PARAMS_CONFIG[network_type]
    param_names = config["param_names"]
    param_parsers = config["param_parsers"]
    
    # Parse parameters using configured parsers
    parsed_params = []
    for i, (name, parser) in enumerate(zip(param_names, param_parsers)):
        parsed_params.append(parser(raw_params[i]))
    
    return network_type, tuple(parsed_params)

def create_directory_path(network_type, *args):
    """Generate directory path based on network configuration."""
    if network_type not in NETWORK_PARAMS_CONFIG:
        raise ValueError(f"Unknown network type: {network_type}")
    
    config = NETWORK_PARAMS_CONFIG[network_type]
    param_names = config["param_names"]
    dirname_template = config["dirname_template"]
    
    params_dict = dict(zip(param_names, args))
    
    # Apply special formatting if it exists
    if "special_formatting" in config:
        for special_key, formatter in config["special_formatting"].items():
            params_dict[special_key] = formatter(**params_dict)
    
    # Format template with parameters
    return dirname_template.format(**params_dict)

def create_dir(directory_path):
    """Create directory structure for network storage."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    print(f"    Created directory: {directory_path}")

def create_network(network_type, *args):
    """Generate network graph based on type and parameters."""
    if network_type not in NETWORK_PARAMS_CONFIG:
        raise ValueError(f"Unknown network type: {network_type}")
    
    config = NETWORK_PARAMS_CONFIG[network_type]
    param_names = config["param_names"]
    generator_function = config["generator_function"]
    generation_params = config["generation_params"]
    
    params_dict = dict(zip(param_names, args))
    
    # Generate the network
    G = generator_function(*args)
    
    # Add metadata to the generated graph
    G.graph['network_type'] = network_type
    G.graph['generation_params'] = {param: params_dict[param] for param in generation_params}
    G.graph['true_type'] = network_type
    
    return G

def save_graph(directory_path, G):
    """Save network graph in multiple formats for compatibility."""
    # Save as pickle (fastest and most complete)
    pickle_path = os.path.join(directory_path, "network.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)
    
    # Save edge list as text backup
    edgelist_path = os.path.join(directory_path, "edgelist.txt")
    nx.write_edgelist(G, edgelist_path, data=False)
    
    print(f"    Saved: network.pkl, edgelist.txt")

def save_data(directory_path, network_type, *args):
    """Save network metadata and parameters with 8 most important topology properties."""
    if network_type not in NETWORK_PARAMS_CONFIG:
        raise ValueError(f"Unknown network type: {network_type}")
    
    config = NETWORK_PARAMS_CONFIG[network_type]
    param_names = config["param_names"]
    description_template = config["description_template"]
    metadata_params = config["metadata_params"]
    
    params_dict = dict(zip(param_names, args))
    metadata_params_dict = {key: params_dict[key] for key in metadata_params}
    
    # Load the graph to get actual properties
    pickle_path = os.path.join(directory_path, "network.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            G = pickle.load(f)
        
        # 1. Basic network size and connectivity
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        # 2. Average degree (fundamental connectivity measure)
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        avg_degree = sum(degree_values) / len(degree_values) if degree_values else 0
        
        # 3. Clustering coefficient (local structure)
        avg_clustering = nx.average_clustering(G)
        
        # 4. Connectivity (global structure)
        is_connected = nx.is_connected(G)
        num_connected_components = nx.number_connected_components(G)
        
        # 5. Path length (efficiency measure) - only for connected graphs
        if is_connected:
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            if len(largest_cc) > 1:
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
            else:
                avg_path_length = None
        
        # 6. Degree assortativity (mixing pattern)
        try:
            degree_assortativity = nx.degree_assortativity_coefficient(G)
        except:
            degree_assortativity = None
        
        # 7. Diameter (network span)
        if is_connected:
            diameter = nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            if len(largest_cc) > 1:
                subgraph = G.subgraph(largest_cc)
                diameter = nx.diameter(subgraph)
            else:
                diameter = None
        
        # 8. Transitivity (global clustering)
        transitivity = nx.transitivity(G)
        
        # Compile the 8 most important network properties
        network_properties = {
            "nodes": num_nodes,
            "edges": num_edges,  
            "density": density,
            "average_degree": avg_degree,
            "average_clustering": avg_clustering,
            "is_connected": is_connected,
            "average_path_length": avg_path_length,
            "degree_assortativity": degree_assortativity,
            "diameter": diameter,
            "transitivity": transitivity
        }
        
        # Remove None values for cleaner JSON
        network_properties = {k: v for k, v in network_properties.items() if v is not None}
        
    else:
        network_properties = {}
    
    metadata = {
        "network_type": network_type,
        "parameters": metadata_params_dict,
        "description": description_template.format(**params_dict),
        "network_properties": network_properties,
        "timestamp": str(pd.Timestamp.now()),
        "directory_path": directory_path,
        "file_format": "Pickle, EdgeList"
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(directory_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    Saved: metadata.json with topology properties")

########################### MAIN FUNCTION #####################################

def gen_networks():
    """
    Generate 15 predefined synthetic networks for link prediction research.
    
    Creates networks with specific density targets across 5 different topologies:
    - 3 Erdős-Rényi networks
    - 3 Watts-Strogatz networks
    - 3 Barabási-Albert networks
    - 3 Powerlaw Cluster networks
    - 3 Stochastic Block Model networks
    """
    
    networks_generated = 0
    networks_skipped = 0
    
    print("Starting network generation process...")
    print("Directory structure: synthetic_networks/topology/instance_name/")
    print()
    
    # Process each predefined network configuration
    for config_idx, config_tuple in enumerate(NETWORK_CONFIGURATIONS, 1):
        try:
            # Parse network type and parameters
            network_type, args = parse_network_config(config_tuple)
            
            # Generate directory path for this network
            directory_path = create_directory_path(network_type, *args)
            
            # Generate network and files only if directory doesn't exist
            if not os.path.exists(directory_path):
                print(f"[{config_idx}/{len(NETWORK_CONFIGURATIONS)}] Generating {network_type} network...")
                print(f"  Parameters: {args}")
                
                # Create directory structure
                create_dir(directory_path)
                
                # Generate the actual network
                G = create_network(network_type, *args)
                
                # Save network graph
                save_graph(directory_path, G)
                
                # Save network metadata and parameters
                save_data(directory_path, network_type, *args)
                
                networks_generated += 1
                print(f"  ✓ Generated network in: {directory_path}")
                
            else:
                networks_skipped += 1
                print(f"[{config_idx}/{len(NETWORK_CONFIGURATIONS)}] Network already exists, skipping: {directory_path}")
                
        except Exception as e:
            print(f"  ✗ Error generating network {config_idx}: {str(e)}")
            continue
    
    print(f"\n=== GENERATION COMPLETED ===")
    print(f"Generated: {networks_generated} new networks")
    print(f"Skipped: {networks_skipped} existing networks")
    print(f"Total networks: {networks_generated + networks_skipped}")
    print(f"Networks saved in 'synthetic_networks/' directory structure.")

if __name__ == "__main__":
    print("=== Synthetic Network Generator for Link Prediction Research ===")
    print("This script generates 15 networks with specific density targets.")
    print()
    
    # Run the network generation
    gen_networks()