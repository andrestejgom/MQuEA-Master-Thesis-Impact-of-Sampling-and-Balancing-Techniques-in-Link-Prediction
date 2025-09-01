# -*- coding: utf-8 -*-
"""
Dutch Corporate Directors Network Generator
Generates 3 person-to-person networks from Dutch corporate directors meetings data (1976, 1996, 2001)
"""

import os
import pickle
import json
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

DIRECTORS_MEETINGS_CONFIG = {
    "directors_meetings_1976": {
        "sheet_name": "Meetings Between Directors 1976",
        "dirname_template": "directors_meetings/directors_1976",
        "description_template": "Dutch Corporate Directors Network 1976",
        "year": 1976
    },
    "directors_meetings_1996": {
        "sheet_name": "Meetings Between Directors 1996", 
        "dirname_template": "directors_meetings/directors_1996",
        "description_template": "Dutch Corporate Directors Network 1996",
        "year": 1996
    },
    "directors_meetings_2001": {
        "sheet_name": "Meetings Between Directors 2001",
        "dirname_template": "directors_meetings/directors_2001", 
        "description_template": "Dutch Corporate Directors Network 2001",
        "year": 2001
    }
}

EXCEL_FILE = "Dutch Corporate Network Board Interlocks 1976 1996 2001 V1.xls"

def identify_person_columns(df):
    """Identify columns containing Person IDs."""
    person_cols = []
    for i, col in enumerate(df.columns):
        if "PERSON ID" in str(col).upper():
            person_cols.append((i, col))
    
    if len(person_cols) >= 2:
        return person_cols[0][0], person_cols[1][0]
    else:
        numeric_cols = []
        for i, col in enumerate(df.columns):
            try:
                numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if numeric_count > len(df) * 0.5:
                    numeric_cols.append((i, col))
            except:
                continue
        
        if len(numeric_cols) >= 2:
            return numeric_cols[0][0], numeric_cols[1][0]
        else:
            return 0, 1

def load_directors_meetings_data(excel_file, sheet_name):
    """Load directors meetings data from Excel sheet."""
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        person_a_idx, person_b_idx = identify_person_columns(df)
        
        person_a_col = df.iloc[:, person_a_idx]  
        person_b_col = df.iloc[:, person_b_idx]  
        
        edges_raw = []
        for i in range(len(person_a_col)):
            person_a = person_a_col.iloc[i]
            person_b = person_b_col.iloc[i]
            
            if pd.isna(person_a) or pd.isna(person_b):
                continue
                
            try:
                person_a = int(float(person_a))
                person_b = int(float(person_b))
            except:
                person_a = str(person_a).strip()
                person_b = str(person_b).strip()
            
            if person_a != person_b:
                edges_raw.append((person_a, person_b))
        
        return edges_raw
        
    except Exception as e:
        print(f"ERROR loading '{sheet_name}': {str(e)}")
        return []

def remap_nodes_to_consecutive(edges_raw):
    """Remap node IDs to consecutive integers starting from 0."""
    unique_nodes = set()
    for edge in edges_raw:
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])
    
    sorted_nodes = sorted(unique_nodes)
    node_mapping = {original_id: new_id for new_id, original_id in enumerate(sorted_nodes)}
    reverse_mapping = {new_id: original_id for original_id, new_id in node_mapping.items()}
    remapped_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges_raw]
    
    return remapped_edges, node_mapping, reverse_mapping

def create_directors_network(edges_raw, network_type):
    """Generate network with consecutive node IDs."""
    if not edges_raw:
        return nx.Graph(), {}, {}
    
    remapped_edges, node_mapping, reverse_mapping = remap_nodes_to_consecutive(edges_raw)
    G = nx.Graph()
    G.add_edges_from(remapped_edges)
    
    config = DIRECTORS_MEETINGS_CONFIG[network_type]
    G.graph['network_type'] = network_type
    G.graph['year'] = config['year']
    G.graph['true_type'] = 'directors_meetings'
    G.graph['description'] = config['description_template']
    G.graph['node_mapping'] = node_mapping
    G.graph['reverse_mapping'] = reverse_mapping
    
    return G, node_mapping, reverse_mapping

def save_graph(directory_path, G):
    """Save network in multiple formats."""
    pickle_path = os.path.join(directory_path, "network.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)
    
    edgelist_path = os.path.join(directory_path, "edgelist.txt")
    nx.write_edgelist(G, edgelist_path, data=False)
    
    dl_path = os.path.join(directory_path, "network.dl")
    with open(dl_path, "w") as f:
        f.write("DL N={}\n".format(G.number_of_nodes()))
        f.write("FORMAT = EDGELIST1\n")
        f.write("DATA:\n")
        for edge in G.edges():
            f.write("{} {}\n".format(edge[0], edge[1]))

def save_node_mappings(directory_path, node_mapping, reverse_mapping):
    """Save node ID mappings."""
    mapping_path = os.path.join(directory_path, "node_mapping.json")
    with open(mapping_path, "w") as f:
        mapping_for_json = {str(orig): new for orig, new in node_mapping.items()}
        reverse_for_json = {str(new): orig for new, orig in reverse_mapping.items()}
        
        json.dump({
            "original_to_consecutive": mapping_for_json,
            "consecutive_to_original": reverse_for_json,
            "total_nodes": len(node_mapping),
            "original_id_range": [min(node_mapping.keys()), max(node_mapping.keys())],
            "consecutive_id_range": [0, len(node_mapping)-1]
        }, f, indent=2)
    
    csv_path = os.path.join(directory_path, "node_mapping.csv")
    mapping_df = pd.DataFrame([
        {"original_id": orig, "consecutive_id": new} 
        for orig, new in sorted(node_mapping.items())
    ])
    mapping_df.to_csv(csv_path, index=False)

def calculate_network_properties(G):
    """Calculate network properties."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G) if num_nodes > 1 else 0
    
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    avg_degree = sum(degree_values) / len(degree_values) if degree_values else 0
    
    is_connected = nx.is_connected(G)
    num_connected_components = nx.number_connected_components(G)
    avg_clustering = nx.average_clustering(G) if num_nodes > 1 else 0
    
    if is_connected and num_nodes > 1:
        try:
            avg_path_length = nx.average_shortest_path_length(G)
        except:
            avg_path_length = None
    else:
        try:
            if num_nodes > 1:
                largest_cc = max(nx.connected_components(G), key=len)
                if len(largest_cc) > 1:
                    subgraph = G.subgraph(largest_cc)
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    avg_path_length = None
            else:
                avg_path_length = None
        except:
            avg_path_length = None
    
    return {
        "nodes": num_nodes,
        "edges": num_edges,  
        "density": round(density, 6),
        "average_degree": round(avg_degree, 2),
        "is_connected": is_connected,
        "number_connected_components": num_connected_components,
        "average_clustering": round(avg_clustering, 4),
        "average_path_length": round(avg_path_length, 4) if avg_path_length else None,
        "is_simple": True,
        "is_directed": False,
        "has_weights": False,
        "has_consecutive_node_ids": True,
        "node_id_range": f"0 to {num_nodes-1}"
    }

def save_metadata(directory_path, network_type, G, node_mapping, reverse_mapping):
    """Save network metadata."""
    config = DIRECTORS_MEETINGS_CONFIG[network_type]
    network_properties = calculate_network_properties(G)
    
    metadata = {
        "network_type": network_type,
        "year": config['year'],
        "data_source": "Dutch Corporate Directors Meetings",
        "description": config['description_template'],
        "network_properties": network_properties,
        "node_mapping_info": {
            "has_remapping": True,
            "total_nodes": len(node_mapping),
            "original_id_range": [min(node_mapping.keys()), max(node_mapping.keys())],
            "consecutive_id_range": [0, len(node_mapping)-1],
            "mapping_files": ["node_mapping.json", "node_mapping.csv"]
        },
        "timestamp": str(pd.Timestamp.now()),
        "directory_path": directory_path,
        "file_format": "Pickle, EdgeList, DL, Mappings",
        "original_file": EXCEL_FILE,
        "sheet_name": config['sheet_name'],
        "reference": "Heemskerk, E.M. (2007), Decline of the Corporate Community"
    }
    
    metadata_path = os.path.join(directory_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

def show_network_summary():
    """Show summary of generated networks."""
    print("\n=== DIRECTORS NETWORKS SUMMARY ===")
    
    for network_type, config in DIRECTORS_MEETINGS_CONFIG.items():
        directory_path = config["dirname_template"]
        metadata_path = os.path.join(directory_path, "metadata.json")
        network_path = os.path.join(directory_path, "network.pkl")
        
        if os.path.exists(metadata_path) and os.path.exists(network_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                with open(network_path, 'rb') as f:
                    G = pickle.load(f)
                
                props = metadata.get('network_properties', {})
                mapping_info = metadata.get('node_mapping_info', {})
                
                print(f"\n{config['description_template']}:")
                print(f"  Directors (nodes): {props.get('nodes', 'N/A')}")
                print(f"  Meetings (edges): {props.get('edges', 'N/A')}")
                print(f"  Density: {props.get('density', 'N/A')}")
                print(f"  Average degree: {props.get('average_degree', 'N/A')}")
                print(f"  Connected: {props.get('is_connected', 'N/A')}")
                print(f"  Components: {props.get('number_connected_components', 'N/A')}")
                print(f"  Clustering: {props.get('average_clustering', 'N/A')}")
                print(f"  Path length: {props.get('average_path_length', 'N/A')}")
                print(f"  Node ID range: {props.get('node_id_range', 'N/A')}")
                print(f"  Original ID range: {mapping_info.get('original_id_range', 'N/A')}")
                
            except Exception as e:
                print(f"\n{config['description_template']}: Error reading data - {e}")
        else:
            print(f"\n{config['description_template']}: Not generated yet")

def gen_directors_networks():
    """Generate 3 directors networks with consecutive node IDs."""
    networks_generated = 0
    networks_skipped = 0
    
    if not os.path.exists(EXCEL_FILE):
        print(f"ERROR: Excel file '{EXCEL_FILE}' not found!")
        return
    
    network_types = list(DIRECTORS_MEETINGS_CONFIG.keys())
    
    for config_idx, network_type in enumerate(network_types, 1):
        try:
            config = DIRECTORS_MEETINGS_CONFIG[network_type]
            directory_path = config["dirname_template"]
            
            if not os.path.exists(directory_path):
                edges_raw = load_directors_meetings_data(EXCEL_FILE, config['sheet_name'])
                
                if not edges_raw:
                    continue
                
                Path(directory_path).mkdir(parents=True, exist_ok=True)
                G, node_mapping, reverse_mapping = create_directors_network(edges_raw, network_type)
                
                if G.number_of_nodes() == 0:
                    continue
                
                save_graph(directory_path, G)
                save_node_mappings(directory_path, node_mapping, reverse_mapping)
                save_metadata(directory_path, network_type, G, node_mapping, reverse_mapping)
                
                networks_generated += 1
                print(f"Generated {config['description_template']}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                
            else:
                networks_skipped += 1
                
        except Exception as e:
            print(f"Error generating network {config_idx}: {str(e)}")
            continue
    
    print(f"\nGenerated: {networks_generated} new networks")
    print(f"Skipped: {networks_skipped} existing networks")

if __name__ == "__main__":
    gen_directors_networks()
    show_network_summary()