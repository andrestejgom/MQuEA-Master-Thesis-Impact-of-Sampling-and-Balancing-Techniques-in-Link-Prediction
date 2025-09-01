# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 12:42:25 2025

@author: Andrés
"""
 
import sys
sys.stdout = open('sbm_s166_167_167_pi0.0250_pe0.00250.txt', 'w')

import os
import pickle
import json
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterGrid
from littleballoffur import HybridNodeEdgeSampler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
NETWORK_PATH = "synthetic_networks/stochastic_block_model/sbm_s166_167_167_pi0.0250_pe0.00250"
RESULTS_DIR = "sbm_s166_167_167_pi0.0250_pe0.00250_predictions"
RANDOM_STATE = 42

def load_network(network_path):
    """Load network from pickle file."""
    pickle_path = os.path.join(network_path, "network.pkl")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Network file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    
    # Load metadata for context
    metadata_path = os.path.join(network_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded network: {metadata['description']}")
        print(f"Network properties: {metadata['network_properties']}")
    
    return G

def broder_alg(G, E):
    """
    Runs Andrei Broder's algorithm to select uniformly at random a spanning tree
    of the input graph. The method works for directed and undirected networks. 
    The edge directions in the resulting spanning tree are taken from the input
    edgelist (E). For node pairs where both edge directions exist, one is chosen at random.
    
    This function is adapted from EvalNE library (evalne.utils.split_train_test.broder_alg)
    to fix compatibility issues with Python versions while maintaining algorithmic
    equivalence to the original implementation.
    
    Changes made:
    1. random.sample(S, 1).pop() → np.random.choice(list(S)) where S is still a set
    2. random.sample(neighbors, 1).pop() → np.random.choice(neighbors_list)
    3. random.random() → np.random.random()
    
    This custom version is for undirected graphs only.
    
    Parameters
    ----------
    G : graph
       A NetworkX graph or digraph.
    E : set
       A set of all directed or undirected edges in G.
    Returns
    -------
    train_E : set
       A set of edges of G that form the random spanning tree.
    """
    # Create two partitions, S and T. Initially store all nodes in S.
    S = set(G.nodes)
    T = set()
    
    # Pick a random node as the "current node" and mark it as visited.
    current_node = np.random.choice(list(S))
    S.remove(current_node)
    T.add(current_node)
    
    # Perform random walk on the graph
    train_E = set()
    while S:
        neighbors_list = list(G.neighbors(current_node))
        neighbour_node = np.random.choice(neighbors_list)
            
        if neighbour_node not in T:
            S.remove(neighbour_node)
            T.add(neighbour_node)
            if np.random.random() < 0.5:
                if (current_node, neighbour_node) in E:
                    train_E.add((current_node, neighbour_node))
                else:
                    train_E.add((neighbour_node, current_node))
            else:
                if (neighbour_node, current_node) in E:
                    train_E.add((neighbour_node, current_node))
                else:
                    train_E.add((current_node, neighbour_node))
        current_node = neighbour_node
    
    # Return the set of edges constituting the spanning tree
    return train_E

def create_train_test_split(G, test_size=0.2, neg_ratio=None, random_state=None, sampling_method='random'):
    """
    Create proper train/test splits with correct baseline calculation.
    
    Parameters:
    -----------
    G : networkx.Graph
        Input network graph
    test_size : float
        Proportion of edges for test set (default: 0.2 for 80/20 split)
    neg_ratio : float or None
        Negative sampling ratio (negative samples per positive sample)
        If None, uses ALL possible negative edges (complete negative sampling)
    random_state : int
        Random seed for reproducibility
    sampling_method : str
        Method for negative edge sampling: 'random', 'BRODER', 'HRNE', 'complete'
        Note: 'complete' is used when neg_ratio=None (complete negative sampling)
    
    Returns:
    --------
    dict
        Dictionary containing all split data and metadata
    """
    np.random.seed(random_state)
    
    edges = list(G.edges())
    np.random.shuffle(edges)
    
    n_test = int(len(edges) * test_size)
    
    test_edges = edges[:n_test]
    train_edges = edges[n_test:]
    
    # Create training graph
    G_train = G.copy()
    G_train.remove_edges_from(test_edges)
    
    # Generate non-edges from training graph and original graph
    print("Generating non-edges from training graph...")
    train_non_edges = list(nx.non_edges(G_train))
    np.random.shuffle(train_non_edges)
    
    print("Generating non-edges from original graph...")
    all_non_edges = list(nx.non_edges(G))
    np.random.shuffle(all_non_edges)
    
    # Calculate densities
    original_density = nx.density(G)
    train_density = nx.density(G_train)
    
    # Handle negative sampling based on neg_ratio
    if neg_ratio is None:
        # Complete negative sampling (sampling_method is ignored)
        print(f"Using complete negative sampling with proportional distribution")
        
        # Calculate the same proportions as positive edges
        total_edges = len(train_edges) + len(test_edges)
        train_proportion = len(train_edges) / total_edges  # ~80%
        test_proportion = len(test_edges) / total_edges    # ~20%
        
        print(f"Using proportions - Train: {train_proportion:.1%}, Test: {test_proportion:.1%}")
        
        # Apply same proportions to ALL negative edges
        total_non_edges = len(all_non_edges)
        n_train_non_edges = int(total_non_edges * train_proportion)
        n_test_non_edges = total_non_edges - n_train_non_edges  # Use remaining
        
        # Distribute non-edges proportionally (no sampling needed)
        train_non_edges_selected = all_non_edges[:n_train_non_edges]
        test_non_edges = all_non_edges[n_train_non_edges:]
        
        adjusted_neg_ratio = "None"
        
        print(f"Proportional negative sampling distribution:")
        print(f"  Total possible edges: {G.number_of_nodes() * (G.number_of_nodes() - 1) // 2}")
        print(f"  Total existing edges: {G.number_of_edges()}")
        print(f"  Total possible negative edges: {len(all_non_edges)}")
        print(f"  Train non-edges: {len(train_non_edges_selected)} ({len(train_non_edges_selected)/total_non_edges:.1%})")
        print(f"  Test non-edges: {len(test_non_edges)} ({len(test_non_edges)/total_non_edges:.1%})")
        print(f"  Total non-edges used: {len(train_non_edges_selected) + len(test_non_edges)}")
        
    else:
        # Use specified negative sampling ratio
        adjusted_neg_ratio = neg_ratio
        print(f"Using specified negative sampling ratio: {adjusted_neg_ratio:.2f}:1")
        print(f"Negative sampling method: {sampling_method}")
        
        # Calculate number of negative samples needed
        n_test_non_edges = int(len(test_edges) * adjusted_neg_ratio)
        n_train_non_edges = int(len(train_edges) * adjusted_neg_ratio)
        
        # Check if we have enough non-edges available
        total_required = n_test_non_edges + n_train_non_edges
        if total_required > len(all_non_edges):
            raise ValueError(f"Not enough non-edges available. Required: {total_required}, Available: {len(all_non_edges)}")
        
        if sampling_method.lower() == 'random':
            print("Using RANDOM negative sampling")
            
            # Extract non-edges following proper strategy to avoid overlapping
            # 1. Take train_non_edges from training graph pool
            # 2. Remove those from all_non_edges (original graph pool)
            # 3. Sample test_non_edges from what remains
            
            # Extract train_non_edges from training graph pool
            if n_train_non_edges > len(train_non_edges):
                raise ValueError(f"Not enough train non-edges available. Required: {n_train_non_edges}, Available: {len(train_non_edges)}")
            
            train_non_edges_selected = train_non_edges[:n_train_non_edges]
            print(f"Selected {len(train_non_edges_selected)} train non-edges from training graph")
            
            # Remove train_non_edges from all_non_edges (original graph pool)
            train_non_edges_set = set(train_non_edges_selected)
            
            # Filter out train_non_edges from all_non_edges
            remaining_non_edges = [edge for edge in all_non_edges 
                                  if edge not in train_non_edges_set]
            
            print(f"Remaining non-edges after removing train non-edges: {len(remaining_non_edges)} (from {len(all_non_edges)} original)")
            
            # Check if we have enough remaining non-edges for test
            if n_test_non_edges > len(remaining_non_edges):
                raise ValueError(f"Not enough remaining non-edges for test. Required: {n_test_non_edges}, Available: {len(remaining_non_edges)}")
            
            # Sample test from remaining non-edges
            
            test_non_edges = remaining_non_edges[:n_test_non_edges]
            
        elif sampling_method.upper() == 'BRODER':
            print("Using BRODER (multiple spanning trees) negative sampling")
            
            print(f"  Need {n_train_non_edges} train non-edges and {n_test_non_edges} test non-edges")
            
            # Apply Multiple Broder spanning trees to training graph
            print("  Applying Multiple Broder algorithm to select train non-edges...")
            
            # Create temporary graph with train_non_edges as edges for Broder sampling
            G_train_non_edges = nx.Graph()
            G_train_non_edges.add_nodes_from(G_train.nodes())
            G_train_non_edges.add_edges_from(train_non_edges)
            
            if n_train_non_edges > 0 and len(train_non_edges) > 0:
                train_non_edges_selected = []
                current_graph = G_train_non_edges.copy()  # Working copy
                tree_count = 0
                
                print(f"    Starting with graph containing {current_graph.number_of_edges()} edges.")
                
                # Generate multiple spanning trees until disconnected or target reached
                while (len(train_non_edges_selected) < n_train_non_edges and 
                       nx.is_connected(current_graph) and 
                       current_graph.number_of_edges() > 0):
                    
                    tree_count += 1
                    print(f"    Generating spanning tree #{tree_count}...")
                    
                    try:
                        # Get current available edges
                        current_edges_set = set(current_graph.edges())
                        
                        # Generate spanning tree using broder_alg
                        broder_edges = broder_alg(current_graph, current_edges_set)
                        new_tree_edges = list(broder_edges)
                        
                        print(f"      Tree #{tree_count}: Generated {len(new_tree_edges)} edges")
                        
                        # Add edges from this spanning tree to selected edges
                        train_non_edges_selected.extend(new_tree_edges)
                        
                        # Remove spanning tree edges from working graph
                        current_graph.remove_edges_from(new_tree_edges)
                        
                        print(f"      Removed tree edges. Graph now has {current_graph.number_of_edges()} edges")
                        print(f"      Total selected so far: {len(train_non_edges_selected)} edges")
                        
                        # Check if we have enough edges or if graph became disconnected
                        if len(train_non_edges_selected) >= n_train_non_edges:
                            # Trim to exact number if we exceeded
                            if len(train_non_edges_selected) > n_train_non_edges:
                                np.random.shuffle(train_non_edges_selected)
                                train_non_edges_selected = train_non_edges_selected[:n_train_non_edges]
                                print(f"      Trimmed to exact target: {len(train_non_edges_selected)} edges")
                            break
                        
                        # Check connectivity for next iteration
                        if not nx.is_connected(current_graph):
                            print(f"      Graph became disconnected after tree #{tree_count}")
                            break
                            
                        if current_graph.number_of_edges() < current_graph.number_of_nodes() - 1:
                            print(f"      Not enough edges for another spanning tree")
                            break
                            
                    except Exception as e:
                        print(f"      Spanning tree #{tree_count} generation failed: {e}")
                        break
                
                print(f"    Generated {tree_count} spanning trees with {len(train_non_edges_selected)} total edges")
                
                # Complete with random sampling if needed
                if len(train_non_edges_selected) < n_train_non_edges:
                    # Get edges not yet selected (from original train_non_edges)
                    selected_set = set(train_non_edges_selected)
                    remaining_edges = [e for e in train_non_edges if e not in selected_set]
                    additional_needed = n_train_non_edges - len(train_non_edges_selected)
                    
                    print(f"    Need {additional_needed} more edges. Available: {len(remaining_edges)}")
                    
                    if len(remaining_edges) >= additional_needed:
                        np.random.shuffle(remaining_edges)
                        train_non_edges_selected.extend(remaining_edges[:additional_needed])
                        print(f"    Added {additional_needed} random edges to complete the set")
                    else:
                        train_non_edges_selected.extend(remaining_edges)
                        print(f"    Added all {len(remaining_edges)} remaining edges")
            else:
                train_non_edges_selected = []
            
            print(f"Selected {len(train_non_edges_selected)} train non-edges using Multiple Broder Trees")
            
            # Remove train_non_edges from all_non_edges
            train_non_edges_set = set(train_non_edges_selected)
            train_non_edges_set.update({(v, u) for u, v in train_non_edges_selected})
            
            remaining_non_edges = [edge for edge in all_non_edges 
                                  if edge not in train_non_edges_set]
            
            print(f"Remaining non-edges after removing train non-edges: {len(remaining_non_edges)} (from {len(all_non_edges)} original)")
            
            # Apply Multiple Broder to remaining non-edges to select test_non_edges
            print("  Applying Multiple Broder algorithm to select test non-edges...")
            
            if len(remaining_non_edges) > 0 and n_test_non_edges > 0:
                G_remaining_non_edges = nx.Graph()
                G_remaining_non_edges.add_nodes_from(G.nodes())
                G_remaining_non_edges.add_edges_from(remaining_non_edges)
                
                test_non_edges = []
                current_test_graph = G_remaining_non_edges.copy()  # Working copy
                test_tree_count = 0
                
                print(f"    Starting test selection with graph containing {current_test_graph.number_of_edges()} edges.")
                
                # Generate multiple spanning trees for test set
                while (len(test_non_edges) < n_test_non_edges and 
                       nx.is_connected(current_test_graph) and 
                       current_test_graph.number_of_edges() > 0):
                    
                    test_tree_count += 1
                    print(f"    Generating test spanning tree #{test_tree_count}...")
                    
                    try:
                        # Get current available edges for test
                        current_test_edges_set = set(current_test_graph.edges())
                        
                        # Generate spanning tree using broder_alg
                        broder_test_edges = broder_alg(current_test_graph, current_test_edges_set)
                        new_test_tree_edges = list(broder_test_edges)
                        
                        print(f"      Test tree #{test_tree_count}: Generated {len(new_test_tree_edges)} edges")
                        
                        # Add edges from this spanning tree to test edges
                        test_non_edges.extend(new_test_tree_edges)
                        
                        # Remove spanning tree edges from working test graph
                        current_test_graph.remove_edges_from(new_test_tree_edges)
                        
                        print(f"      Test total selected so far: {len(test_non_edges)} edges")
                        
                        # Check if we have enough test edges
                        if len(test_non_edges) >= n_test_non_edges:
                            # Trim to exact number if we exceeded
                            if len(test_non_edges) > n_test_non_edges:
                                np.random.shuffle(test_non_edges)
                                test_non_edges = test_non_edges[:n_test_non_edges]
                                print(f"      Trimmed test set to exact target: {len(test_non_edges)} edges")
                            break
                        
                        # Check connectivity for next iteration
                        if not nx.is_connected(current_test_graph):
                            print(f"      Test graph became disconnected after tree #{test_tree_count}")
                            break
                            
                        if current_test_graph.number_of_edges() < current_test_graph.number_of_nodes() - 1:
                            print(f"      Not enough edges for another test spanning tree")
                            break
                            
                    except Exception as e:
                        print(f"      Test spanning tree #{test_tree_count} generation failed: {e}")
                        break
                
                print(f"    Generated {test_tree_count} test spanning trees with {len(test_non_edges)} total edges")
                
                # Complete test set with random sampling if needed
                if len(test_non_edges) < n_test_non_edges:
                    test_selected_set = set(test_non_edges)
                    remaining_test_edges = [e for e in remaining_non_edges if e not in test_selected_set]
                    additional_test_needed = n_test_non_edges - len(test_non_edges)
                    
                    print(f"    Need {additional_test_needed} more test edges. Available: {len(remaining_test_edges)}")
                    
                    if len(remaining_test_edges) >= additional_test_needed:
                        np.random.shuffle(remaining_test_edges)
                        test_non_edges.extend(remaining_test_edges[:additional_test_needed])
                        print(f"    Added {additional_test_needed} random test edges to complete the set")
                    else:
                        test_non_edges.extend(remaining_test_edges)
                        print(f"    Added all {len(remaining_test_edges)} remaining test edges")
            else:
                test_non_edges = []
            
            # Ensure correct NEG RATIOS (fallback to random if needed)
            if len(train_non_edges_selected) < n_train_non_edges:
                print(f"  Warning: Broder selected {len(train_non_edges_selected)}, needed {n_train_non_edges}. Adding random samples.")
                remaining_train = [e for e in train_non_edges if e not in train_non_edges_set]
                additional_needed = n_train_non_edges - len(train_non_edges_selected)
                if len(remaining_train) >= additional_needed:
                    np.random.shuffle(remaining_train)
                    train_non_edges_selected.extend(remaining_train[:additional_needed])
            
            if len(test_non_edges) < n_test_non_edges:
                print(f"  Warning: Broder selected {len(test_non_edges)}, needed {n_test_non_edges}. Adding random samples.")
                current_test_set = set(test_non_edges)
                remaining_test = [e for e in remaining_non_edges if e not in current_test_set]
                additional_needed = n_test_non_edges - len(test_non_edges)
                if len(remaining_test) >= additional_needed:
                    np.random.shuffle(remaining_test)
                    test_non_edges.extend(remaining_test[:additional_needed])
        
        elif sampling_method.upper() == 'HRNE':
            print("Using HRNE (HybridNodeEdgeSampler) negative sampling")
            
            print(f"  Need {n_train_non_edges} train non-edges and {n_test_non_edges} test non-edges")
            
            # Apply HRNE to training graph to select train_non_edges_selected
            print("  Applying HRNE to select train non-edges...")
            
            # Create temporary graph with train_non_edges as edges for HRNE sampling
            G_train_non_edges = nx.Graph()
            G_train_non_edges.add_nodes_from(G_train.nodes())
            G_train_non_edges.add_edges_from(train_non_edges)
            
            if n_train_non_edges > 0 and len(train_non_edges) > 0:
                sampler_train = HybridNodeEdgeSampler(number_of_edges=min(n_train_non_edges, len(train_non_edges)), seed=random_state)
                G_train_sampled = sampler_train.sample(G_train_non_edges)
                train_non_edges_selected = list(G_train_sampled.edges())
            else:
                train_non_edges_selected = []
            
            print(f"Selected {len(train_non_edges_selected)} train non-edges using HRNE")
            
            # Remove train_non_edges from all_non_edges
            train_non_edges_set = set(train_non_edges_selected)
            train_non_edges_set.update({(v, u) for u, v in train_non_edges_selected})
            
            remaining_non_edges = [edge for edge in all_non_edges 
                                  if edge not in train_non_edges_set]
            
            print(f"Remaining non-edges after removing train non-edges: {len(remaining_non_edges)} (from {len(all_non_edges)} original)")
            
            # Apply HRNE to remaining non-edges to select test_non_edges
            print("  Applying HRNE to select test non-edges...")
            
            if len(remaining_non_edges) > 0 and n_test_non_edges > 0:
                G_remaining_non_edges = nx.Graph()
                G_remaining_non_edges.add_nodes_from(G.nodes())
                G_remaining_non_edges.add_edges_from(remaining_non_edges)
                
                sampler_test = HybridNodeEdgeSampler(number_of_edges=min(n_test_non_edges, len(remaining_non_edges)), seed=random_state + 1)
                G_test_sampled = sampler_test.sample(G_remaining_non_edges)
                test_non_edges = list(G_test_sampled.edges())
            else:
                test_non_edges = []
                
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}. Choose from: 'random', 'BRODER', 'HRNE'")
    
    # Use original network density as proper baseline for AUPR
    expected_aupr = original_density
    print(f"Using original network density {original_density:.6f} as AUPR baseline")
    
    print(f"Split created:")
    print(f"  Original network density: {original_density:.6f}")
    if neg_ratio is None:
        print(f"  Train: {len(train_edges)} edges, {len(train_non_edges_selected)} non-edges (complete sampling)")
        print(f"  Test: {len(test_edges)} edges, {len(test_non_edges)} non-edges")
        print(f"  Complete negative sampling applied")
    else:
        print(f"  Train: {len(train_edges)} edges, {len(train_non_edges_selected)} non-edges (ratio: 1:{len(train_non_edges_selected)/len(train_edges):.2f})")
        print(f"  Test: {len(test_edges)} edges, {len(test_non_edges)} non-edges")
        print(f"  Negative sampling ratio used: {adjusted_neg_ratio:.1f}:1")
        print(f"  Sampling method used: {sampling_method}")
    print(f"  Expected AUPR baseline: {expected_aupr:.6f}")
    print(f"  Training graph density: {train_density:.6f}")
    
    return {
        'train_graph': G_train,
        'train_edges': train_edges,
        'train_non_edges': train_non_edges_selected,
        'test_edges': test_edges,
        'test_non_edges': test_non_edges,
        'expected_aupr': expected_aupr,
        'original_density': original_density,
        'train_density': train_density,
        'neg_ratio': adjusted_neg_ratio,
        'sampling_method': sampling_method
    }


#################### TOPOLOGY PRESERVATION: FUNCIONS ##########################

def add_topology_analysis_to_split_result(G_original, split_result):
    """
    Simple topology analysis for dense auxiliary graphs using 4 optimized metrics.
    
    Metrics: spectral gap, hub node preservation, original graph structure preservation, 
    and sampling quality index.
    """
    
    # Skip topology analysis for complete sampling
    if split_result['neg_ratio'] == "None" or split_result['neg_ratio'] is None:
        split_result['topology_analysis'] = {
            'spectral_gap_preservation': 1.0,
            'hub_node_preservation': 1.0,
            'original_graph_structure_preservation': 1.0,
            'sampling_quality_index': 1.0,
            'note': 'Reference (Complete sampling)'
        }
        return split_result
    
    print("  Computing optimized topology metrics...")
    
    # ===== GENERATE AUXILIARY GRAPHS =====
    all_neg_edges = list(nx.non_edges(G_original))
    sampled_neg_edges = split_result['train_non_edges'] + split_result['test_non_edges']
    
    # Complete auxiliary graph (reference)
    G_complete_aux = nx.Graph()
    G_complete_aux.add_nodes_from(G_original.nodes())
    G_complete_aux.add_edges_from(all_neg_edges)
    
    # Sampled auxiliary graph
    G_sampled_aux = nx.Graph()
    G_sampled_aux.add_nodes_from(G_original.nodes())
    G_sampled_aux.add_edges_from(sampled_neg_edges)
    
    # ===== CALCULATE COMPLETENESS WEIGHT =====
    completeness = len(sampled_neg_edges) / len(all_neg_edges) if len(all_neg_edges) > 0 else 0
    # Sigmoid function to penalize small samples
    completeness_weight = 1 / (1 + np.exp(-8 * (completeness - 0.2)))
    
    # ===== 1. SPECTRAL GAP PRESERVATION =====
    try:
        eigenvals_complete = np.sort(nx.laplacian_spectrum(G_complete_aux))
        eigenvals_sampled = np.sort(nx.laplacian_spectrum(G_sampled_aux))
        
        gap_complete = eigenvals_complete[1] - eigenvals_complete[0]
        gap_sampled = eigenvals_sampled[1] - eigenvals_sampled[0]
        
        if gap_complete > 0:
            spectral_gap_preservation = min(1.0, gap_sampled / gap_complete)
        else:
            spectral_gap_preservation = 1.0 if gap_sampled <= gap_complete else 0.0
    except Exception as e:
        print(f"    Error in spectral gap analysis: {e}")
        spectral_gap_preservation = 0.0
        gap_complete = gap_sampled = 0
    
    # ===== 2. HUB NODE PRESERVATION =====
    try:
        degrees_complete = dict(G_complete_aux.degree())
        degrees_sampled = dict(G_sampled_aux.degree())
        
        sorted_nodes_complete = sorted(degrees_complete.items(), key=lambda x: x[1], reverse=True)
        n_hubs = max(1, len(sorted_nodes_complete) // 10)
        hub_nodes = [node for node, degree in sorted_nodes_complete[:n_hubs]]
        
        hub_preservation_scores = []
        for hub in hub_nodes:
            degree_complete = degrees_complete[hub]
            degree_sampled = degrees_sampled[hub]
            if degree_complete > 0:
                preservation = min(1.0, degree_sampled / degree_complete)
                hub_preservation_scores.append(preservation)
        
        hub_node_preservation = np.mean(hub_preservation_scores) if hub_preservation_scores else 0.0
    except Exception as e:
        print(f"    Error in hub preservation analysis: {e}")
        hub_node_preservation = 0.0
    
    # ===== 3. ORIGINAL GRAPH STRUCTURE PRESERVATION =====
    try:
        # Measure how well the sampling preserves structural patterns from the original graph
        # Focus on: common neighbors, shortest paths, and degree correlations
        
        sampled_edges = split_result['train_non_edges'] + split_result['test_non_edges']
        original_degrees = dict(G_original.degree())
        
        # 3.1 Common neighbors preservation score
        cn_scores = []
        for u, v in sampled_edges[:min(200, len(sampled_edges))]:  # Sample for efficiency
            cn_count = len(list(nx.common_neighbors(G_original, u, v)))
            cn_scores.append(cn_count)
        
        mean_cn_sampled = np.mean(cn_scores) if cn_scores else 0
        
        # Compare with expected common neighbors for random sampling
        if len(all_neg_edges) > 500:
            random_sample = np.random.choice(len(all_neg_edges), size=min(500, len(all_neg_edges)), replace=False)
            random_edges = [all_neg_edges[i] for i in random_sample]
        else:
            random_edges = all_neg_edges
        
        cn_scores_random = []
        for u, v in random_edges[:200]:
            cn_count = len(list(nx.common_neighbors(G_original, u, v)))
            cn_scores_random.append(cn_count)
        
        mean_cn_random = np.mean(cn_scores_random) if cn_scores_random else 0
        
        # 3.2 Shortest path preservation
        path_scores = []
        for u, v in sampled_edges[:min(100, len(sampled_edges))]:
            try:
                path_length = nx.shortest_path_length(G_original, u, v)
                path_scores.append(1.0 / path_length if path_length > 0 else 0)
            except:
                path_scores.append(0)
        
        mean_path_sampled = np.mean(path_scores) if path_scores else 0
        
        path_scores_random = []
        for u, v in random_edges[:100]:
            try:
                path_length = nx.shortest_path_length(G_original, u, v)
                path_scores_random.append(1.0 / path_length if path_length > 0 else 0)
            except:
                path_scores_random.append(0)
        
        mean_path_random = np.mean(path_scores_random) if path_scores_random else 0
        
        # 3.3 Degree correlation preservation
        degree_products_sampled = []
        for u, v in sampled_edges[:min(200, len(sampled_edges))]:
            product = original_degrees.get(u, 1) * original_degrees.get(v, 1)
            degree_products_sampled.append(product)
        
        mean_degree_product_sampled = np.mean(degree_products_sampled) if degree_products_sampled else 0
        
        degree_products_random = []
        for u, v in random_edges[:200]:
            product = original_degrees.get(u, 1) * original_degrees.get(v, 1)
            degree_products_random.append(product)
        
        mean_degree_product_random = np.mean(degree_products_random) if degree_products_random else 0
        
        # Combine structural measures (higher = better structure preservation)
        cn_advantage = mean_cn_sampled / mean_cn_random if mean_cn_random > 0 else 1.0
        path_advantage = mean_path_sampled / mean_path_random if mean_path_random > 0 else 1.0
        degree_advantage = mean_degree_product_sampled / mean_degree_product_random if mean_degree_product_random > 0 else 1.0
        
        # Normalize advantages (cap at 2.0 for reasonable scale)
        cn_advantage = min(2.0, cn_advantage) / 2.0
        path_advantage = min(2.0, path_advantage) / 2.0
        degree_advantage = min(2.0, degree_advantage) / 2.0
        
        # Calculate base structure score
        base_structure_score = (0.4 * cn_advantage + 0.3 * path_advantage + 0.3 * degree_advantage)
        
        # Apply completeness weighting
        original_graph_structure_preservation = base_structure_score * completeness_weight
        
    except Exception as e:
        print(f"    Error in original graph structure analysis: {e}")
        original_graph_structure_preservation = 0.0
    
    # ===== 4. SAMPLING QUALITY INDEX =====
    try:
        # Comprehensive measure of sampling quality combining multiple factors
        sampled_edges = split_result['train_non_edges'] + split_result['test_non_edges']
        
        # 4.1 Coverage uniformity (how evenly distributed across original graph structure)
        original_degrees = dict(G_original.degree())
        
        # Group nodes by degree quartiles in original graph
        all_degrees = list(original_degrees.values())
        q1 = np.percentile(all_degrees, 25)
        q2 = np.percentile(all_degrees, 50)
        q3 = np.percentile(all_degrees, 75)
        
        quartile_coverage = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        
        for u, v in sampled_edges:
            deg_u = original_degrees.get(u, 0)
            deg_v = original_degrees.get(v, 0)
            
            # Count edges touching each quartile
            for deg in [deg_u, deg_v]:
                if deg <= q1:
                    quartile_coverage['Q1'] += 0.5
                elif deg <= q2:
                    quartile_coverage['Q2'] += 0.5
                elif deg <= q3:
                    quartile_coverage['Q3'] += 0.5
                else:
                    quartile_coverage['Q4'] += 0.5
        
        # Calculate uniformity (lower coefficient of variation = better)
        coverage_values = list(quartile_coverage.values())
        if coverage_values and np.mean(coverage_values) > 0:
            cv_coverage = np.std(coverage_values) / np.mean(coverage_values)
            coverage_uniformity = max(0, 1 - cv_coverage)
        else:
            coverage_uniformity = 0
        
        # 4.2 Structural diversity (variety of edge types sampled)
        edge_types = {'low_low': 0, 'low_med': 0, 'low_high': 0, 'med_med': 0, 'med_high': 0, 'high_high': 0}
        
        for u, v in sampled_edges:
            deg_u = original_degrees.get(u, 0)
            deg_v = original_degrees.get(v, 0)
            
            # Classify edge type
            if deg_u > deg_v:
                deg_u, deg_v = deg_v, deg_u  # Ensure deg_u <= deg_v
            
            if deg_u <= q1 and deg_v <= q1:
                edge_types['low_low'] += 1
            elif deg_u <= q1 and deg_v <= q2:
                edge_types['low_med'] += 1
            elif deg_u <= q1 and deg_v > q2:
                edge_types['low_high'] += 1
            elif deg_u <= q2 and deg_v <= q2:
                edge_types['med_med'] += 1
            elif deg_u <= q2 and deg_v > q2:
                edge_types['med_high'] += 1
            else:
                edge_types['high_high'] += 1
        
        # Calculate type diversity (entropy-based)
        total_edges = sum(edge_types.values())
        if total_edges > 0:
            type_entropy = 0
            for count in edge_types.values():
                if count > 0:
                    p = count / total_edges
                    type_entropy -= p * np.log2(p)
            
            max_entropy = np.log2(6)  # 6 edge types
            structural_diversity = type_entropy / max_entropy if max_entropy > 0 else 0
        else:
            structural_diversity = 0
        
        # 4.3 Sampling efficiency (avoid redundancy, maximize information)
        # Check for clustering in sampled edges (lower clustering = better coverage)
        sampled_edges_set = set(sampled_edges)
        triangles_in_sampled = 0
        possible_triangles = 0
        
        for u, v in sampled_edges[:min(100, len(sampled_edges))]:  # Sample for efficiency
            neighbors_u = set(w for w in G_original.neighbors(u) if (u, w) in sampled_edges_set or (w, u) in sampled_edges_set)
            neighbors_v = set(w for w in G_original.neighbors(v) if (v, w) in sampled_edges_set or (w, v) in sampled_edges_set)
            
            common_neighbors = neighbors_u & neighbors_v
            triangles_in_sampled += len(common_neighbors)
            possible_triangles += min(len(neighbors_u), len(neighbors_v))
        
        if possible_triangles > 0:
            sampling_clustering = triangles_in_sampled / possible_triangles
            sampling_efficiency = max(0, 1 - sampling_clustering)  # Lower clustering = higher efficiency
        else:
            sampling_efficiency = 1.0
        
        # Calculate base quality score
        base_quality_score = (0.4 * coverage_uniformity + 0.4 * structural_diversity + 0.2 * sampling_efficiency)
        
        # Apply completeness weighting
        sampling_quality_index = base_quality_score * completeness_weight
        
    except Exception as e:
        print(f"    Error in sampling quality analysis: {e}")
        sampling_quality_index = 0.0
    
    # ===== ADD RESULTS TO SPLIT =====
    split_result['topology_analysis'] = {
        # Main metrics for histogram
        'spectral_gap_preservation': round(spectral_gap_preservation, 4),
        'hub_node_preservation': round(hub_node_preservation, 4),
        'original_graph_structure_preservation': round(original_graph_structure_preservation, 4),
        'sampling_quality_index': round(sampling_quality_index, 4),
        
        # Additional details
        'spectral_gap_complete': round(gap_complete, 4),
        'spectral_gap_sampled': round(gap_sampled, 4),
        'completeness': round(completeness, 6),
        'completeness_weight': round(completeness_weight, 4),
        
        # Basic auxiliary graph info
        'complete_aux_graph': {
            'nodes': G_complete_aux.number_of_nodes(),
            'edges': G_complete_aux.number_of_edges(),
            'density': round(nx.density(G_complete_aux), 6),
            'is_connected': nx.is_connected(G_complete_aux)
        },
        'sampled_aux_graph': {
            'nodes': G_sampled_aux.number_of_nodes(),
            'edges': G_sampled_aux.number_of_edges(),
            'density': round(nx.density(G_sampled_aux), 6),
            'is_connected': nx.is_connected(G_sampled_aux)
        }
    }
    
    return split_result

def create_topology_preservation_histogram(all_results, save_path=None):
    """Create 4 histograms for optimized topology preservation metrics."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    topology_data = []
    
    for result in all_results:
        neg_ratio = result['neg_ratio']
        sampling_method = result['sampling_method']
        
        if neg_ratio is not None and neg_ratio != "None":
            if 'topology_analysis' in result:
                topo = result['topology_analysis']
                
                topology_data.append({
                    'neg_ratio': float(neg_ratio),
                    'sampling_method': sampling_method,
                    'spectral_gap_preservation': topo['spectral_gap_preservation'],
                    'hub_node_preservation': topo['hub_node_preservation'],
                    'original_graph_structure_preservation': topo['original_graph_structure_preservation'],
                    'sampling_quality_index': topo['sampling_quality_index']
                })
    
    df = pd.DataFrame(topology_data)
    
    if df.empty:
        print("No topology data to display")
        return
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    colors = {
        'random': '#1f77b4',
        'BRODER': '#d62728', 
        'HRNE': '#ff7f0e'
    }
    
    sampling_methods = ['random', 'BRODER', 'HRNE']
    unique_neg_ratios = sorted(df['neg_ratio'].unique())
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, 
                         top=0.88, bottom=0.12, left=0.08, right=0.95)
    
    metrics = [
        ('spectral_gap_preservation', 'Spectral Gap Preservation'),
        ('hub_node_preservation', 'Hub Node Preservation'),
        ('original_graph_structure_preservation', 'Graph Structure Preservation'),
        ('sampling_quality_index', 'Sampling Quality Index')
    ]
    
    x = np.arange(len(unique_neg_ratios))
    width = 0.25
    
    for idx, (metric_key, metric_title) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx//2, idx%2])
        
        bars_collection = []
        
        for i, method in enumerate(sampling_methods):
            method_data = df[df['sampling_method'] == method]
            
            scores = []
            for neg_ratio in unique_neg_ratios:
                method_ratio_data = method_data[method_data['neg_ratio'] == neg_ratio]
                if not method_ratio_data.empty:
                    scores.append(method_ratio_data[metric_key].iloc[0])
                else:
                    scores.append(0)
            
            bars = ax.bar(x + i * width - width, scores, width,
                         label=method.upper(), 
                         color=colors[method], 
                         alpha=0.85,
                         edgecolor='white',
                         linewidth=1.5,
                         zorder=3)
            
            bars_collection.append(bars)
            
            for j, (bar, score) in enumerate(zip(bars, scores)):
                if score > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                           f'{score:.2f}',
                           ha='center', va='bottom', 
                           fontsize=10, fontweight='600',
                           color='#2c3e50')
        
        ax.set_xlabel('Negative Sampling Ratio', fontsize=13, fontweight='600', color='#2c3e50')
        ax.set_ylabel('Preservation Score', fontsize=13, fontweight='600', color='#2c3e50')
        ax.set_title(metric_title, fontsize=15, fontweight='700', color='#2c3e50', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{ratio:.0f}:1' for ratio in unique_neg_ratios], fontsize=11)
        
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f'{y:.1f}' for y in np.arange(0, 1.1, 0.2)], fontsize=10)
        
        ax.axhline(y=0.8, color='#27ae60', linestyle='--', alpha=0.7, linewidth=2, zorder=1)
        ax.axhline(y=0.6, color='#f39c12', linestyle='--', alpha=0.7, linewidth=2, zorder=1)
        ax.axhline(y=0.4, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2, zorder=1)
        
        ax.text(ax.get_xlim()[1], 0.8, 'Excellent', va='center', ha='left', 
                fontsize=9, color='#27ae60', fontweight='600', alpha=0.8)
        ax.text(ax.get_xlim()[1], 0.6, 'Good', va='center', ha='left',
                fontsize=9, color='#f39c12', fontweight='600', alpha=0.8)
        ax.text(ax.get_xlim()[1], 0.4, 'Fair', va='center', ha='left',
                fontsize=9, color='#e74c3c', fontweight='600', alpha=0.8)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        
        if idx == 0:
            legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                             fancybox=True, shadow=True, ncol=1)
            legend.get_frame().set_facecolor('#f8f9fa')
            legend.get_frame().set_edgecolor('#dee2e6')
    
    fig.suptitle('Topology Preservation Analysis: Structured vs Random Sampling', 
                fontsize=20, fontweight='700', color='#2c3e50', y=0.95)
    
    subtitle_text = ('Performance comparison across negative sampling strategies with completeness weighting\n'
                    'Higher scores indicate better preservation of original network topology')
    fig.text(0.5, 0.02, subtitle_text, fontsize=12, ha='center', color='#7f8c8d',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8, edgecolor='none'))
    
    if save_path:
        plot_path = os.path.join(save_path, "final_topology_histogram_weighted.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        print(f"Final topology histogram saved to: {plot_path}")
    
    plt.show()
    
    print("\n=== TOPOLOGY PRESERVATION ANALYSIS ===")
    for metric_key, metric_title in metrics:
        print(f"\n{metric_title.upper()}:")
        summary_stats = df.groupby('sampling_method')[metric_key].agg(['mean', 'std', 'min', 'max']).round(3)
        print(summary_stats)
    
    print("\n=== COMPLETENESS ANALYSIS ===")
    for result in all_results:
        if result['neg_ratio'] is not None and result['neg_ratio'] != "None":
            if 'topology_analysis' in result:
                topo = result['topology_analysis']
                print(f"Neg Ratio {result['neg_ratio']:.1f} ({result['sampling_method']}): "
                      f"Completeness = {topo['completeness']:.4f} "
                      f"(Weight = {topo['completeness_weight']:.3f})")
    
    return df

#################### CUSTOM FUNCTIONS: LINK PREDICTION ########################

def yang_zhang_similarity(G, ebunch=None):
   """
   Yang-Zhang link prediction method (2016).
   Formula: score = (common_neighbors + 1) / distance
   
   Parameters:
   -----------
   G : graph
   ebunch : iterable of node pairs
       Node pairs to evaluate. If None, uses all non-edges.
       
   Yields:
   -------
   tuple
       (u, v, Yang-Zhang score) for each pair in ebunch
   """
   import networkx as nx
   
   if ebunch is None:
       ebunch = nx.non_edges(G)
   
   for u, v in ebunch:
       # Calculate common neighbors using NetworkX
       common_neighbors = len(list(nx.common_neighbors(G, u, v)))
       
       # Calculate shortest path distance
       try:
           distance = nx.shortest_path_length(G, u, v)
       except nx.NetworkXNoPath:
           # Disconnected nodes get score 0
           score = 0.0
       else:
           # Yang-Zhang formula: (common_neighbors + 1) / distance
           score = (common_neighbors + 1) / distance
       
       yield (u, v, score)

def degree_of_popularity(G, ebunch=None):
    """
    Calculate Degree of Popularity (DP) similarity for node pairs.
    
    Formula: DP(x,y) = max(k(x), k(y)) / (n-1)
    where k(x) is the degree of node x, and n is the number of nodes.
    
    Based on: "Predicting missing links in complex networks using information 
    about common neighbors and a degree of popularity" (Abdelhamid Saif, 2025)
    """
    if ebunch is None:
        ebunch = nx.non_edges(G)
    
    n = G.number_of_nodes()
    
    for u, v in ebunch:
        degree_u = G.degree(u)
        degree_v = G.degree(v)
        
        # DP formula from the paper
        dp_score = max(degree_u, degree_v) / (n - 1) if n > 1 else 0
        
        yield (u, v, dp_score)


def dp_jaccard_coefficient(G, ebunch=None):
    """
    Enhanced Jaccard Coefficient with Degree of Popularity.
    
    Original Jaccard: |Γ(x)∩Γ(y)| / |Γ(x)∪Γ(y)|
    Enhanced: Jaccard + DP
    """
    if ebunch is None:
        ebunch = nx.non_edges(G)
    
    n = G.number_of_nodes()
    edge_pairs = list(ebunch)
    
    # Get original Jaccard scores 
    jaccard_scores = {(u, v): score for u, v, score in nx.jaccard_coefficient(G, ebunch=edge_pairs)}
    
    # Get DP scores 
    dp_scores = {(u, v): score for u, v, score in degree_of_popularity(G, ebunch=edge_pairs)}
    
    # Combine both measures
    for u, v in edge_pairs:
        jaccard_score = jaccard_scores.get((u, v), 0)
        dp_score = dp_scores.get((u, v), 0)
        
        # Enhanced score = Original + DP
        enhanced_score = jaccard_score + dp_score
        
        yield (u, v, enhanced_score)

def dp_yang_zhang_similarity(G, ebunch=None):
    """
    Enhanced Yang-Zhang similarity with Degree of Popularity.
    
    Original Yang-Zhang: (common_neighbors + 1) / distance
    Enhanced: Yang-Zhang + DP
    """
    if ebunch is None:
        ebunch = nx.non_edges(G)
    
    n = G.number_of_nodes()
    edge_pairs = list(ebunch)
    
    # Get original Yang-Zhang scores 
    yz_scores = {(u, v): score for u, v, score in yang_zhang_similarity(G, ebunch=edge_pairs)}
    
    # Get DP scores 
    dp_scores = {(u, v): score for u, v, score in degree_of_popularity(G, ebunch=edge_pairs)}
    
    # Combine both measures
    for u, v in edge_pairs:
        yz_score = yz_scores.get((u, v), 0)
        dp_score = dp_scores.get((u, v), 0)
        
        # Enhanced score = Original + DP
        enhanced_score = yz_score + dp_score
        
        yield (u, v, enhanced_score)

def dp_preferential_attachment_scaled(G, ebunch=None):
    """
    Enhanced Preferential Attachment with Scaled Degree of Popularity.
    
    Original PA: k(x) * k(y)
    Enhanced: PA + (α × DP)
    Where the scaling factor α = mean_degree² / (max_degree / (n-1))
    
    - mean_degree² = typical PA score
    - (max_degree / (n-1)) = maximum DP score
    """
    if ebunch is None:
        ebunch = nx.non_edges(G)
    
    n = G.number_of_nodes()
    edge_pairs = list(ebunch)
    
    # Calculate scaling factor α
    degrees = [G.degree(node) for node in G.nodes()]
    mean_degree = np.mean(degrees)
    max_degree = max(degrees)
    
    # Scaling factor
    alpha = (mean_degree ** 2) / (max_degree / (n - 1))
    
    # Get PA scores
    pa_scores = {(u, v): score for u, v, score in nx.preferential_attachment(G, ebunch=edge_pairs)}
    
    # Get DP scores
    dp_scores = {(u, v): score for u, v, score in degree_of_popularity(G, ebunch=edge_pairs)}
    
    # Combine both measures
    for u, v in edge_pairs:
        pa_score = pa_scores.get((u, v), 0)
        dp_score = dp_scores.get((u, v), 0)
        
        # Enhanced score = Original + Scaled DP
        enhanced_score = pa_score + (alpha * dp_score)
        
        yield (u, v, enhanced_score)

###############################################################################

# Negative sampling ratios configuration
NEG_RATIOS = [None, 20.0, 10.0, 5.0, 2.0, 1.0]

# Sampling methods configuration
SAMPLING_METHODS = ['random', 'BRODER', 'HRNE']

# Link prediction methods configuration
LINK_PREDICTION_METHODS = {
    'preferential_attachment': nx.preferential_attachment,
    'jaccard_coefficient': nx.jaccard_coefficient,
    'yang_zhang': yang_zhang_similarity,
    'dp_jaccard_coefficient': dp_jaccard_coefficient,
    'dp_yang_zhang': dp_yang_zhang_similarity,
    'dp_preferential_attachment_scaled': dp_preferential_attachment_scaled
}

###############################################################################

def evaluate_link_prediction_method(method_name, method_config, data_split):
    """
    Evaluate a link prediction method with AUPR Improvement Factor as main metric.
    
    Parameters:
    -----------
    method_name : str
        Name of the link prediction method
    method_config : callable
        Configuration dictionary for the method
    data_split : dict
        Data split dictionary from create_train_test_split
    
    Returns:
    --------
    dict
        Results including performance metrics
    """
    
    G_train = data_split['train_graph']
    test_edges = data_split['test_edges']
    test_non_edges = data_split['test_non_edges']
    original_density = data_split['original_density']  # Random classifier baseline
    
    # Evaluate on test set
    print(f"  Evaluating {method_name} on test set...")
    
    test_true_labels = [1] * len(test_edges) + [0] * len(test_non_edges)
    test_edge_pairs = test_edges + test_non_edges
    
    try:
        predictions = method_config(G_train, ebunch=test_edge_pairs)
        test_scores = [score for _, _, score in predictions]
        
        # Calculate performance metrics
        test_auc = roc_auc_score(test_true_labels, test_scores)
        test_aupr = average_precision_score(test_true_labels, test_scores)
        
        # Calculate AUPR Improvement Factor (main metric)
        aupr_improvement_factor = test_aupr / original_density 
        
        print(f"  Test Results:")
        print(f"    AUPR Improvement Factor: x{aupr_improvement_factor:.1f}")
        print(f"    Test AUC: {test_auc:.4f}")
        print(f"    Test AUPR: {test_aupr:.4f} (baseline: {original_density:.4f})")
        
        return {
            'method': method_name,
            'aupr_improvement_factor': aupr_improvement_factor,  # Main metric
            'test_auc': test_auc,
            'test_aupr': test_aupr,
            'random_baseline': original_density,
            'test_scores': test_scores,
            'test_labels': test_true_labels
        }
        
    except Exception as e:
        print(f"  Error evaluating {method_name}: {str(e)}")
        return {
            'method': method_name,
            'error': str(e)
        }

def run_experiment_for_neg_ratio(G, neg_ratio, sampling_method, experiment_id):
    """
    Run complete experiment for a specific negative sampling ratio and sampling method.
    
    Parameters:
    -----------
    G : networkx.Graph
        Input network graph
    neg_ratio : float or None
        Negative sampling ratio to test
    sampling_method : str
        Sampling method to use ('random', 'BRODER', 'HRNE', 'complete')
    experiment_id : int
        Experiment identifier
    
    Returns:
    --------
    dict
        Complete experiment results
    """
    if neg_ratio is None:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_id}: Complete Negative Sampling")
        print(f"{'='*80}")
        # Force sampling_method to 'complete' for clarity
        sampling_method = 'complete'
    else:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_id}: Negative Ratio = {neg_ratio}, Sampling = {sampling_method}")
        print(f"{'='*80}")
    
    # Create data split with specified sampling method
    data_split = create_train_test_split(G, neg_ratio=neg_ratio, 
                                       sampling_method=sampling_method, 
                                       random_state=RANDOM_STATE)
    
    print("Adding topology analysis to split result...")
    data_split = add_topology_analysis_to_split_result(G, data_split)
    
    # Store experiment metadata
    experiment_results = {
        'experiment_id': experiment_id,
        'neg_ratio': neg_ratio,
        'sampling_method': sampling_method,
        'data_split_info': {
            'train_edges': len(data_split['train_edges']),
            'train_non_edges': len(data_split['train_non_edges']),
            'test_edges': len(data_split['test_edges']),
            'test_non_edges': len(data_split['test_non_edges']),
            'original_density': data_split['original_density'],
            'train_density': data_split['train_density'],
            'expected_aupr': data_split['expected_aupr']
        },
        'method_results': {}
    }
    
    if 'topology_analysis' in data_split:
        experiment_results['topology_analysis'] = data_split['topology_analysis']
    
    # Test each link prediction method
    for method_name, method_config in LINK_PREDICTION_METHODS.items():
        print(f"\nTesting {method_name}...")
        
        method_results = evaluate_link_prediction_method(method_name, method_config, data_split)
        experiment_results['method_results'][method_name] = method_results
    
    return experiment_results

def save_experiment_results(all_results, network_info):
    """Save experiment results to a single Excel file."""
    import pandas as pd
    from pathlib import Path
    
    # Create results directory
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Create summary data
    summary_data = []
    for result in all_results:
        # Get topology analysis if available
        topology = result.get('topology_analysis', {})
        
        for method_name, method_result in result['method_results'].items():
            if 'error' not in method_result:
                row = {
                    'neg_ratio': result['neg_ratio'],
                    'sampling_method': result['sampling_method'],
                    'method': method_name,
                    'aupr_improvement_factor': method_result['aupr_improvement_factor'],
                    'test_auc': method_result['test_auc'],
                    'test_aupr': method_result['test_aupr'],
                    'random_baseline': method_result['random_baseline'],
                    'train_edges': result['data_split_info']['train_edges'],
                    'train_non_edges': result['data_split_info']['train_non_edges'],
                    'test_edges': result['data_split_info']['test_edges'],
                    'test_non_edges': result['data_split_info']['test_non_edges'],
                    'original_density': result['data_split_info']['original_density'],
                    'train_density': result['data_split_info']['train_density'],
                    # Add topology metrics
                    'spectral_gap_preservation': topology.get('spectral_gap_preservation'),
                    'hub_node_preservation': topology.get('hub_node_preservation'),
                    'original_graph_structure_preservation': topology.get('original_graph_structure_preservation'),
                    'sampling_quality_index': topology.get('sampling_quality_index'),
                    'completeness': topology.get('completeness'),
                    'completeness_weight': topology.get('completeness_weight')
                }
                summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by main metric (AUPR improvement factor) in descending order
    summary_df = summary_df.sort_values('aupr_improvement_factor', ascending=False)
    
    # Save to Excel file
    excel_file = os.path.join(RESULTS_DIR, "sbm_s166_167_167_pi0.0250_pe0.00250_results.xlsx")
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Main results sheet
        summary_df.to_excel(writer, sheet_name='Results', index=False)
        
        # Network information sheet
        network_df = pd.DataFrame([network_info])
        network_df.to_excel(writer, sheet_name='Network_Info', index=False)
        
        # Best performing combinations sheet
        best_combinations = summary_df.head(20).copy()  # Top 20 instead of 10
        best_combinations.to_excel(writer, sheet_name='Top_20_Best', index=False)
        
        # Summary statistics by method
        method_stats = summary_df.groupby('method').agg({
            'aupr_improvement_factor': ['mean', 'std', 'max', 'min'],
            'test_auc': ['mean', 'std', 'max', 'min'],
            'test_aupr': ['mean', 'std', 'max', 'min']
        }).round(4)
        method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns.values]
        method_stats.to_excel(writer, sheet_name='Method_Statistics')
        
        # Summary statistics by negative ratio
        ratio_stats = summary_df.groupby('neg_ratio').agg({
            'aupr_improvement_factor': ['mean', 'std', 'max', 'min'],
            'test_auc': ['mean', 'std', 'max', 'min'],
            'test_aupr': ['mean', 'std', 'max', 'min']
        }).round(4)
        ratio_stats.columns = ['_'.join(col).strip() for col in ratio_stats.columns.values]
        ratio_stats.to_excel(writer, sheet_name='Ratio_Statistics')
        
        # Summary statistics by sampling method
        sampling_stats = summary_df.groupby('sampling_method').agg({
            'aupr_improvement_factor': ['mean', 'std', 'max', 'min'],
            'test_auc': ['mean', 'std', 'max', 'min'],
            'test_aupr': ['mean', 'std', 'max', 'min']
        }).round(4)
        sampling_stats.columns = ['_'.join(col).strip() for col in sampling_stats.columns.values]
        sampling_stats.to_excel(writer, sheet_name='Sampling_Statistics')
        
        # Combined statistics by ratio + sampling method
        combined_stats = summary_df.groupby(['neg_ratio', 'sampling_method']).agg({
            'aupr_improvement_factor': ['mean', 'max'],
            'test_auc': ['mean', 'max']
        }).round(4)
        combined_stats.columns = ['_'.join(col).strip() for col in combined_stats.columns.values]
        combined_stats.to_excel(writer, sheet_name='Combined_Statistics')
        
        topology_data = []
        for result in all_results:
            if 'topology_analysis' in result and result['neg_ratio'] is not None:
                topo = result['topology_analysis']
                topology_data.append({
                    'neg_ratio': result['neg_ratio'],
                    'sampling_method': result['sampling_method'],
                    'spectral_gap_preservation': topo.get('spectral_gap_preservation'),
                    'hub_node_preservation': topo.get('hub_node_preservation'),
                    'original_graph_structure_preservation': topo.get('original_graph_structure_preservation'),
                    'sampling_quality_index': topo.get('sampling_quality_index'),
                    'completeness': topo.get('completeness'),
                    'completeness_weight': topo.get('completeness_weight'),
                    'spectral_gap_complete': topo.get('spectral_gap_complete'),
                    'spectral_gap_sampled': topo.get('spectral_gap_sampled')
                })
        
        if topology_data:
            topology_df = pd.DataFrame(topology_data)
            topology_df.to_excel(writer, sheet_name='Topology_Metrics', index=False)
            
            topology_stats = topology_df.groupby('sampling_method').agg({
                'spectral_gap_preservation': ['mean', 'std', 'max', 'min'],
                'hub_node_preservation': ['mean', 'std', 'max', 'min'],
                'original_graph_structure_preservation': ['mean', 'std', 'max', 'min'],
                'sampling_quality_index': ['mean', 'std', 'max', 'min'],
                'completeness': ['mean', 'std', 'max', 'min']
            }).round(4)
            topology_stats.columns = ['_'.join(col).strip() for col in topology_stats.columns.values]
            topology_stats.to_excel(writer, sheet_name='Topology_Statistics')
    
    print(f"Results saved to Excel file: {excel_file}")
    
    return summary_df

def create_visualizations(summary_df, network_info):
    """Create visualizations comparing negative ratio vs AUPR IF, AUC, and AUPR performance."""
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'legend.framealpha': 0.95
    })
    
    color_palette = {
        'preferential_attachment': '#2E86AB',
        'dp_preferential_attachment_scaled': '#0D3A5C',
        'jaccard_coefficient': '#A23B72',
        'dp_jaccard_coefficient': '#7D2E58',
        'yang_zhang': '#F18F01',
        'dp_yang_zhang': '#CC7600'
    }
    
    marker_styles = {
        'preferential_attachment': 'o',
        'dp_preferential_attachment_scaled': '^',
        'jaccard_coefficient': 'D',
        'dp_jaccard_coefficient': 'v',
        'yang_zhang': 'p',
        'dp_yang_zhang': 'h'
    }
    
    line_styles = {
        'preferential_attachment': '-',
        'dp_preferential_attachment_scaled': '-.',
        'jaccard_coefficient': '-',
        'dp_jaccard_coefficient': '--',
        'yang_zhang': '-',
        'dp_yang_zhang': '--'
    }
    
    methods = summary_df['method'].unique()
    
    method_labels = {
        'preferential_attachment': 'Pref. Attachment',
        'dp_preferential_attachment_scaled': 'DP-PA (Scaled)',
        'jaccard_coefficient': 'Jaccard Coefficient',
        'dp_jaccard_coefficient': 'DP-Jaccard Coefficient',
        'yang_zhang': 'Yang-Zhang',
        'dp_yang_zhang': 'DP-Yang-Zhang'
    }
    
    sampling_methods = ['random', 'BRODER', 'HRNE']
    
    for sampling_method in sampling_methods:
        print(f"\nCreating professional visualizations for {sampling_method.upper()} sampling...")
        
        current_data = summary_df[
            (summary_df['sampling_method'] == sampling_method) | 
            (summary_df['sampling_method'] == 'complete')
        ].copy()
        
        if current_data.empty:
            print(f"No data found for {sampling_method} sampling method")
            continue
        
        sampling_suffix = sampling_method.lower()
        
        # ===================== AUPR IMPROVEMENT FACTOR PLOT =====================
        fig1, ax1 = plt.subplots(1, 1, figsize=(16, 10))
        
        plotted_values = {}
        
        for method in methods:
           if method in current_data['method'].values:
               method_data = current_data[current_data['method'] == method].copy()
               
               if method_data.empty:
                   continue
               
               method_data['neg_ratio_numeric'] = method_data['neg_ratio'].apply(
                   lambda x: 30.0 if (pd.isna(x) or x == '' or x == 'None') else float(x)
               )
               
               method_data = method_data.sort_values('neg_ratio_numeric')
               
               color = color_palette.get(method, '#34495e')
               marker = marker_styles.get(method, 'o')
               linestyle = line_styles.get(method, '-')
               label = method_labels.get(method, method.replace('_', ' ').title())
               
               data_tuple = tuple(method_data['aupr_improvement_factor'])
               
               if data_tuple in plotted_values:
                   y_values = method_data['aupr_improvement_factor'] * 0.97
               else:
                   y_values = method_data['aupr_improvement_factor']
                   plotted_values[data_tuple] = method
               
               line = ax1.plot(method_data['neg_ratio_numeric'], 
                              y_values,
                              marker=marker, 
                              color=color, 
                              label=label,
                              linewidth=3.5, 
                              markersize=9,
                              alpha=0.9,
                              linestyle=linestyle,
                              markerfacecolor=color,
                              markeredgecolor='white',
                              markeredgewidth=2,
                              zorder=3)
               
               for _, row in method_data.iterrows():
                   ax1.annotate(f'×{row["aupr_improvement_factor"]:.1f}', 
                              (row['neg_ratio_numeric'], row['aupr_improvement_factor']),
                              xytext=(8, 8), textcoords='offset points', 
                              fontsize=10, fontweight='600', color=color, alpha=0.8,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                      edgecolor=color, alpha=0.7))
        
        ax1.axhline(y=1.0, color='#e74c3c', linestyle=':', alpha=0.8, linewidth=3,
                   label='Random Baseline (×1.0)', zorder=1)
        
        ax1.fill_between([0.5, 35.5], 1, 2, alpha=0.1, color='#f1c40f', label='_nolegend_')
        ax1.fill_between([0.5, 35.5], 2, 5, alpha=0.1, color='#e67e22', label='_nolegend_')
        ax1.fill_between([0.5, 35.5], 5, 100, alpha=0.1, color='#27ae60', label='_nolegend_')
        
        ax1.set_xlabel('Negative Sampling Ratio', fontsize=16, fontweight='700', 
                      color='#2c3e50', labelpad=15)
        ax1.set_ylabel('AUPR Improvement Factor', fontsize=16, fontweight='700', 
                      color='#2c3e50', labelpad=15)
        ax1.set_title(f'Link Prediction Performance Analysis\n{sampling_method.upper()} Sampling Strategy', 
                     fontsize=20, fontweight='700', color='#2c3e50', pad=25)
        
        ax1.set_yscale('log')
        ax1.set_ylim(0.5, None)
        
        x_ticks = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
        x_labels = ['1:1', '2:1', '5:1', '10:1', '20:1', 'Complete']
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels, fontsize=13, fontweight='600')
        
        ax1.set_xlim(0.8, 35.0)
        
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=1, zorder=0)
        ax1.set_axisbelow(True)
        
        legend = ax1.legend(loc='upper right', fontsize=12, framealpha=0.95, 
                           fancybox=True, shadow=True, ncol=2, columnspacing=1.5)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#dee2e6')
        
        ax1.tick_params(axis='both', which='major', labelsize=12, width=1.5)
        ax1.tick_params(axis='both', which='minor', width=1)
        
        fig1.tight_layout()
        plot_file1 = os.path.join(RESULTS_DIR, f"sbm_s166_167_167_pi0.0250_pe0.00250_aupr_if_predictions_{sampling_suffix}.png")
        fig1.savefig(plot_file1, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.3)
        
        # ===================== AUC PLOT =====================
        fig2, ax2 = plt.subplots(1, 1, figsize=(16, 10))
        
        for method in methods:
            if method in current_data['method'].values:
                method_data = current_data[current_data['method'] == method].copy()
                
                if method_data.empty:
                    continue
                
                method_data['neg_ratio_numeric'] = method_data['neg_ratio'].apply(
                    lambda x: 30.0 if (pd.isna(x) or x == '' or x == 'None') else float(x)
                )
                
                method_data = method_data.sort_values('neg_ratio_numeric')
                
                color = color_palette.get(method, '#34495e')
                marker = marker_styles.get(method, 'o')
                linestyle = line_styles.get(method, '-')
                label = method_labels.get(method, method.replace('_', ' ').title())
                
                ax2.plot(method_data['neg_ratio_numeric'], 
                        method_data['test_auc'], 
                        marker=marker, 
                        color=color, 
                        label=label,
                        linewidth=3.5, 
                        markersize=9,
                        alpha=0.9,
                        linestyle=linestyle,
                        markerfacecolor=color,
                        markeredgecolor='white',
                        markeredgewidth=2,
                        zorder=3)
                
                for _, row in method_data.iterrows():
                    ax2.annotate(f'{row["test_auc"]:.3f}', 
                               (row['neg_ratio_numeric'], row['test_auc']),
                               xytext=(8, 8), textcoords='offset points', 
                               fontsize=10, fontweight='600', color=color, alpha=0.8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor=color, alpha=0.7))
        
        ax2.axhline(y=0.5, color='#e74c3c', linestyle=':', alpha=0.8, linewidth=3,
                   label='Random Baseline (0.5)', zorder=1)
        
        ax2.fill_between([0.5, 35.5], 0.5, 0.7, alpha=0.08, color='#f1c40f', label='_nolegend_')
        ax2.fill_between([0.5, 35.5], 0.7, 0.85, alpha=0.08, color='#e67e22', label='_nolegend_')
        ax2.fill_between([0.5, 35.5], 0.85, 1.0, alpha=0.08, color='#27ae60', label='_nolegend_')
        
        ax2.set_xlabel('Negative Sampling Ratio', fontsize=16, fontweight='700', 
                      color='#2c3e50', labelpad=15)
        ax2.set_ylabel('AUC Score', fontsize=16, fontweight='700', 
                      color='#2c3e50', labelpad=15)
        ax2.set_title(f'AUC Performance Analysis\n{sampling_method.upper()} Sampling Strategy', 
                     fontsize=20, fontweight='700', color='#2c3e50', pad=25)
        
        x_ticks = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
        x_labels = ['1:1', '2:1', '5:1', '10:1', '20:1', 'Complete']
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, fontsize=13, fontweight='600')
        
        ax2.set_xlim(0.8, 35.0)
        ax2.set_ylim(0.1, 1.0)
        
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=1, zorder=0)
        ax2.set_axisbelow(True)
        
        legend = ax2.legend(loc='upper right', fontsize=12, framealpha=0.95, 
                           fancybox=True, shadow=True, ncol=2, columnspacing=1.5)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#dee2e6')
        
        ax2.tick_params(axis='both', which='major', labelsize=12, width=1.5)
        
        fig2.tight_layout()
        plot_file2 = os.path.join(RESULTS_DIR, f"sbm_s166_167_167_pi0.0250_pe0.00250_auc_predictions_{sampling_suffix}.png")
        fig2.savefig(plot_file2, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.3)
        
        # ===================== AUPR PLOT =====================
        fig3, ax3 = plt.subplots(1, 1, figsize=(16, 10))
        
        for method in methods:
            if method in current_data['method'].values:
                method_data = current_data[current_data['method'] == method].copy()
                
                if method_data.empty:
                    continue
                
                method_data['neg_ratio_numeric'] = method_data['neg_ratio'].apply(
                    lambda x: 30.0 if (pd.isna(x) or x == '' or x == 'None') else float(x)
                )
                
                method_data = method_data.sort_values('neg_ratio_numeric')
                
                color = color_palette.get(method, '#34495e')
                marker = marker_styles.get(method, 'o')
                linestyle = line_styles.get(method, '-')
                label = method_labels.get(method, method.replace('_', ' ').title())
                
                ax3.plot(method_data['neg_ratio_numeric'], 
                        method_data['test_aupr'], 
                        marker=marker, 
                        color=color, 
                        label=label,
                        linewidth=3.5, 
                        markersize=9,
                        alpha=0.9,
                        linestyle=linestyle,
                        markerfacecolor=color,
                        markeredgecolor='white',
                        markeredgewidth=2,
                        zorder=3)
                
                for _, row in method_data.iterrows():
                    ax3.annotate(f'{row["test_aupr"]:.4f}', 
                               (row['neg_ratio_numeric'], row['test_aupr']),
                               xytext=(8, 8), textcoords='offset points', 
                               fontsize=10, fontweight='600', color=color, alpha=0.8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor=color, alpha=0.7))
        
        # Add baseline line for AUPR (network density)
        baseline_aupr = current_data['random_baseline'].iloc[0] if not current_data.empty else 0.016
        ax3.axhline(y=baseline_aupr, color='#e74c3c', linestyle=':', alpha=0.8, linewidth=3,
                   label=f'Random Baseline ({baseline_aupr:.4f})', zorder=1)
        
        # Add performance zones for AUPR
        ax3.fill_between([0.5, 35.5], baseline_aupr, baseline_aupr*2, alpha=0.08, color='#f1c40f', label='_nolegend_')
        ax3.fill_between([0.5, 35.5], baseline_aupr*2, baseline_aupr*5, alpha=0.08, color='#e67e22', label='_nolegend_')
        ax3.fill_between([0.5, 35.5], baseline_aupr*5, 1.0, alpha=0.08, color='#27ae60', label='_nolegend_')
        
        ax3.set_xlabel('Negative Sampling Ratio', fontsize=16, fontweight='700', 
                      color='#2c3e50', labelpad=15)
        ax3.set_ylabel('AUPR Score', fontsize=16, fontweight='700', 
                      color='#2c3e50', labelpad=15)
        ax3.set_title(f'AUPR Performance Analysis\n{sampling_method.upper()} Sampling Strategy', 
                     fontsize=20, fontweight='700', color='#2c3e50', pad=25)
        
        x_ticks = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
        x_labels = ['1:1', '2:1', '5:1', '10:1', '20:1', 'Complete']
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(x_labels, fontsize=13, fontweight='600')
        
        ax3.set_xlim(0.8, 35.0)
        ax3.set_ylim(0, max(current_data['test_aupr'].max() * 1.1, baseline_aupr * 10))
        
        ax3.grid(True, alpha=0.3, linestyle='-', linewidth=1, zorder=0)
        ax3.set_axisbelow(True)
        
        legend = ax3.legend(loc='upper right', fontsize=12, framealpha=0.95, 
                           fancybox=True, shadow=True, ncol=2, columnspacing=1.5)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#dee2e6')
        
        ax3.tick_params(axis='both', which='major', labelsize=12, width=1.5)
        
        fig3.tight_layout()
        plot_file3 = os.path.join(RESULTS_DIR, f"sbm_s166_167_167_pi0.0250_pe0.00250_aupr_predictions_{sampling_suffix}.png")
        fig3.savefig(plot_file3, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.3)
        
        # ===================== AUPR IMPROVEMENT FACTOR TABLE =====================
        fig4 = plt.figure(figsize=(18, 12))
        ax4 = fig4.add_subplot(111)
        ax4.axis('tight')
        ax4.axis('off')
        
        aupr_pivot = current_data.pivot(index='method', columns='neg_ratio', values='aupr_improvement_factor')
        
        column_order = [1.0, 2.0, 5.0, 10.0, 20.0, None]
        aupr_pivot = aupr_pivot.reindex(columns=column_order)
        
        method_order = [
            'preferential_attachment', 'dp_preferential_attachment_scaled',
            'jaccard_coefficient', 'dp_jaccard_coefficient',
            'yang_zhang', 'dp_yang_zhang'
        ]
        
        aupr_pivot = aupr_pivot.reindex([m for m in method_order if m in aupr_pivot.index])
        
        table_data = []
        for method in aupr_pivot.index:
            row = [method_labels.get(method, method)]
            for col in aupr_pivot.columns:
                value = aupr_pivot.loc[method, col]
                if pd.notna(value):
                    if value >= 10:
                        row.append(f'×{value:.0f}')
                    else:
                        row.append(f'×{value:.1f}')
                else:
                    row.append('—')
            table_data.append(row)
        
        col_headers = ['Method', '1:1', '2:1', '5:1', '10:1', '20:1', 'Complete']
        
        table = ax4.table(cellText=table_data,
                         colLabels=col_headers,
                         cellLoc='center',
                         loc='center',
                         bbox=[0.05, 0.15, 0.9, 0.75])
        
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.3, 3)
        
        header_color = '#34495e'
        pa_color = '#ebf3fd'
        jc_color = '#fdeef5'
        yz_color = '#fef9e7'
        
        for i in range(len(col_headers)):
            table[(0, i)].set_facecolor(header_color)
            table[(0, i)].set_text_props(weight='bold', color='white', size=14)
            table[(0, i)].set_height(0.08)
        
        for i, method in enumerate(aupr_pivot.index, 1):
            if 'preferential' in method:
                color = pa_color
            elif 'jaccard' in method:
                color = jc_color
            elif 'yang' in method:
                color = yz_color
            else:
                color = '#f8f9fa'
            
            for j in range(len(col_headers)):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_height(0.07)
                if j == 0:
                    table[(i, j)].set_text_props(weight='bold', size=12, color='#2c3e50')
                else:
                    table[(i, j)].set_text_props(size=12, color='#2c3e50', weight='600')
        
        title_text = f'AUPR Improvement Factor Performance Matrix\n{sampling_method.upper()} Sampling Strategy'
        ax4.text(0.5, 0.95, title_text, transform=ax4.transAxes, fontsize=18, 
                fontweight='700', ha='center', color='#2c3e50')
        
        subtitle_text = 'Higher values indicate better performance relative to random classifier baseline'
        ax4.text(0.5, 0.05, subtitle_text, transform=ax4.transAxes, fontsize=12, 
                ha='center', color='#7f8c8d', style='italic')
        
        fig4.tight_layout()
        plot_file4 = os.path.join(RESULTS_DIR, f"sbm_s166_167_167_pi0.0250_pe0.00250_aupr_if_table_{sampling_suffix}.png")
        fig4.savefig(plot_file4, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.4)
        
        # ===================== AUC TABLE =====================
        fig5 = plt.figure(figsize=(18, 12))
        ax5 = fig5.add_subplot(111)
        ax5.axis('tight')
        ax5.axis('off')
        
        auc_pivot = current_data.pivot(index='method', columns='neg_ratio', values='test_auc')
        auc_pivot = auc_pivot.reindex(columns=column_order)
        auc_pivot = auc_pivot.reindex([m for m in method_order if m in auc_pivot.index])
        
        table_data_auc = []
        for method in auc_pivot.index:
            row = [method_labels.get(method, method)]
            for col in auc_pivot.columns:
                value = auc_pivot.loc[method, col]
                if pd.notna(value):
                    row.append(f'{value:.3f}')
                else:
                    row.append('—')
            table_data_auc.append(row)
        
        table_auc = ax5.table(cellText=table_data_auc,
                             colLabels=col_headers,
                             cellLoc='center',
                             loc='center',
                             bbox=[0.05, 0.15, 0.9, 0.75])
        
        table_auc.auto_set_font_size(False)
        table_auc.set_fontsize(13)
        table_auc.scale(1.3, 3)
        
        for i in range(len(col_headers)):
            table_auc[(0, i)].set_facecolor(header_color)
            table_auc[(0, i)].set_text_props(weight='bold', color='white', size=14)
            table_auc[(0, i)].set_height(0.08)
        
        for i, method in enumerate(auc_pivot.index, 1):
            if 'preferential' in method:
                color = pa_color
            elif 'jaccard' in method:
                color = jc_color
            elif 'yang' in method:
                color = yz_color
            else:
                color = '#f8f9fa'
            
            for j in range(len(col_headers)):
                table_auc[(i, j)].set_facecolor(color)
                table_auc[(i, j)].set_height(0.07)
                if j == 0:
                    table_auc[(i, j)].set_text_props(weight='bold', size=12, color='#2c3e50')
                else:
                    table_auc[(i, j)].set_text_props(size=12, color='#2c3e50', weight='600')
        
        title_text = f'AUC Performance Matrix\n{sampling_method.upper()} Sampling Strategy'
        ax5.text(0.5, 0.95, title_text, transform=ax5.transAxes, fontsize=18, 
                fontweight='700', ha='center', color='#2c3e50')
        
        subtitle_text = 'Area Under ROC Curve scores (0.5 = random, 1.0 = perfect classification)'
        ax5.text(0.5, 0.05, subtitle_text, transform=ax5.transAxes, fontsize=12, 
                ha='center', color='#7f8c8d', style='italic')
        
        fig5.tight_layout()
        plot_file5 = os.path.join(RESULTS_DIR, f"sbm_s166_167_167_pi0.0250_pe0.00250_auc_table_{sampling_suffix}.png")
        fig5.savefig(plot_file5, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.4)
        
        # ===================== AUPR TABLE =====================
        fig6 = plt.figure(figsize=(18, 12))
        ax6 = fig6.add_subplot(111)
        ax6.axis('tight')
        ax6.axis('off')
        
        aupr_table_pivot = current_data.pivot(index='method', columns='neg_ratio', values='test_aupr')
        aupr_table_pivot = aupr_table_pivot.reindex(columns=column_order)
        aupr_table_pivot = aupr_table_pivot.reindex([m for m in method_order if m in aupr_table_pivot.index])
        
        table_data_aupr = []
        for method in aupr_table_pivot.index:
            row = [method_labels.get(method, method)]
            for col in aupr_table_pivot.columns:
                value = aupr_table_pivot.loc[method, col]
                if pd.notna(value):
                    row.append(f'{value:.4f}')
                else:
                    row.append('—')
            table_data_aupr.append(row)
        
        table_aupr = ax6.table(cellText=table_data_aupr,
                              colLabels=col_headers,
                              cellLoc='center',
                              loc='center',
                              bbox=[0.05, 0.15, 0.9, 0.75])
        
        table_aupr.auto_set_font_size(False)
        table_aupr.set_fontsize(13)
        table_aupr.scale(1.3, 3)
        
        for i in range(len(col_headers)):
            table_aupr[(0, i)].set_facecolor(header_color)
            table_aupr[(0, i)].set_text_props(weight='bold', color='white', size=14)
            table_aupr[(0, i)].set_height(0.08)
        
        for i, method in enumerate(aupr_table_pivot.index, 1):
            if 'preferential' in method:
                color = pa_color
            elif 'jaccard' in method:
                color = jc_color
            elif 'yang' in method:
                color = yz_color
            else:
                color = '#f8f9fa'
            
            for j in range(len(col_headers)):
                table_aupr[(i, j)].set_facecolor(color)
                table_aupr[(i, j)].set_height(0.07)
                if j == 0:
                    table_aupr[(i, j)].set_text_props(weight='bold', size=12, color='#2c3e50')
                else:
                    table_aupr[(i, j)].set_text_props(size=12, color='#2c3e50', weight='600')
        
        title_text = f'AUPR Performance Matrix\n{sampling_method.upper()} Sampling Strategy'
        ax6.text(0.5, 0.95, title_text, transform=ax6.transAxes, fontsize=18, 
                fontweight='700', ha='center', color='#2c3e50')
        
        subtitle_text = f'Area Under Precision-Recall Curve scores (baseline: {baseline_aupr:.4f})'
        ax6.text(0.5, 0.05, subtitle_text, transform=ax6.transAxes, fontsize=12, 
                ha='center', color='#7f8c8d', style='italic')
        
        fig6.tight_layout()
        plot_file6 = os.path.join(RESULTS_DIR, f"sbm_s166_167_167_pi0.0250_pe0.00250_aupr_table_{sampling_suffix}.png")
        fig6.savefig(plot_file6, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.4)
        
        # Show all plots
        plt.show(fig1)
        plt.show(fig2)
        plt.show(fig3)
        plt.show(fig4)
        plt.show(fig5)
        plt.show(fig6)
        
        print(f"✓ AUPR Improvement Factor plot: {plot_file1}")
        print(f"✓ AUC performance plot: {plot_file2}")
        print(f"✓ AUPR performance plot: {plot_file3}")
        print(f"✓ AUPR Improvement Factor table: {plot_file4}")
        print(f"✓ AUC performance table: {plot_file5}")
        print(f"✓ AUPR performance table: {plot_file6}")
        
        plt.close('all')        

########################### MAIN EXPERIMENT FUNCTION #######################

def main():
    """Run the complete link prediction experiment."""
    print("LINK PREDICTION EXPERIMENT ON BARABÁSI-ALBERT NETWORK")
    print("="*60)
    
    # Load the network
    G = load_network(NETWORK_PATH)
    
    # Network information
    network_info = {
        'path': NETWORK_PATH,
        'description': f"Stochastic Block Model network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges",
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'type': G.graph.get('network_type', 'stochastic_block_model')
    }
   
    all_results = []
    exp_id = 1
    
    for neg_ratio in NEG_RATIOS:
        if neg_ratio is None:
            # Complete negative sampling 
            try:
                experiment_results = run_experiment_for_neg_ratio(G, neg_ratio, 'complete', exp_id)
                all_results.append(experiment_results)
                exp_id += 1
            except Exception as e:
                print(f"Error in complete sampling experiment {exp_id}: {str(e)}")
                exp_id += 1
                continue
        else:
            for sampling_method in SAMPLING_METHODS:
                try:
                    experiment_results = run_experiment_for_neg_ratio(G, neg_ratio, sampling_method, exp_id)
                    all_results.append(experiment_results)
                    exp_id += 1
                except Exception as e:
                    print(f"Error in experiment {exp_id}: {str(e)}")
                    exp_id += 1
                    continue
    
    # Save and analyze results
    print("\n EXPERIMENT COMPLETED - ANALYZING RESULTS")
    print("="*60)
    
    summary_df = save_experiment_results(all_results, network_info)
    
    # Create the plots
    create_visualizations(summary_df, network_info)
    
    # Create topology analysis plots
    create_topology_preservation_histogram(all_results, RESULTS_DIR)
    
    # Print best performing combinations
    best_aupr = summary_df.loc[summary_df['aupr_improvement_factor'].idxmax()]
    
    print(f"\nBest AUPR Improvement: x{best_aupr['aupr_improvement_factor']:.1f}")
    print(f"  Method: {best_aupr['method']}")
    print(f"  Negative Ratio: {best_aupr['neg_ratio']}")
    print(f"  Sampling Method: {best_aupr['sampling_method']}")
    print(f"  AUC: {best_aupr['test_auc']:.4f} | AUPR: {best_aupr['test_aupr']:.4f}")
    
    print(f"\nResults saved in '{RESULTS_DIR}' directory")

if __name__ == "__main__":
    main()