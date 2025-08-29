# Impact of Sampling and Balancing Techniques in Link Prediction

This repository contains the complete implementation and experimental framework for the Master's thesis "Impact of Sampling and Balancing Techniques in Link Prediction" by Andrés Tejero Gómez (Universidad Autónoma de Madrid, 2024-2025).

## Abstract

This research addresses the severe class imbalance problem in sparse network link prediction through systematic negative edge undersampling. The methodology preserves original network topology while achieving substantial performance improvements across diverse synthetic and real-world networks.

**Key findings:**
- AUPR scores improved from ~0.01 (complete sampling) to 0.6-0.7 (1:1 balanced ratio)
- AUC performance remained stable across all sampling ratios, confirming no loss of discriminative capacity
- Results consistent across 5 synthetic network topologies and validated on Dutch corporate board networks

## Repository Structure

### Core Implementation
- **`gen_networks.py`** - Synthetic network generation (15 networks across 5 topologies)
- **`gen_real_world_networks.py`** - Real-world network processing (Dutch corporate data)
- **`general_analysis.py`** - Cross-network aggregated analysis and visualization

### Network-Specific Experiments
Individual experiment files for each synthetic network:
- **Barabási-Albert**: `ba_n500_m1.py`, `ba_n500_m2.py`, `ba_n500_m4.py`
- **Erdős-Rényi**: `er_n500_p0.005.py`, `er_n500_p0.010.py`, `er_n500_p0.015.py`
- **Watts-Strogatz**: `ws_n500_k2_p0.100.py`, `ws_n500_k6_p0.100.py`, `ws_n500_k8_p0.100.py`
- **Powerlaw Cluster**: `pc_n500_m1_p0.200.py`, `pc_n500_m2_p0.200.py`, `pc_n500_m4_p0.200.py`
- **Stochastic Block Model**: `sbm_s166_167_167_*.py` variants

### Real-World Applications
- **Directors Networks**: `directors_1976.py`, `directors_1996.py`, `directors_2001.py`

## Methodology

### Network Generation
- **Synthetic Networks**: 15 networks (5 topologies × 3 density levels: 0.5%, 1.0%, 1.5%)
- **Topologies**: Barabási-Albert, Erdős-Rényi, Watts-Strogatz, Powerlaw Cluster, Stochastic Block Model
- **Real Networks**: Dutch corporate board interlocks (1976, 1996, 2001)

### Negative Sampling Techniques
1. **Random Sampling** - Baseline approach using NetworkX
2. **BRODER Sampling** - Multiple spanning trees (adapted from EvalNE)
3. **HRNE Sampling** - Hybrid node-edge approach (littleballoffur)

### Link Prediction Methods
**Base Methods:**
- Yang-Zhang Similarity
- Jaccard Coefficient  
- Preferential Attachment

**Enhanced Methods (with Degree of Popularity):**
- DP-Yang-Zhang Similarity
- DP-Jaccard Coefficient
- DP-Preferential Attachment (Scaled)

### Evaluation Metrics
- **AUPR** (Area Under Precision-Recall Curve) - Primary metric
- **AUPR Improvement Factor** - Performance vs. random classifier
- **AUC** (Area Under ROC Curve) - Secondary comparison metric
- **Topology Preservation Analysis** - 4 custom metrics for sampling quality

## Key Features

### Class Imbalance Solution
- **Negative Ratio Testing**: Complete, 20:1, 10:1, 5:1, 2:1, 1:1
- **Topology Preservation**: Original network structure remains intact
- **Computational Efficiency**: Aggressive undersampling reduces complexity

### Experimental Framework
- **1,440 unique combinations** tested across all configurations
- **Reproducible results** with fixed random seeds
- **Comprehensive evaluation** across multiple sampling strategies

### Advanced Analysis
- **Topology Preservation Metrics**: Spectral gap, hub preservation, structure preservation, sampling quality
- **Cross-Network Aggregation**: General patterns across network types
- **Real-World Validation**: Corporate social network analysis

## Results Summary

### Performance Improvements
- **AUPR**: 60x improvement (0.01 → 0.6) with balanced sampling
- **Consistent Patterns**: Exponential improvement across all network types
- **Method Robustness**: All link prediction methods benefit equally

### Sampling Strategy Analysis
- **No significant differences** between sampling methods under extreme undersampling
- **Random sampling** performs comparably to sophisticated approaches
- **Computational simplicity** preferred given equivalent performance

### Real-World Validation
- **Dutch corporate networks** confirm synthetic results
- **Social network dynamics** preserved across different time periods
- **Practical applicability** demonstrated in complex organizational structures

## Requirements

### Core Dependencies
```python
networkx>=2.8
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
openpyxl>=3.1.0
