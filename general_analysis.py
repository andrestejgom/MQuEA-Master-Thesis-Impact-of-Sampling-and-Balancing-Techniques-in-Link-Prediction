# -*- coding: utf-8 -*-
"""
Network Results Aggregator for Link Prediction Research.
Aggregates results from all synthetic networks and shows general mean performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = "general_results"

def find_all_result_files(base_directory="predictions"):
    """Find all Excel result files in predictions folder."""
    print("Searching for result files in 'predictions' folder...")
    
    if not os.path.exists(base_directory):
        print(f"Error: Folder '{base_directory}' does not exist.")
        return []
    
    result_files = []
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('_results.xlsx'):
                result_files.append(os.path.join(root, file))
    
    result_files = sorted(result_files)
    print(f"Found {len(result_files)} result files")
    return result_files

def load_and_aggregate_results(result_files):
    """Load and aggregate results from all Excel files."""
    print("Loading and aggregating results...")
    
    all_results = []
    
    for file_path in result_files:
        try:
            results_df = pd.read_excel(file_path, sheet_name='Results')
            results_df['source_file'] = os.path.basename(file_path)
            
            path_parts = file_path.replace('\\', '/').split('/')
            folder_name = None
            
            for part in path_parts:
                if '_predictions' in part:
                    folder_name = part.replace('_predictions', '')
                    break
            
            if folder_name:
                results_df['network_instance'] = folder_name
                
                if folder_name.startswith('ba_'):
                    network_type = 'barabasi_albert'
                elif folder_name.startswith('er_'):
                    network_type = 'erdos_renyi'
                elif folder_name.startswith('ws_'):
                    network_type = 'watts_strogatz'
                elif folder_name.startswith('pc_'):
                    network_type = 'powerlaw_cluster'
                elif folder_name.startswith('sbm_'):
                    network_type = 'stochastic_block_model'
                else:
                    network_type = 'unknown'
            else:
                network_type = 'unknown'
                results_df['network_instance'] = 'unknown'
            
            results_df['network_type'] = network_type
            all_results.append(results_df)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not all_results:
        raise ValueError("Could not load results from any file")
    
    general_results = pd.concat(all_results, ignore_index=True)
    
    print(f"Aggregated results:")
    print(f"  Total experiments: {len(general_results)}")
    print(f"  Unique networks: {general_results['network_instance'].nunique()}")
    print(f"  Network types: {', '.join(general_results['network_type'].unique())}")
    
    return general_results

def calculate_general_mean_performance(aggregated_df):
    """Calculate general mean performance across all networks and sampling methods."""
    print("Calculating general mean performance...")
    
    # Convert None values to 'complete' string for proper groupby handling
    aggregated_df_fixed = aggregated_df.copy()
    aggregated_df_fixed['neg_ratio_fixed'] = aggregated_df_fixed['neg_ratio'].fillna('complete')
    
    # Group by method and neg_ratio_fixed only (no sampling_method distinction)
    mean_performance = aggregated_df_fixed.groupby(['method', 'neg_ratio_fixed']).agg({
        'aupr_improvement_factor': ['mean', 'std', 'count'],
        'test_auc': ['mean', 'std'],
        'test_aupr': ['mean', 'std'],
        'random_baseline': ['mean'],
        'original_density': ['mean']
    }).round(4)
    
    mean_performance.columns = ['_'.join(col).strip() for col in mean_performance.columns.values]
    mean_performance = mean_performance.reset_index()
    
    # Convert back 'complete' to None for consistency with visualization code
    mean_performance['neg_ratio'] = mean_performance['neg_ratio_fixed'].apply(
        lambda x: None if x == 'complete' else float(x)
    )
    mean_performance = mean_performance.drop('neg_ratio_fixed', axis=1)
    
    mean_performance = mean_performance.rename(columns={
        'aupr_improvement_factor_mean': 'aupr_improvement_factor',
        'test_auc_mean': 'test_auc',
        'test_aupr_mean': 'test_aupr',
        'random_baseline_mean': 'random_baseline',
        'original_density_mean': 'original_density',
        'aupr_improvement_factor_count': 'experiment_count'
    })
    
    # Add general sampling method label
    mean_performance['sampling_method'] = 'general'
    
    print(f"Calculated general means for {len(mean_performance)} unique combinations")
    print(f"Average experiments per combination: {mean_performance['experiment_count'].mean():.1f}")
    
    # Debug: Check if complete sampling is included
    complete_data = mean_performance[mean_performance['neg_ratio'].isna()]
    if not complete_data.empty:
        print(f"Found {len(complete_data)} complete sampling combinations")
    else:
        print("WARNING: No complete sampling data found after processing")
    
    return mean_performance

def load_and_aggregate_topology_data(result_files):
    """Load and aggregate topology data from Excel files."""
    print("Loading topology data from Excel files...")
    
    all_topology_data = []
    
    for file_path in result_files:
        try:
            path_parts = file_path.replace('\\', '/').split('/')
            folder_name = None
            for part in path_parts:
                if '_predictions' in part:
                    folder_name = part.replace('_predictions', '')
                    break
            
            if not folder_name:
                continue
                
            results_df = pd.read_excel(file_path, sheet_name='Results')
            
            topology_columns = [
                'spectral_gap_preservation',
                'hub_node_preservation', 
                'original_graph_structure_preservation',
                'sampling_quality_index',
                'completeness',
                'completeness_weight'
            ]
            
            has_topology_data = all(col in results_df.columns for col in topology_columns)
            
            if has_topology_data:
                for _, row in results_df.iterrows():
                    neg_ratio = row['neg_ratio']
                    sampling_method = row['sampling_method']
                    method = row['method']
                    
                    if pd.isna(neg_ratio) or neg_ratio == "None" or sampling_method == 'complete':
                        continue
                    
                    if pd.isna(row['spectral_gap_preservation']):
                        continue
                    
                    topology_record = {
                        'network_instance': folder_name,
                        'method': method,
                        'neg_ratio': float(neg_ratio),
                        'sampling_method': sampling_method,
                        'spectral_gap_preservation': row['spectral_gap_preservation'],
                        'hub_node_preservation': row['hub_node_preservation'],
                        'original_graph_structure_preservation': row['original_graph_structure_preservation'],
                        'sampling_quality_index': row['sampling_quality_index'],
                        'completeness': row['completeness'],
                        'completeness_weight': row['completeness_weight']
                    }
                    
                    all_topology_data.append(topology_record)
                    
        except Exception as e:
            continue
    
    if not all_topology_data:
        print("No topology data found")
        return pd.DataFrame()
    
    topology_df = pd.DataFrame(all_topology_data)
    print(f"Loaded topology data for {len(topology_df)} experiments")
    
    return topology_df

def calculate_mean_topology_performance(topology_df):
    """Calculate mean topology performance by sampling method."""
    print("Calculating mean topology performance...")
    
    if topology_df.empty:
        return pd.DataFrame()
    
    mean_topology = topology_df.groupby(['sampling_method', 'neg_ratio']).agg({
        'spectral_gap_preservation': ['mean', 'std', 'count'],
        'hub_node_preservation': ['mean', 'std'],
        'original_graph_structure_preservation': ['mean', 'std'],
        'sampling_quality_index': ['mean', 'std'],
        'completeness': ['mean', 'std'],
        'completeness_weight': ['mean', 'std'],
        'network_instance': 'nunique'
    }).round(4)
    
    mean_topology.columns = ['_'.join(col).strip() for col in mean_topology.columns.values]
    mean_topology = mean_topology.reset_index()
    
    mean_topology = mean_topology.rename(columns={
        'spectral_gap_preservation_mean': 'spectral_gap_preservation',
        'hub_node_preservation_mean': 'hub_node_preservation',
        'original_graph_structure_preservation_mean': 'original_graph_structure_preservation',
        'sampling_quality_index_mean': 'sampling_quality_index',
        'completeness_mean': 'completeness',
        'completeness_weight_mean': 'completeness_weight',
        'spectral_gap_preservation_count': 'experiment_count',
        'network_instance_nunique': 'network_count'
    })
    
    print(f"Calculated topology means for {len(mean_topology)} combinations")
    
    return mean_topology

def create_general_visualizations(summary_df, network_info):
    """Create general visualizations with mean performance across all networks."""
    
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
    
    print("Creating general visualizations...")
    
    # ===================== AUPR IMPROVEMENT FACTOR PLOT =====================
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 10))
    
    for method in methods:
        if method in summary_df['method'].values:
            method_data = summary_df[summary_df['method'] == method].copy()
            
            if method_data.empty:
                continue
            
            method_data['neg_ratio_numeric'] = method_data['neg_ratio'].apply(
                lambda x: 30.0 if pd.isna(x) else float(x)
            )
            
            method_data = method_data.sort_values('neg_ratio_numeric')
            
            color = color_palette.get(method, '#34495e')
            marker = marker_styles.get(method, 'o')
            linestyle = line_styles.get(method, '-')
            label = method_labels.get(method, method.replace('_', ' ').title())
            
            ax1.plot(method_data['neg_ratio_numeric'], 
                    method_data['aupr_improvement_factor'],
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
    
    ax1.fill_between([0.5, 35.5], 1, 2, alpha=0.1, color='#f1c40f')
    ax1.fill_between([0.5, 35.5], 2, 5, alpha=0.1, color='#e67e22')
    ax1.fill_between([0.5, 35.5], 5, 100, alpha=0.1, color='#27ae60')
    
    ax1.set_xlabel('Negative Sampling Ratio', fontsize=16, fontweight='700', 
                  color='#2c3e50', labelpad=15)
    ax1.set_ylabel('AUPR Improvement Factor', fontsize=16, fontweight='700', 
                  color='#2c3e50', labelpad=15)
    ax1.set_title('General Link Prediction Performance Analysis\nAverage Across All Networks and Sampling Methods', 
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
    plot_file1 = os.path.join(OUTPUT_DIR, "general_aupr_if_predictions.png")
    fig1.savefig(plot_file1, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.3)
    
    # ===================== AUC PLOT =====================
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 10))
    
    for method in methods:
        if method in summary_df['method'].values:
            method_data = summary_df[summary_df['method'] == method].copy()
            
            if method_data.empty:
                continue
            
            method_data['neg_ratio_numeric'] = method_data['neg_ratio'].apply(
                lambda x: 30.0 if pd.isna(x) else float(x)
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
    
    ax2.fill_between([0.5, 35.5], 0.5, 0.7, alpha=0.08, color='#f1c40f')
    ax2.fill_between([0.5, 35.5], 0.7, 0.85, alpha=0.08, color='#e67e22')
    ax2.fill_between([0.5, 35.5], 0.85, 1.0, alpha=0.08, color='#27ae60')
    
    ax2.set_xlabel('Negative Sampling Ratio', fontsize=16, fontweight='700', 
                  color='#2c3e50', labelpad=15)
    ax2.set_ylabel('AUC Score', fontsize=16, fontweight='700', 
                  color='#2c3e50', labelpad=15)
    ax2.set_title('General AUC Performance Analysis\nAverage Across All Networks and Sampling Methods', 
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
    plot_file2 = os.path.join(OUTPUT_DIR, "general_auc_predictions.png")
    fig2.savefig(plot_file2, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.3)
    
    # ===================== AUPR PLOT =====================
    fig3, ax3 = plt.subplots(1, 1, figsize=(16, 10))
    
    for method in methods:
        if method in summary_df['method'].values:
            method_data = summary_df[summary_df['method'] == method].copy()
            
            if method_data.empty:
                continue
            
            method_data['neg_ratio_numeric'] = method_data['neg_ratio'].apply(
                lambda x: 30.0 if pd.isna(x) else float(x)
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
    
    # Add baseline line for AUPR (average network density)
    baseline_aupr = summary_df['random_baseline'].mean() if not summary_df.empty else 0.01
    ax3.axhline(y=baseline_aupr, color='#e74c3c', linestyle=':', alpha=0.8, linewidth=3,
               label=f'Random Baseline ({baseline_aupr:.4f})', zorder=1)
    
    # Add performance zones for AUPR
    ax3.fill_between([0.5, 35.5], baseline_aupr, baseline_aupr*2, alpha=0.08, color='#f1c40f')
    ax3.fill_between([0.5, 35.5], baseline_aupr*2, baseline_aupr*5, alpha=0.08, color='#e67e22')
    ax3.fill_between([0.5, 35.5], baseline_aupr*5, 1.0, alpha=0.08, color='#27ae60')
    
    ax3.set_xlabel('Negative Sampling Ratio', fontsize=16, fontweight='700', 
                  color='#2c3e50', labelpad=15)
    ax3.set_ylabel('AUPR Score', fontsize=16, fontweight='700', 
                  color='#2c3e50', labelpad=15)
    ax3.set_title('General AUPR Performance Analysis\nAverage Across All Networks and Sampling Methods', 
                 fontsize=20, fontweight='700', color='#2c3e50', pad=25)
    
    x_ticks = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    x_labels = ['1:1', '2:1', '5:1', '10:1', '20:1', 'Complete']
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_labels, fontsize=13, fontweight='600')
    
    ax3.set_xlim(0.8, 35.0)
    ax3.set_ylim(0, max(summary_df['test_aupr'].max() * 1.1, baseline_aupr * 10))
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=1, zorder=0)
    ax3.set_axisbelow(True)
    
    legend = ax3.legend(loc='upper right', fontsize=12, framealpha=0.95, 
                       fancybox=True, shadow=True, ncol=2, columnspacing=1.5)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    
    ax3.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    
    fig3.tight_layout()
    plot_file3 = os.path.join(OUTPUT_DIR, "general_aupr_predictions.png")
    fig3.savefig(plot_file3, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.3)
    
    # ===================== AUPR IMPROVEMENT FACTOR TABLE =====================
    fig4 = plt.figure(figsize=(18, 12))
    ax4 = fig4.add_subplot(111)
    ax4.axis('tight')
    ax4.axis('off')
    
    # Debug: Check what neg_ratio values we have
    print(f"Available neg_ratio values: {sorted([x for x in summary_df['neg_ratio'].unique() if pd.notna(x)])}")
    print(f"Has None values: {summary_df['neg_ratio'].isna().any()}")
    
    aupr_pivot = summary_df.pivot(index='method', columns='neg_ratio', values='aupr_improvement_factor')
    
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
    
    title_text = 'General AUPR Improvement Factor Performance Matrix\nAverage Across All Networks and Sampling Methods'
    ax4.text(0.5, 0.95, title_text, transform=ax4.transAxes, fontsize=18, 
            fontweight='700', ha='center', color='#2c3e50')
    
    subtitle_text = 'Average performance across all synthetic networks (higher values = better performance)'
    ax4.text(0.5, 0.05, subtitle_text, transform=ax4.transAxes, fontsize=12, 
            ha='center', color='#7f8c8d', style='italic')
    
    fig4.tight_layout()
    plot_file4 = os.path.join(OUTPUT_DIR, "general_aupr_if_table.png")
    fig4.savefig(plot_file4, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.4)
    
    # ===================== AUC TABLE =====================
    fig5 = plt.figure(figsize=(18, 12))
    ax5 = fig5.add_subplot(111)
    ax5.axis('tight')
    ax5.axis('off')
    
    auc_pivot = summary_df.pivot(index='method', columns='neg_ratio', values='test_auc')
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
    
    title_text = 'General AUC Performance Matrix\nAverage Across All Networks and Sampling Methods'
    ax5.text(0.5, 0.95, title_text, transform=ax5.transAxes, fontsize=18, 
            fontweight='700', ha='center', color='#2c3e50')
    
    subtitle_text = 'Average AUC scores across all synthetic networks (0.5 = random, 1.0 = perfect)'
    ax5.text(0.5, 0.05, subtitle_text, transform=ax5.transAxes, fontsize=12, 
            ha='center', color='#7f8c8d', style='italic')
    
    fig5.tight_layout()
    plot_file5 = os.path.join(OUTPUT_DIR, "general_auc_table.png")
    fig5.savefig(plot_file5, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.4)
    
    # ===================== AUPR TABLE =====================
    fig6 = plt.figure(figsize=(18, 12))
    ax6 = fig6.add_subplot(111)
    ax6.axis('tight')
    ax6.axis('off')
    
    aupr_table_pivot = summary_df.pivot(index='method', columns='neg_ratio', values='test_aupr')
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
    
    title_text = 'General AUPR Performance Matrix\nAverage Across All Networks and Sampling Methods'
    ax6.text(0.5, 0.95, title_text, transform=ax6.transAxes, fontsize=18, 
            fontweight='700', ha='center', color='#2c3e50')
    
    subtitle_text = f'Average AUPR scores across all synthetic networks (baseline: {baseline_aupr:.4f})'
    ax6.text(0.5, 0.05, subtitle_text, transform=ax6.transAxes, fontsize=12, 
            ha='center', color='#7f8c8d', style='italic')
    
    fig6.tight_layout()
    plot_file6 = os.path.join(OUTPUT_DIR, "general_aupr_table.png")
    fig6.savefig(plot_file6, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.4)
    
    plt.show(fig1)
    plt.show(fig2)
    plt.show(fig3)
    plt.show(fig4)
    plt.show(fig5)
    plt.show(fig6)
    
    print(f"✓ General AUPR Improvement Factor plot: {plot_file1}")
    print(f"✓ General AUC plot: {plot_file2}")
    print(f"✓ General AUPR plot: {plot_file3}")
    print(f"✓ General AUPR Improvement Factor table: {plot_file4}")
    print(f"✓ General AUC table: {plot_file5}")
    print(f"✓ General AUPR table: {plot_file6}")
    
    plt.close('all')

def create_topology_preservation_histogram(all_results, save_path=None):
    """Create topology histogram distinguishing between sampling methods."""
    
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
        
        for i, method in enumerate(sampling_methods):
            method_data = df[df['sampling_method'] == method]
            
            scores = []
            for neg_ratio in unique_neg_ratios:
                method_ratio_data = method_data[method_data['neg_ratio'] == neg_ratio]
                if not method_ratio_data.empty:
                    scores.append(method_ratio_data[metric_key].mean())
                else:
                    scores.append(0)
            
            bars = ax.bar(x + i * width - width, scores, width,
                         label=method.upper(), 
                         color=colors[method], 
                         alpha=0.85,
                         edgecolor='white',
                         linewidth=1.5,
                         zorder=3)
            
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
    
    fig.suptitle('Average Topology Preservation Analysis: Structured vs Random Sampling', 
                fontsize=20, fontweight='700', color='#2c3e50', y=0.95)
    
    subtitle_text = ('Average performance across all synthetic networks with completeness weighting\n'
                    'Higher scores indicate better preservation of original network topology')
    fig.text(0.5, 0.02, subtitle_text, fontsize=12, ha='center', color='#7f8c8d',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8, edgecolor='none'))
    
    if save_path:
        plot_path = os.path.join(save_path, "general_topology_histogram.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        print(f"Topology histogram saved to: {plot_path}")
    
    plt.show()
    
    print("\n=== AVERAGE TOPOLOGY PRESERVATION ANALYSIS ===")
    for metric_key, metric_title in metrics:
        print(f"\n{metric_title.upper()}:")
        summary_stats = df.groupby('sampling_method')[metric_key].agg(['mean', 'std', 'min', 'max']).round(3)
        print(summary_stats)
    
    return df

def create_topology_results_for_histogram(mean_topology_df):
    """Convert mean topology DataFrame to format expected by histogram function."""
    
    if mean_topology_df.empty:
        print("No topology data available for histogram")
        return []
    
    results = []
    for _, row in mean_topology_df.iterrows():
        result = {
            'neg_ratio': row['neg_ratio'],
            'sampling_method': row['sampling_method'],
            'topology_analysis': {
                'spectral_gap_preservation': row['spectral_gap_preservation'],
                'hub_node_preservation': row['hub_node_preservation'],
                'original_graph_structure_preservation': row['original_graph_structure_preservation'],
                'sampling_quality_index': row['sampling_quality_index']
            }
        }
        results.append(result)
    
    print(f"Created {len(results)} topology result records for histogram")
    return results

def save_general_results(mean_df, aggregated_df, output_dir=OUTPUT_DIR):
    """Save aggregated results to Excel file."""
    Path(output_dir).mkdir(exist_ok=True)
    
    excel_file = os.path.join(output_dir, "general_all_networks_results.xlsx")
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        mean_df.to_excel(writer, sheet_name='General_Results', index=False)
        aggregated_df.to_excel(writer, sheet_name='All_Individual_Results', index=False)
        
        best_avg = mean_df.nlargest(20, 'aupr_improvement_factor')
        best_avg.to_excel(writer, sheet_name='Top_20_General', index=False)
        
        network_summary = aggregated_df.groupby('network_type').agg({
            'aupr_improvement_factor': ['mean', 'std', 'count'],
            'test_auc': ['mean', 'std'],
            'network_instance': 'nunique'
        }).round(4)
        network_summary.columns = ['_'.join(col).strip() for col in network_summary.columns.values]
        network_summary.to_excel(writer, sheet_name='Network_Type_Summary')
    
    print(f"General results saved to: {excel_file}")
    return excel_file

def main():
    """Main function to execute general aggregated analysis."""
    print("=== GENERAL LINK PREDICTION ANALYSIS ===")
    print("=" * 50)
    
    result_files = find_all_result_files("predictions")
    
    if not result_files:
        print("No result files found. Check 'predictions' folder.")
        return
    
    aggregated_df = load_and_aggregate_results(result_files)
    mean_df = calculate_general_mean_performance(aggregated_df)
    
    topology_df = load_and_aggregate_topology_data(result_files)
    mean_topology_df = calculate_mean_topology_performance(topology_df)
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    network_info = {
        'description': f"General analysis of {aggregated_df['network_instance'].nunique()} synthetic networks",
        'total_networks': aggregated_df['network_instance'].nunique(),
        'network_types': list(aggregated_df['network_type'].unique()),
        'total_experiments': len(aggregated_df)
    }
    
    print("\nCreating general performance visualizations...")
    create_general_visualizations(mean_df, network_info)
    
    if not mean_topology_df.empty:
        print("\nCreating topology histogram...")
        topology_results = create_topology_results_for_histogram(mean_topology_df)
        create_topology_preservation_histogram(topology_results, OUTPUT_DIR)
        
        topology_excel = os.path.join(OUTPUT_DIR, "general_topology_results.xlsx")
        with pd.ExcelWriter(topology_excel, engine='openpyxl') as writer:
            mean_topology_df.to_excel(writer, sheet_name='Mean_Topology_Results', index=False)
            topology_df.to_excel(writer, sheet_name='All_Topology_Data', index=False)
        print(f"Topology results saved to: {topology_excel}")
    else:
        print("No topology data found, skipping topology analysis")
    
    excel_file = save_general_results(mean_df, aggregated_df)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Files processed: {len(result_files)}")
    print(f"Total experiments: {len(aggregated_df)}")
    print(f"Unique networks: {aggregated_df['network_instance'].nunique()}")
    if not topology_df.empty:
        print(f"Networks with topology data: {topology_df['network_instance'].nunique()}")
        print(f"Total topology experiments: {len(topology_df)}")
    
    best_general = mean_df.loc[mean_df['aupr_improvement_factor'].idxmax()]
    print(f"\nBest general performance:")
    print(f"  Method: {best_general['method']}")
    print(f"  Negative ratio: {best_general['neg_ratio']}")
    print(f"  AUPR Improvement Factor: ×{best_general['aupr_improvement_factor']:.2f}")
    print(f"  AUC average: {best_general['test_auc']:.4f}")
    print(f"  Based on {int(best_general['experiment_count'])} experiments")
    
    print(f"\n=== PERFORMANCE BY NETWORK TYPE ===")
    network_performance = aggregated_df.groupby('network_type')['aupr_improvement_factor'].agg(['mean', 'std', 'count']).round(2)
    for network_type, stats in network_performance.iterrows():
        print(f"  {network_type}: ×{stats['mean']:.1f} ± {stats['std']:.1f} (n={int(stats['count'])})")
    
    print(f"\nResults saved to '{OUTPUT_DIR}' folder")
    print("Visualizations show general mean performance across all networks and sampling methods")

if __name__ == "__main__":
    main()