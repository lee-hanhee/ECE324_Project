#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Weighted Metrics Script.

This script reads existing precision, recall, and F1 scores from model metrics
and calculates weighted metrics based on support values. Results are saved to
the specified output directory.

The weighting is done according to the following formula:
    weighted_metric = Σ(metric_i * support_i) / Σ(support_i)

Where:
- metric_i is the precision, recall, or F1 score for class i
- support_i is the number of samples for class i
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics_from_json(json_path):
    """
    Load metrics from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file.
        
    Returns:
        dict: Dictionary containing the metrics.
    """
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def load_metrics_from_csv(csv_path):
    """
    Load metrics from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the metrics.
    """
    metrics_df = pd.read_csv(csv_path)
    return metrics_df


def calculate_weighted_metrics(class_metrics):
    """
    Calculate weighted precision, recall, and F1 scores.
    
    Args:
        class_metrics (dict): Dictionary containing per-class metrics.
        
    Returns:
        dict: Dictionary containing weighted metrics.
    """
    # Extract values from class_metrics
    precision = []
    recall = []
    f1 = []
    support = []
    
    for class_name, metrics in class_metrics.items():
        precision.append(metrics['precision'])
        recall.append(metrics['recall'])
        f1.append(metrics['f1_score'])
        support.append(metrics['support'])
    
    # Convert to numpy arrays
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)
    support = np.array(support)
    
    # Calculate weighted metrics
    total_support = np.sum(support)
    weighted_precision = np.sum(precision * support) / total_support
    weighted_recall = np.sum(recall * support) / total_support
    weighted_f1 = np.sum(f1 * support) / total_support
    
    # Return weighted metrics
    return {
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'total_support': int(total_support)
    }


def analyze_metrics_json(json_file, output_dir):
    """
    Analyze metrics from a JSON file and save weighted metrics.
    
    Args:
        json_file (str): Path to the JSON file.
        output_dir (str): Directory to save the results.
    
    Returns:
        tuple: Model name and weighted metrics dictionary.
    """
    # Load metrics
    model_name = os.path.basename(json_file).replace('_metrics.json', '')
    metrics = load_metrics_from_json(json_file)
    
    # Calculate weighted metrics
    weighted_metrics = calculate_weighted_metrics(metrics)
    
    # Create output file path
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{model_name}_weighted_metrics.json')
    
    # Save weighted metrics
    with open(output_file, 'w') as f:
        json.dump(weighted_metrics, f, indent=4)
    
    print(f'Weighted metrics for {model_name} saved to {output_file}')
    print(f'Weighted precision: {weighted_metrics["weighted_precision"]:.4f}')
    print(f'Weighted recall: {weighted_metrics["weighted_recall"]:.4f}')
    print(f'Weighted F1: {weighted_metrics["weighted_f1"]:.4f}')
    print(f'Total support: {weighted_metrics["total_support"]}')
    print('-' * 50)
    
    return model_name, weighted_metrics


def analyze_all_models(input_dir, output_dir):
    """
    Analyze metrics for all models in the input directory.
    
    Args:
        input_dir (str): Directory containing the metrics files.
        output_dir (str): Directory to save the results.
    """
    # Find all JSON metrics files
    json_files = list(Path(input_dir).glob('*_metrics.json'))
    
    if not json_files:
        consolidated_json_path = os.path.join(input_dir, 'metrics.json')
        if os.path.exists(consolidated_json_path):
            print(f"Using consolidated metrics file: {consolidated_json_path}")
            metrics_data = load_metrics_from_json(consolidated_json_path)
            
            # Create summary dataframe
            summary_data = []
            for model_name, class_metrics in metrics_data.items():
                weighted_metrics = calculate_weighted_metrics(class_metrics)
                
                # Save individual model weighted metrics
                model_output_file = os.path.join(
                    output_dir, f'{model_name}_weighted_metrics.json')
                with open(model_output_file, 'w') as f:
                    json.dump(weighted_metrics, f, indent=4)
                
                # Add to summary data
                summary_data.append({
                    'Model': model_name,
                    'Weighted Precision': weighted_metrics['weighted_precision'],
                    'Weighted Recall': weighted_metrics['weighted_recall'],
                    'Weighted F1': weighted_metrics['weighted_f1'],
                    'Total Support': weighted_metrics['total_support']
                })
                
                print(f'Weighted metrics for {model_name}:')
                print(f'Weighted precision: {weighted_metrics["weighted_precision"]:.4f}')
                print(f'Weighted recall: {weighted_metrics["weighted_recall"]:.4f}')
                print(f'Weighted F1: {weighted_metrics["weighted_f1"]:.4f}')
                print(f'Total support: {weighted_metrics["total_support"]}')
                print('-' * 50)
            
            # Create and save summary dataframe
            summary_df = pd.DataFrame(summary_data)
            summary_csv = os.path.join(output_dir, 'weighted_metrics_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            
            # Create and save consolidated JSON
            all_weighted_metrics = {
                row['Model']: {
                    'weighted_precision': row['Weighted Precision'],
                    'weighted_recall': row['Weighted Recall'],
                    'weighted_f1': row['Weighted F1'],
                    'total_support': row['Total Support']
                } for row in summary_data
            }
            
            with open(os.path.join(output_dir, 'weighted_metrics.json'), 'w') as f:
                json.dump(all_weighted_metrics, f, indent=4)
            
            return
    
    # Process individual metrics files if available
    summary_data = []
    
    for json_file in json_files:
        model_name, weighted_metrics = analyze_metrics_json(json_file, output_dir)
        summary_data.append({
            'Model': model_name,
            'Weighted Precision': weighted_metrics['weighted_precision'],
            'Weighted Recall': weighted_metrics['weighted_recall'],
            'Weighted F1': weighted_metrics['weighted_f1'],
            'Total Support': weighted_metrics['total_support']
        })
    
    # Create and save summary dataframe
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_dir, 'weighted_metrics_summary.csv')
    summary_df.to_csv(summary_csv, index=False)


def visualize_weighted_metrics(weighted_metrics_file, output_dir=None):
    """
    Create a visualization of weighted metrics for comparison.
    
    Args:
        weighted_metrics_file (str): Path to the CSV file with weighted metrics.
        output_dir (str, optional): Directory to save the visualization.
    """
    # Load metrics
    metrics_df = pd.read_csv(weighted_metrics_file)
    
    # Create a bar chart for comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get model names
    models = metrics_df['Model'].values
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.25
    
    # Plot bars for each metric
    ax.bar(x - width, metrics_df['Weighted Precision'], width, 
           label='Weighted Precision', color='#5DA5DA')
    ax.bar(x, metrics_df['Weighted Recall'], width, 
           label='Weighted Recall', color='#FAA43A')
    ax.bar(x + width, metrics_df['Weighted F1'], width, 
           label='Weighted F1', color='#60BD68')
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Weighted Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add value labels on top of bars
    for i, model in enumerate(models):
        ax.text(i - width, metrics_df.loc[metrics_df['Model'] == model, 'Weighted Precision'].values[0] + 0.01,
                f"{metrics_df.loc[metrics_df['Model'] == model, 'Weighted Precision'].values[0]:.3f}",
                ha='center', va='bottom', rotation=0, fontsize=9)
        
        ax.text(i, metrics_df.loc[metrics_df['Model'] == model, 'Weighted Recall'].values[0] + 0.01,
                f"{metrics_df.loc[metrics_df['Model'] == model, 'Weighted Recall'].values[0]:.3f}",
                ha='center', va='bottom', rotation=0, fontsize=9)
        
        ax.text(i + width, metrics_df.loc[metrics_df['Model'] == model, 'Weighted F1'].values[0] + 0.01,
                f"{metrics_df.loc[metrics_df['Model'] == model, 'Weighted F1'].values[0]:.3f}",
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'weighted_metrics_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Visualization saved to {fig_path}')
    
    plt.close()


def main():
    """Run the weighted metrics calculation and visualization."""
    # Define directories
    input_dir = 'models/yamnet/results/metrics'
    output_dir = 'models/yamnet/results/metrics'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze all models
    analyze_all_models(input_dir, output_dir)
    
    # Visualize results
    weighted_metrics_file = os.path.join(output_dir, 'weighted_metrics_summary.csv')
    if os.path.exists(weighted_metrics_file):
        visualize_weighted_metrics(weighted_metrics_file, output_dir)
    else:
        print(f"Warning: Weighted metrics summary file not found at {weighted_metrics_file}")


if __name__ == "__main__":
    main() 