#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Weighted Metrics Script

This script reads existing precision, recall, and F1 scores from saved_data_v2
and calculates weighted metrics based on support values. Results are saved to saved_data_v3.

The weighting is done according to the following formula:
    weighted_metric = Σ(metric_i * support_i) / Σ(support_i)

Where:
- metric_i is the precision, recall, or F1 score for class i
- support_i is the number of samples for class i
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
                model_output_file = os.path.join(output_dir, f'{model_name}_weighted_metrics.json')
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
            summary_df.to_csv(os.path.join(output_dir, 'weighted_metrics_summary.csv'), index=False)
            
            # Create and save consolidated JSON
            all_weighted_metrics = {row['Model']: {
                'weighted_precision': row['Weighted Precision'],
                'weighted_recall': row['Weighted Recall'],
                'weighted_f1': row['Weighted F1'],
                'total_support': row['Total Support']
            } for row in summary_data}
            
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
    summary_df.to_csv(os.path.join(output_dir, 'weighted_metrics_summary.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    indices = np.arange(len(summary_data))
    
    plt.bar(indices, [d['Weighted Precision'] for d in summary_data], 
            bar_width, label='Precision', color='blue', alpha=0.7)
    plt.bar(indices + bar_width, [d['Weighted Recall'] for d in summary_data], 
            bar_width, label='Recall', color='green', alpha=0.7)
    plt.bar(indices + 2*bar_width, [d['Weighted F1'] for d in summary_data], 
            bar_width, label='F1', color='red', alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Weighted Metrics Comparison')
    plt.xticks(indices + bar_width, [d['Model'] for d in summary_data])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'weighted_metrics_comparison.png'))
    print(f'Weighted metrics visualization saved to {os.path.join(output_dir, "weighted_metrics_comparison.png")}')

def main():
    # Define directories
    input_dir = 'models/pretrained/saved_data_v2'
    output_dir = 'models/pretrained/saved_data_v3'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Calculating weighted metrics from {input_dir}')
    print(f'Saving results to {output_dir}')
    print('-' * 50)
    
    # Analyze all models
    analyze_all_models(input_dir, output_dir)
    
    print('-' * 50)
    print(f'All weighted metrics saved to {output_dir}')

if __name__ == '__main__':
    main() 