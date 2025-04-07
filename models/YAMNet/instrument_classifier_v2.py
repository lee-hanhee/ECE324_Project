#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instrument Classifier using YAMNet embeddings (Version 2).

This version extends the original instrument classifier to include:
- Precision, recall, and F1 score calculations per instrument class
- Saving detailed metrics in JSON/CSV format
- Reusing existing embeddings and models from saved data to reduce computation

Requirements:
- tensorflow >= 2.5.0
- tensorflow-hub
- librosa
- scikit-learn
- numpy
- pandas
- matplotlib
- tqdm
- json

Installation:
pip install tensorflow tensorflow-hub librosa scikit-learn numpy pandas matplotlib tqdm
"""

import os
import glob
import pickle
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline
import warnings


def load_dataset(data_path, test_size=0.2, random_state=42, save_path=None, 
                original_save_path=None):
    """
    Load audio files and extract labels from filenames.
    
    Args:
        data_path (str): Path to the instruments directory.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        save_path (str, optional): If provided, saves the dataset splits to disk.
        original_save_path (str, optional): Path to check for existing dataset splits.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, label_encoder
    """
    # Check if dataset split already exists in original directory
    if original_save_path is not None:
        original_file = os.path.join(original_save_path, 'dataset_split.pkl')
        if os.path.exists(original_file):
            print(f"Loading existing dataset split from {original_file}")
            with open(original_file, 'rb') as f:
                dataset = pickle.load(f)
                X_train = dataset['X_train']
                X_test = dataset['X_test']
                y_train = dataset['y_train']
                y_test = dataset['y_test']
                label_encoder = dataset['label_encoder']
                
                # Save a copy to the new directory if provided
                if save_path is not None and save_path != original_save_path:
                    os.makedirs(save_path, exist_ok=True)
                    save_file = os.path.join(save_path, 'dataset_split.pkl')
                    with open(save_file, 'wb') as f_new:
                        pickle.dump(dataset, f_new)
                    print(f"Dataset split copied to {save_file}")
                    
                print(f"Training set: {len(X_train)} files, Test set: {len(X_test)} files")
                return X_train, X_test, y_train, y_test, label_encoder
    
    # If no existing dataset is found, load and process from scratch
    # Find all instrument directories
    instrument_dirs = [d for d in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, d))]
    
    # Collect audio files and labels
    audio_files = []
    labels = []
    
    for instrument in instrument_dirs:
        if instrument == 'combined':  # Skip the combined directory
            continue
            
        instrument_path = os.path.join(data_path, instrument)
        wav_files = glob.glob(os.path.join(instrument_path, '**', '*.wav'), recursive=True)
        
        for wav_file in wav_files:
            audio_files.append(wav_file)
            labels.append(instrument)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        audio_files, y_encoded, test_size=test_size, 
        random_state=random_state, stratify=y_encoded
    )
    
    # Save the dataset split if path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_encoder': label_encoder
        }
        save_file = os.path.join(save_path, 'dataset_split.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset split saved to {save_file}")
    
    print(f"Loaded {len(audio_files)} audio files from {len(instrument_dirs)} instruments")
    print(f"Training set: {len(X_train)} files, Test set: {len(X_test)} files")
    
    return X_train, X_test, y_train, y_test, label_encoder


def load_yamnet():
    """
    Load pre-trained YAMNet model.
    
    Returns:
        model: Loaded YAMNet model from TensorFlow Hub.
    """
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model


def extract_yamnet_embedding(audio_file, model):
    """
    Extract YAMNet embedding from audio file.
    
    Args:
        audio_file (str): Path to audio file.
        model: YAMNet model.
        
    Returns:
        numpy.ndarray: YAMNet embedding vector or None if processing fails.
    """
    try:
        # Load and resample audio to 16kHz mono
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Process with YAMNet
        # embedding is 1024 based on the model, which can be used to train a classifier
        scores, embeddings, log_mel_spectrogram = model(audio)
        
        # Average pooling over time
        embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        return embedding
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


def extract_embeddings(audio_files, yamnet_model, save_path=None, 
                      load_from_disk=False, original_save_path=None):
    """
    Extract embeddings from audio files using YAMNet.
    
    Args:
        audio_files (list): List of audio file paths.
        yamnet_model: YAMNet model.
        save_path (str, optional): Path to save embeddings to disk.
        load_from_disk (bool): Whether to try loading embeddings from disk first.
        original_save_path (str, optional): Path to check for existing embeddings.
        
    Returns:
        tuple: Array of embeddings and list of successfully processed files.
    """
    # First check if embeddings exist in the original directory
    if load_from_disk and original_save_path is not None:
        original_file = os.path.join(original_save_path, 'yamnet_embeddings.pkl')
        if os.path.exists(original_file):
            print(f"Checking for embeddings in original location: {original_file}")
            with open(original_file, 'rb') as f:
                data = pickle.load(f)
                yamnet_embeddings = data['yamnet_embeddings']
                successful_files = data['successful_files']
                
                # Check if all files in audio_files are in successful_files
                if all(file in successful_files for file in audio_files):
                    print("All requested embeddings found in original saved file.")
                    # Filter embeddings to match the order of audio_files
                    indices = [successful_files.index(file) for file in audio_files]
                    filtered_embeddings = np.array(yamnet_embeddings)[indices]
                    
                    # Copy to new location if different
                    if save_path is not None and save_path != original_save_path:
                        os.makedirs(save_path, exist_ok=True)
                        # We'll save the filtered embeddings to avoid duplicating large files
                        new_data = {
                            'yamnet_embeddings': filtered_embeddings,
                            'successful_files': audio_files
                        }
                        save_file = os.path.join(save_path, 'yamnet_embeddings.pkl')
                        with open(save_file, 'wb') as f_new:
                            pickle.dump(new_data, f_new)
                        print(f"Embeddings copied to {save_file}")
                    
                    return filtered_embeddings, audio_files
                else:
                    print("Not all requested files found in original saved embeddings. "
                          "Extracting missing embeddings...")
    
    # Check if embeddings exist in the new save_path
    if load_from_disk and save_path is not None:
        embed_file = os.path.join(save_path, 'yamnet_embeddings.pkl')
        if os.path.exists(embed_file):
            print(f"Checking for embeddings in new location: {embed_file}")
            with open(embed_file, 'rb') as f:
                data = pickle.load(f)
                yamnet_embeddings = data['yamnet_embeddings']
                successful_files = data['successful_files']
                
                # Check if all files in audio_files are in successful_files
                if all(file in successful_files for file in audio_files):
                    print("All requested embeddings found in new saved file.")
                    # Filter embeddings to match the order of audio_files
                    indices = [successful_files.index(file) for file in audio_files]
                    return np.array(yamnet_embeddings)[indices], audio_files
                else:
                    print("Not all requested files found in new saved embeddings. "
                          "Extracting missing embeddings...")
    
    # If we reach here, we need to extract embeddings
    yamnet_embeddings = []
    successful_files = []
    
    for audio_file in tqdm(audio_files, desc="Extracting embeddings"):
        # Extract YAMNet embedding
        yamnet_emb = extract_yamnet_embedding(audio_file, yamnet_model)
        
        if yamnet_emb is not None:
            yamnet_embeddings.append(yamnet_emb)
            successful_files.append(audio_file)
    
    # Save embeddings if path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        data = {
            'yamnet_embeddings': yamnet_embeddings,
            'successful_files': successful_files
        }
        save_file = os.path.join(save_path, 'yamnet_embeddings.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Embeddings saved to {save_file}")
    
    # Print summary
    if len(successful_files) == 0:
        print("Warning: No files were successfully processed!")
        return np.array([]), []
    else:
        print(f"Successfully extracted embeddings from "
              f"{len(successful_files)}/{len(audio_files)} files")
        return np.array(yamnet_embeddings), successful_files


def train_classifiers(X_train_yamnet, y_train, save_path=None, original_save_path=None):
    """
    Train classifiers on the embeddings.
    
    Args:
        X_train_yamnet (numpy.ndarray): YAMNet embeddings for training.
        y_train (numpy.ndarray): Training labels.
        save_path (str, optional): Path to save models to disk.
        original_save_path (str, optional): Path to check for existing models.
        
    Returns:
        dict: Dictionary of trained models.
    """
    # Check if models exist in the original directory
    if original_save_path is not None:
        original_file = os.path.join(original_save_path, 'trained_models.pkl')
        if os.path.exists(original_file):
            print(f"Loading existing models from {original_file}")
            with open(original_file, 'rb') as f:
                models = pickle.load(f)
                
                # Copy to new location if different
                if save_path is not None and save_path != original_save_path:
                    os.makedirs(save_path, exist_ok=True)
                    save_file = os.path.join(save_path, 'trained_models.pkl')
                    with open(save_file, 'wb') as f_new:
                        pickle.dump(models, f_new)
                    print(f"Models copied to {save_file}")
                
                return models
    
    # Define classifiers
    classifiers = {
        'Logistic_Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000, random_state=42, n_jobs=-1))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=1000, 
                random_state=42, verbose=False))
        ])
    }
    
    print("Training classifiers on YAMNet embeddings...")
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train_yamnet, y_train)
        print(f"{name} trained successfully.")
    
    # Save models if path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, 'trained_models.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(classifiers, f)
        print(f"Models saved to {save_file}")
    
    return classifiers


def evaluate_classifiers(models, X_test, y_test, label_encoder, save_path=None):
    """
    Evaluate classifiers on the test set and save detailed metrics.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        label_encoder: LabelEncoder used to encode the labels.
        save_path (str, optional): Path to save evaluation results to disk.
        
    Returns:
        dict: Dictionary of evaluation results.
    """
    results = {}
    
    # Prepare for saving model metrics
    metrics_dict = {}
    all_metrics_rows = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        # Get class names from label encoder
        class_names = label_encoder.classes_
        
        # Calculate metrics for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=range(len(class_names)))
        
        # Create metrics dictionary for this model
        model_metrics = {}
        model_metrics_rows = []
        
        for i, class_name in enumerate(class_names):
            class_metrics = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            model_metrics[class_name] = class_metrics
            
            # Add to CSV rows
            model_metrics_rows.append({
                'Model': name,
                'Class': class_name,
                'Precision': precision[i],
                'Recall': recall[i],
                'F1 Score': f1[i],
                'Support': support[i]
            })
            
            all_metrics_rows.append(model_metrics_rows[-1])
        
        # Add to overall metrics dictionary
        metrics_dict[name] = model_metrics
        
        # Calculate and display overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {accuracy:.4f}")
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'class_names': class_names
        }
        
        # Save model-specific metrics if path is provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            
            # Save JSON metrics for this model
            model_metrics_file = os.path.join(save_path, f"{name}_metrics.json")
            with open(model_metrics_file, 'w') as f:
                json.dump(model_metrics, f, indent=4)
            
            # Save CSV metrics for this model
            model_metrics_df = pd.DataFrame(model_metrics_rows)
            model_metrics_csv = os.path.join(save_path, f"{name}_metrics.csv")
            model_metrics_df.to_csv(model_metrics_csv, index=False)
            
            # Save confusion matrix visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'{name} Confusion Matrix')
            plt.colorbar()
            
            # Add axis labels
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=90)
            plt.yticks(tick_marks, class_names)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            plt.tight_layout()
            cm_dir = os.path.join(save_path, 'confusion_matrices')
            os.makedirs(cm_dir, exist_ok=True)
            cm_file = os.path.join(cm_dir, f"{name}_confusion_matrix.png")
            plt.savefig(cm_file, dpi=300)
            plt.close()
    
    # Save consolidated metrics if path is provided
    if save_path is not None:
        # Save consolidated JSON metrics
        metrics_json_file = os.path.join(save_path, "metrics.json")
        with open(metrics_json_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        # Save consolidated CSV metrics
        all_metrics_df = pd.DataFrame(all_metrics_rows)
        metrics_csv_file = os.path.join(save_path, "metrics.csv")
        all_metrics_df.to_csv(metrics_csv_file, index=False)
        
        # Save evaluation results
        eval_file = os.path.join(save_path, 'evaluation_results.pkl')
        with open(eval_file, 'wb') as f:
            pickle.dump(results, f)
    
    return results


def display_results(results, save_path=None):
    """
    Display and optionally save evaluation results.
    
    Args:
        results (dict): Dictionary of evaluation results.
        save_path (str, optional): Path to save results visualization.
    """
    # Create a summary dataframe
    summary_data = []
    for model_name, model_results in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': model_results['accuracy']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nResults Summary:")
    print(summary_df)
    
    # Save summary if path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        summary_csv = os.path.join(save_path, 'results_summary.csv')
        summary_df.to_csv(summary_csv, index=False)


def main():
    """Run the instrument classification pipeline with enhanced metrics."""
    # Define paths
    data_path = 'data/processed/yamnet'
    save_dir = 'models/yamnet/data'
    metrics_dir = 'models/yamnet/results/metrics'
    
    # Load dataset
    X_train, X_test, y_train, y_test, label_encoder = load_dataset(
        data_path, save_path=save_dir)
    
    # Load YAMNet model
    yamnet_model = load_yamnet()
    
    # Extract embeddings
    X_train_yamnet, train_files = extract_embeddings(
        X_train, yamnet_model, save_path=save_dir, load_from_disk=True)
    X_test_yamnet, test_files = extract_embeddings(
        X_test, yamnet_model, save_path=save_dir, load_from_disk=True)
    
    # Update labels if some files failed to process
    if len(train_files) != len(X_train):
        # Filter y_train to match successful files
        indices = [X_train.index(file) for file in train_files]
        y_train = y_train[indices]
    
    if len(test_files) != len(X_test):
        # Filter y_test to match successful files
        indices = [X_test.index(file) for file in test_files]
        y_test = y_test[indices]
    
    # Train classifiers
    models = train_classifiers(X_train_yamnet, y_train, save_path=save_dir)
    
    # Evaluate classifiers
    results = evaluate_classifiers(
        models, X_test_yamnet, y_test, label_encoder, save_path=metrics_dir)
    
    # Display results
    display_results(results, save_path=metrics_dir)
    
    # Calculate weighted metrics
    print("\nCalculating weighted metrics...")
    from models.yamnet.utils.metrics import main as calculate_metrics
    calculate_metrics()


if __name__ == "__main__":
    main() 