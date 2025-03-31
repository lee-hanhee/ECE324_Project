#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instrument Classifier using YAMNet embeddings

Requirements:
- tensorflow >= 2.5.0
- tensorflow-hub
- librosa
- scikit-learn
- numpy
- pandas
- matplotlib
- tqdm

Installation:
pip install tensorflow tensorflow-hub librosa scikit-learn numpy pandas matplotlib tqdm
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


def load_dataset(data_path, test_size=0.2, random_state=42, save_path=None):
    """
    Load audio files and extract labels from filenames.
    
    Args:
        data_path (str): Path to the instruments directory.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        save_path (str, optional): If provided, saves the dataset splits to disk.
        
    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    # Find all instrument directories
    instrument_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
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
        audio_files, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
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
        with open(os.path.join(save_path, 'dataset_split.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset split saved to {os.path.join(save_path, 'dataset_split.pkl')}")
    
    print(f"Loaded {len(audio_files)} audio files from {len(instrument_dirs)} instruments")
    print(f"Training set: {len(X_train)} files, Test set: {len(X_test)} files")
    
    return X_train, X_test, y_train, y_test, label_encoder

def load_yamnet():
    """Load pre-trained YAMNet model."""
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

def extract_yamnet_embedding(audio_file, model):
    """
    Extract YAMNet embedding from audio file.
    
    Args:
        audio_file (str): Path to audio file.
        model: YAMNet model.
        
    Returns:
        embedding (np.ndarray): YAMNet embedding.
    """
    try:
        # Load and resample audio to 16kHz mono
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Process with YAMNet
        scores, embeddings, log_mel_spectrogram = model(audio)
        
        # Average pooling over time
        embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        return embedding
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def extract_embeddings(audio_files, yamnet_model, save_path=None, load_from_disk=False):
    """
    Extract embeddings from audio files using YAMNet.
    
    Args:
        audio_files (list): List of audio file paths.
        yamnet_model: YAMNet model.
        save_path (str, optional): Path to save embeddings to disk.
        load_from_disk (bool): Whether to try loading embeddings from disk first.
        
    Returns:
        yamnet_embeddings, successful_files (tuple): Array of embeddings and list of successfully processed files.
    """
    # Check if embeddings already exist on disk
    if load_from_disk and save_path is not None and os.path.exists(os.path.join(save_path, 'yamnet_embeddings.pkl')):
        print(f"Loading embeddings from {os.path.join(save_path, 'yamnet_embeddings.pkl')}")
        with open(os.path.join(save_path, 'yamnet_embeddings.pkl'), 'rb') as f:
            data = pickle.load(f)
            yamnet_embeddings = data['yamnet_embeddings']
            successful_files = data['successful_files']
            
            # Check if all files in audio_files are in successful_files
            if all(file in successful_files for file in audio_files):
                print("All requested embeddings found in saved file.")
                # Filter embeddings to match the order of audio_files
                indices = [successful_files.index(file) for file in audio_files]
                return np.array(yamnet_embeddings)[indices], audio_files
            else:
                print("Not all requested files found in saved embeddings. Extracting missing embeddings...")
        
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
        with open(os.path.join(save_path, 'yamnet_embeddings.pkl'), 'wb') as f:
            pickle.dump(data, f)
        print(f"Embeddings saved to {os.path.join(save_path, 'yamnet_embeddings.pkl')}")
    
    # Print summary
    if len(successful_files) == 0:
        print("Warning: No files were successfully processed!")
        return np.array([]), []
    else:
        print(f"Successfully extracted embeddings from {len(successful_files)}/{len(audio_files)} files")
        return np.array(yamnet_embeddings), successful_files

def train_classifiers(X_train_yamnet, y_train, save_path=None):
    """
    Train classifiers on the embeddings.
    
    Args:
        X_train_yamnet (np.ndarray): YAMNet embeddings for training.
        y_train (np.ndarray): Training labels.
        save_path (str, optional): Path to save models to disk.
        
    Returns:
        yamnet_models (dict): Trained models.
    """
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
        'MLP': MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, early_stopping=True)
    }
    
    # Pipelines with scaling
    yamnet_models = {}
    
    print("Training classifiers...")
    
    # Train on YAMNet embeddings
    for name, clf in classifiers.items():
        print(f"Training {name} on YAMNet embeddings...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        pipeline.fit(X_train_yamnet, y_train)
        yamnet_models[name] = pipeline
    
    # Save models if path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'trained_models.pkl'), 'wb') as f:
            pickle.dump(yamnet_models, f)
        print(f"Models saved to {os.path.join(save_path, 'trained_models.pkl')}")
    
    return yamnet_models

def evaluate_classifiers(models, X_test, y_test, label_encoder, save_path=None):
    """
    Evaluate classifiers and return results.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (np.ndarray): Test embeddings.
        y_test (np.ndarray): Test labels.
        label_encoder (LabelEncoder): Label encoder.
        save_path (str, optional): Path to save results to disk.
        
    Returns:
        results (dict): Evaluation results.
    """
    results = {}
    
    print("\nEvaluating YAMNet classifiers...")
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        
        print(f"YAMNet - {name} Accuracy: {accuracy:.4f}")
        
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
        
        # Create and save confusion matrix
        if save_path is not None:
            os.makedirs(os.path.join(save_path, 'confusion_matrices'), exist_ok=True)
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {name}')
            plt.colorbar()
            tick_marks = np.arange(len(label_encoder.classes_))
            plt.xticks(tick_marks, label_encoder.classes_, rotation=90)
            plt.yticks(tick_marks, label_encoder.classes_)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'confusion_matrices', f'{name.replace(" ", "_")}_confusion_matrix.png'))
            plt.close()
    
    # Save results if path is provided
    if save_path is not None:
        with open(os.path.join(save_path, 'evaluation_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print(f"Evaluation results saved to {os.path.join(save_path, 'evaluation_results.pkl')}")
    
    return results

def display_results(results, save_path=None):
    """
    Display and save classification results.
    
    Args:
        results (dict): Evaluation results.
        save_path (str, optional): Path to save results to disk.
    """
    # Create comparison table
    data = {
        'Model': [],
        'YAMNet Accuracy': []
    }
    
    for model_name in results.keys():
        yamnet_acc = results[model_name]['accuracy']
        
        data['Model'].append(model_name)
        data['YAMNet Accuracy'].append(f"{yamnet_acc:.4f}")
    
    # Create and display DataFrame
    df = pd.DataFrame(data)
    print("\nClassifier Performance:")
    print(df.to_string(index=False))
    
    # Determine the best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_name = best_model[0]
    best_acc = best_model[1]['accuracy']
    
    print(f"\nBest model: YAMNet with {best_name} (Accuracy: {best_acc:.4f})")
    
    # Save results table if path is provided
    if save_path is not None:
        df.to_csv(os.path.join(save_path, 'results_summary.csv'), index=False)
        print(f"Results summary saved to {os.path.join(save_path, 'results_summary.csv')}")

def main():
    print("Instrument Classifier using YAMNet embeddings")
    print("-" * 50)
    
    # Create output directory for saved data
    output_dir = 'models/pretrained/saved_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    data_path = 'data/instruments'
    X_train, X_test, y_train, y_test, label_encoder = load_dataset(
        data_path, save_path=output_dir
    )
    
    # Load YAMNet model
    print("Loading YAMNet model...")
    yamnet_model = load_yamnet()
    
    # Extract embeddings with saving to disk
    print("\nExtracting embeddings from training set...")
    X_train_yamnet, X_train_successful = extract_embeddings(
        X_train, yamnet_model, 
        save_path=output_dir,
        load_from_disk=True  # Try to load embeddings if they exist
    )
    
    print("\nExtracting embeddings from test set...")
    X_test_yamnet, X_test_successful = extract_embeddings(
        X_test, yamnet_model, 
        save_path=output_dir,
        load_from_disk=True  # Try to load embeddings if they exist
    )
    
    # Update labels to match successful files
    y_train_filtered = []
    for i, file_path in enumerate(X_train):
        if file_path in X_train_successful:
            y_train_filtered.append(y_train[i])
    
    y_test_filtered = []
    for i, file_path in enumerate(X_test):
        if file_path in X_test_successful:
            y_test_filtered.append(y_test[i])
    
    y_train_filtered = np.array(y_train_filtered)
    y_test_filtered = np.array(y_test_filtered)
    
    # Check if we have enough data to continue
    if len(X_train_yamnet) == 0 or len(X_test_yamnet) == 0:
        print("Error: Not enough successful embeddings extracted to train models.")
        print("Try updating the audio processing functions or check the audio files.")
        return
    
    # Print embedding dimensions
    print(f"\nYAMNet embedding dimension: {X_train_yamnet.shape[1]}")
    
    # Train classifiers
    yamnet_models = train_classifiers(
        X_train_yamnet, y_train_filtered, save_path=output_dir
    )
    
    # Evaluate classifiers
    results = evaluate_classifiers(
        yamnet_models, X_test_yamnet, y_test_filtered, label_encoder, save_path=output_dir
    )
    
    # Display results
    display_results(results, save_path=output_dir)
    
    print(f"\nAll data and models saved to {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    main() 