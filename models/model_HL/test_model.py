import torch
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
from model import create_model
from dataset import InstrumentDataset
from torch.utils.data import DataLoader, random_split
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

def load_model(model_path, num_classes):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path (str): Path to the model checkpoint
        num_classes (int): Number of instrument classes
        
    Returns:
        model: Loaded model with weights
    """
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test_ensemble(model_paths, test_loader, device, class_names):
    """
    Test an ensemble of models on a test dataset.
    
    Args:
        model_paths (list): List of paths to model checkpoints
        test_loader (DataLoader): DataLoader for test data
        device (torch.device): Device to run inference on
        class_names (list): List of class names
        
    Returns:
        tuple: (predictions, true_labels, accuracy)
    """
    # Load all models
    models = []
    for path in model_paths:
        model = load_model(path, len(class_names))
        model = model.to(device)
        models.append(model)
    
    # Initialize prediction and label lists
    all_preds = []
    all_labels = []
    
    # Test each batch
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get predictions from each model
            batch_preds = []
            for model in models:
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                batch_preds.append(probs)
            
            # Average predictions from all models
            ensemble_probs = torch.stack(batch_preds).mean(dim=0)
            ensemble_preds = (ensemble_probs > 0.5).float()
            
            all_preds.append(ensemble_preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate accuracy
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.numel()
    accuracy = correct / total
    
    return all_preds, all_labels, accuracy

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix for each instrument class.
    
    Args:
        y_true (torch.Tensor): True labels
        y_pred (torch.Tensor): Predicted labels
        class_names (list): List of class names
    """
    plt.figure(figsize=(15, 12))
    
    for i, class_name in enumerate(class_names):
        # Get true and predicted values for this class
        true_class = y_true[:, i].numpy()
        pred_class = y_pred[:, i].numpy()
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_class, pred_class)
        
        # Plot
        plt.subplot(3, 4, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Not Present', 'Present'],
                   yticklabels=['Not Present', 'Present'])
        plt.title(f'Instrument: {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
    data_dir = os.path.join(project_root, 'data', 'raw')
    
    print(f"Loading data from: {data_dir}")
    
    # Create dataset
    dataset = InstrumentDataset(
        root_dir=data_dir,
        transform=None,
        sample_rate=44100,
        n_mels=128,
        segment_duration=5.0
    )
    
    # Get class names
    class_names = list(dataset.class_to_idx.keys())
    print(f"Class names: {class_names}")
    
    # Split dataset into train and test sets
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    _, test_dataset = random_split(dataset, [train_size, test_size], 
                                   generator=torch.Generator().manual_seed(42))
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Define paths to ensemble models
    ensemble_paths = [f'ensemble_model_{i+1}.pth' for i in range(5)]
    
    # Check which models exist
    existing_models = [path for path in ensemble_paths if os.path.exists(path)]
    
    if not existing_models:
        print("No trained models found. Checking for fold models...")
        fold_models = [f'best_model_fold_{i+1}.pth' for i in range(5)]
        existing_models = [path for path in fold_models if os.path.exists(path)]
    
    if not existing_models:
        print("No trained models found. Please train the model first.")
        return
    
    print(f"Found {len(existing_models)} model(s): {existing_models}")
    
    # Test ensemble
    predictions, true_labels, accuracy = test_ensemble(
        model_paths=existing_models,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )
    
    # Print results
    print(f'\nEnsemble Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    # Calculate and print F1 scores per class
    f1_scores = []
    for i, class_name in enumerate(class_names):
        f1 = f1_score(true_labels[:, i].numpy(), predictions[:, i].numpy())
        f1_scores.append(f1)
        print(f'F1 Score for {class_name}: {f1:.4f}')
    
    print(f'Average F1 Score: {np.mean(f1_scores):.4f}')
    
    # Generate detailed classification report
    print("\nClassification Report:")
    for i, class_name in enumerate(class_names):
        report = classification_report(
            true_labels[:, i].numpy(), 
            predictions[:, i].numpy(),
            target_names=['Not Present', 'Present'],
            output_dict=True
        )
        print(f"\n{class_name}:")
        print(f"Precision: {report['Present']['precision']:.4f}")
        print(f"Recall: {report['Present']['recall']:.4f}")
        print(f"F1-score: {report['Present']['f1-score']:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrix(true_labels, predictions, class_names)
    print("Confusion matrices saved to 'confusion_matrices.png'")

if __name__ == '__main__':
    main() 