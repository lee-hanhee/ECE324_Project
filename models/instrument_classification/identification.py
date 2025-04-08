import torch
import librosa
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torchinfo import summary  
import sklearn.metrics as metrics
import os
import copy
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR  # Added for LR scheduling
from visualizations import (
    plot_metrics,
    plot_confusion_matrix,
    plot_confusion_bar,
    show_saliency,
)
from processing import get_data
from pathlib import Path
from tqdm import tqdm
from jaxtyping import Float
from torch import Tensor

# Global variables (used across multiple functions)
LABELS = {}  # Will be populated during data loading

class InstrumentClassifier(nn.Module):
    """CNN-based model for instrument classification in audio samples.
    
    This model uses a series of convolutional layers with batch normalization,
    followed by pooling and fully connected layers to classify instruments
    from mel spectrograms of audio.
    
    Attributes:
        conv1, conv2, conv3 (nn.Conv2d): Convolutional layers
        bn1, bn2, bn3 (nn.BatchNorm2d): Batch normalization layers
        pool (nn.MaxPool2d): Max pooling layer
        global_pool (nn.AdaptiveAvgPool2d): Global adaptive pooling
        fc1, fc2 (nn.Linear): Fully connected layers
        dropout (nn.Dropout): Dropout layer for regularization
    """
    def __init__(self, num_classes=14, padding=1, dropout=0.3, kernel_size=3):
        """Initialize the instrument classifier network.
        
        Args:
            num_classes (int): Number of instrument classes to classify
            padding (int): Padding size for convolutional layers
            dropout (float): Dropout rate for regularization
            kernel_size (int): Size of convolutional kernels
        """
        super(InstrumentClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Adaptive pooling to 1x1

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)  # Regularization with dropout
        self.fc2 = nn.Linear(64, num_classes)

    def forward(
        self, x: Float[Tensor, "batch 1 mel time"]
    ) -> Float[Tensor, "batch num_classes"]:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch, 1, mel_bins, time_frames]
               representing mel spectrograms
               
        Returns:
            Tensor of shape [batch, num_classes] with class logits
        """
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third convolutional block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten before FC layers

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BabySlakhDataset(Dataset):
    """Dataset for multi-instrument classification using audio segments.
    
    This dataset processes audio files into fixed-length segments and
    converts them to mel spectrograms for model input. It handles
    multi-hot encoding for multi-label classification of instruments.
    """
    def __init__(
        self,
        stem_dict,
        LABELS,
        segment_duration=2.0,
        sample_rate=22050,
        n_mels=128,
        energy_threshold=0.01,
        num_classes=5,
    ):
        """Initialize the dataset with audio files and their instrument labels.
        
        Args:
            stem_dict (dict): Dictionary mapping file paths to lists of label indices
            LABELS (dict): Dictionary mapping label names to indices
            segment_duration (float): Duration in seconds of each audio segment
            sample_rate (int): Audio sample rate in Hz
            n_mels (int): Number of mel frequency bins
            energy_threshold (float): Minimum energy to consider a segment as containing music
            num_classes (int): Total number of instrument classes
        """
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.segments = []
        self.num_classes = num_classes
        self.LABELS = LABELS

        # Process each audio file and its labels
        for path, labels in stem_dict.items():
            # Load audio file
            waveform, sr = librosa.load(path, sr=self.sample_rate)
            total_samples = len(waveform)

            # Split into fixed-length segments
            for start in range(0, total_samples, self.segment_samples):
                end = start + self.segment_samples
                segment = waveform[start:end]
                if len(segment) < self.segment_samples:
                    continue  # Skip incomplete segment

                # Check if segment has enough energy to be considered music
                energy = np.mean(np.abs(segment))
                if energy < energy_threshold:
                    # No music detected, set no_music label
                    multi_hot = torch.zeros(num_classes)
                    multi_hot[self.LABELS["no_music"]] = 1
                else:
                    # Create multi-hot encoding for present instruments
                    multi_hot = torch.zeros(num_classes)
                    for label in labels:
                        multi_hot[label] = 1

                self.segments.append((segment, multi_hot))

    def __len__(self):
        """Return the number of segments in the dataset."""
        return len(self.segments)

    def __getitem__(
        self, idx: int
    ) -> tuple[Float[Tensor, "1 mel time"], Float[Tensor, "num_classes"]]:
        """Get an audio segment as mel spectrogram and its multi-hot encoded labels.
        
        Args:
            idx (int): Index of the segment to retrieve
            
        Returns:
            tuple: (mel_spectrogram, multi_hot_labels)
                - mel_spectrogram: Log-scaled mel spectrogram tensor
                - multi_hot_labels: Multi-hot encoded instrument labels
        """
        waveform_segment, multi_hot_label = self.segments[idx]
        
        # Convert waveform to mel spectrogram
        mel_spec = self.mel_transform(torch.tensor(waveform_segment).unsqueeze(0))
        mel_spec = torch.log1p(mel_spec)  # Convert to log-scale for stability
        
        return (
            mel_spec,
            multi_hot_label,
        )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    label: str = "Validation",
    display_conf_matrix: bool = False,
) -> tuple[float, np.ndarray]:
    """
    Evaluate the model on the given dataset.
    Parameters:
        model: Trained model.
        dataloader: DataLoader for the dataset.
        label: Label for the dataset (e.g., "Validation", "Test").
        display_conf_matrix: Boolean indicating whether to display confusion matrix.
    Returns:
        accuracy: Accuracy of the model on the dataset.
        all_preds: All predictions made by the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel_spec, labels in tqdm(
            dataloader, desc=f"Evaluating {label}", leave=False
        ):
            mel_spec, labels = mel_spec.to(device), labels.to(device)

            outputs = model(mel_spec)
            predicted = torch.sigmoid(outputs) >= 0.5  # Convert to binary predictions

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute multi-label accuracy
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    print(f"{label} Accuracy: {accuracy*100:.2f}%")

    class_names = list(LABELS.keys())
    if display_conf_matrix:
        # Plot the confusion matrix
        plot_confusion_matrix(all_labels, all_preds, class_names, label=label)
        plot_confusion_bar(all_labels, all_preds, class_names, label=label)
    if label != "Validation":
        plot_metrics(all_labels, all_preds, class_names)

    return accuracy, all_preds


def train_with_val(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    lr: float = 0.001,
    num_classes: int = 5,
    patience: int = 5,  # Early stopping patience
    scheduler_type: str = "plateau",  # 'plateau' or 'step'
    scheduler_step_size: int = 3,  # For StepLR
    scheduler_gamma: float = 0.1,  # Factor to reduce LR by
) -> tuple[float | None, dict]:
    """Train the model with validation and advanced training features.
    
    Implements learning rate scheduling and early stopping to improve
    model performance and training efficiency.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (can be None)
        epochs: Maximum number of training epochs
        lr: Initial learning rate
        num_classes: Number of output classes
        patience: Number of epochs to wait for validation improvement before stopping
        scheduler_type: Type of learning rate scheduler ('plateau' or 'step')
        scheduler_step_size: Number of epochs between LR reductions for StepLR
        scheduler_gamma: Multiplicative factor for LR reduction
        
    Returns:
        tuple: (best_val_accuracy, best_model_info)
            - best_val_accuracy: Best validation accuracy (None if val_loader is None)
            - best_model_info: Dictionary with best model state and training info
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # Multi-label loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Set up learning rate scheduler
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=scheduler_gamma, 
            patience=2, verbose=True
        )
    else:  # step scheduler
        scheduler = StepLR(
            optimizer, step_size=scheduler_step_size, 
            gamma=scheduler_gamma
        )
    
    # Early stopping variables
    best_val_accuracy = 0.0
    early_stop_counter = 0
    best_model_info = {
        "state_dict": None,
        "epoch": 0,
        "train_loss": float('inf'),
        "val_accuracy": 0.0
    }
    
    # For plotting
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        display_cm = False

        # Training loop
        for mel_spec, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        ):
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(mel_spec)
            
            # Compute loss and backpropagate
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predicted = torch.sigmoid(outputs)  # Convert logits to probabilities
            predicted = (predicted >= 0.5).float()  # Thresholding
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.numel()

        # Calculate and record training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * total_correct / total_samples
        train_losses.append(train_loss)
        
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2f}%"
        )
        
        # Validation phase
        if epoch == epochs - 1:
            display_cm = True
            
        if val_loader is not None:
            val_accuracy, _ = evaluate(model, val_loader, "Validation", display_cm)
            val_accuracies.append(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
            
            # Learning rate scheduler step
            if scheduler_type == "plateau":
                scheduler.step(val_accuracy)
            else:
                scheduler.step()
                
            # Check for improvement for early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stop_counter = 0
                # Save best model
                best_model_info = {
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_accuracy": val_accuracy
                }
                print(f"Best model saved at epoch {epoch+1}")
            else:
                early_stop_counter += 1
                print(f"Early stopping counter: {early_stop_counter}/{patience}")
                
            # Check early stopping condition
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model
                model.load_state_dict(best_model_info["state_dict"])
                break
        else:
            # If no validation set, use training loss for scheduler
            if scheduler_type == "step":
                scheduler.step()
            # Save final model as best
            best_model_info = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": None
            }
    
    # If we completed all epochs without early stopping, ensure we return the best model
    if val_loader is not None and best_model_info["state_dict"] is not None:
        model.load_state_dict(best_model_info["state_dict"])
        print(f"Restored best model from epoch {best_model_info['epoch']}")
        
    return best_val_accuracy, best_model_info


def run_cross_validation(
    LABELS, 
    train_val_len, 
    train_val_data, 
    k=5, 
    fold_epochs=3,
    patience=3,
    scheduler_type="plateau"
):
    """Run k-fold cross-validation on the dataset.
    
    Args:
        LABELS: Dictionary mapping label names to indices
        train_val_len: Length of the training+validation dataset
        train_val_data: Dataset for training and validation
        k: Number of folds for cross-validation
        fold_epochs: Maximum number of epochs for each fold
        patience: Early stopping patience (epochs)
        scheduler_type: Type of learning rate scheduler to use
        
    Returns:
        dict: Information about the best performing model across all folds
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_metrics = []
    best_fold_info = None
    best_accuracy = 0.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(train_val_len))):
        print(f"\n--- Fold {fold+1}/{k} ---")

        # Create train and validation subsets
        train_subset = Subset(train_val_data, train_idx)
        val_subset = Subset(train_val_data, val_idx)

        # Data loaders
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        # Initialize model
        model = InstrumentClassifier(num_classes=len(LABELS))

        # Train model with early stopping and LR scheduling
        fold_accuracy, fold_info = train_with_val(
            model, 
            train_loader, 
            val_loader, 
            epochs=fold_epochs,
            patience=patience,
            scheduler_type=scheduler_type
        )

        # Store fold metrics
        fold_metrics.append({
            "fold": fold + 1, 
            "accuracy": fold_accuracy,
            "best_epoch": fold_info["epoch"]
        })
        
        # Keep track of best fold model
        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy
            best_fold_info = {
                "fold": fold + 1,
                "state_dict": fold_info["state_dict"],
                "accuracy": fold_accuracy,
                "epoch": fold_info["epoch"]
            }

        print(f"Fold {fold+1} Best Validation Accuracy: {fold_accuracy*100:.2f}% (Epoch {fold_info['epoch']})")
        print("\n--- Fold End ---\n")

    # Compute average validation accuracy
    avg_val_accuracy = sum([m["accuracy"] for m in fold_metrics]) / len(fold_metrics)
    print(f"\n--- Cross-validation Completed ---")
    print(f"Average Validation Accuracy: {avg_val_accuracy*100:.2f}%")
    print(f"Best Fold: {best_fold_info['fold']} with {best_fold_info['accuracy']*100:.2f}% accuracy")
    
    return best_fold_info


def get_train_test_split(
    dataset: Dataset, test_split: float = 0.2
) -> tuple[Dataset, int, Dataset]:

    """
    Split the dataset into training+validation and test sets.
    Parameters:
        dataset: Dataset to be split.
        test_split: Proportion of the dataset to include in the test split.
    Returns:
        train_val_dataset: Training+validation dataset.
        train_val_len: Length of the training+validation dataset."""
    # Split into train+val (80%) and test (20%)
    total_len = len(dataset)
    test_len = int(test_split * total_len)
    train_val_len = total_len - test_len

    train_val_dataset, test_dataset = random_split(dataset, [train_val_len, test_len])
    return train_val_dataset, train_val_len, test_dataset


def get_model(
    train_val_data: Dataset,
    train_val_length: int,
    LABELS: dict[str, int],
    cross_validation: bool = False,
    k_fold_splits: int = 5,
    fold_epochs: int = 5,
    final_epochs: int = 10,
    padding: int = 1,
    dropout: float = 0.3,
    kernel_size: int = 3,
    patience: int = 5,
    scheduler_type: str = "plateau",
    lr: float = 0.001,
) -> tuple[nn.Module, dict]:
    """Train and return a model on the dataset with advanced training features.
    
    This function either performs cross-validation to find the best hyperparameters
    or directly trains on the full training dataset.
    
    Args:
        train_val_data: Dataset for training and validation
        train_val_length: Length of the training+validation dataset
        LABELS: Dictionary mapping labels to indices
        cross_validation: Whether to perform cross-validation
        k_fold_splits: Number of folds for cross-validation
        fold_epochs: Number of epochs for each fold in cross-validation
        final_epochs: Maximum number of epochs for final training
        padding: Padding for convolutional layers
        dropout: Dropout rate for the model
        kernel_size: Kernel size for convolutional layers
        patience: Patience for early stopping
        scheduler_type: Type of learning rate scheduler
        lr: Learning rate
        
    Returns:
        tuple: (final_model, training_info)
            - final_model: Trained model
            - training_info: Dictionary with training information
    """
    best_fold_info = None
    
    if cross_validation:
        # Cross-validation to find best hyperparameters
        print("\n--- Starting Cross-Validation ---")
        best_fold_info = run_cross_validation(
            LABELS, 
            train_val_length, 
            train_val_data, 
            k=k_fold_splits, 
            fold_epochs=fold_epochs,
            patience=patience,
            scheduler_type=scheduler_type
        )
        print("\n--- Retraining on Full Train+Val Dataset with CV Results ---")
    else:
        print("\n--- Training on Full Train+Val Dataset - NO CROSS VAL ---")

    # Initialize the final model
    final_model = InstrumentClassifier(
        num_classes=len(LABELS),
        padding=padding,
        dropout=dropout,
        kernel_size=kernel_size
    )
    
    # If we did cross-validation, initialize with the best fold's weights
    if cross_validation and best_fold_info and best_fold_info["state_dict"]:
        final_model.load_state_dict(best_fold_info["state_dict"])
        print(f"Initialized with best weights from fold {best_fold_info['fold']}")
    
    # Prepare data for final training
    final_train_loader = DataLoader(train_val_data, batch_size=8, shuffle=True)
    
    # Train the final model
    _, training_info = train_with_val(
        final_model, 
        final_train_loader, 
        None,  # No validation set for final training
        epochs=final_epochs,
        lr=lr,
        num_classes=len(LABELS),
        patience=patience,
        scheduler_type=scheduler_type
    )
    
    print(f"Final model trained for {training_info['epoch']} epochs")
    
    return final_model, training_info


def generate_saliency_map(
    model: nn.Module,
    mel_spec: Float[Tensor, "1 mel time"],
    class_index: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Generate saliency map for a specific class using gradients.
    Parameters:
        model: Trained model.
        mel_spec: Mel spectrogram tensor.
        class_index: Index of the class to visualize.
        device: Device to run the model on (CPU or GPU).
    Returns:
        saliency: Saliency map as a numpy array.
    """
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_spec = mel_spec.to(device)
    mel_spec.requires_grad_()

    model.to(device)
    output = model(mel_spec)
    score = output[:, class_index].sum()  # Focus on one class score

    score.backward()
    saliency = (
        mel_spec.grad.data.abs().squeeze().cpu().numpy()
    )  # Get gradient magnitude
    return saliency


def interpret_full_audio(
    model: nn.Module,
    audio_path: str,
    LABELS: dict[str, int],
    sample_rate: int = 22050,
    n_mels: int = 128,
) -> None:
    """
    Interpret the full audio file and visualize saliency maps for relevant classes.
    Parameters:
        model: Trained model.
        audio_path: Path to the audio file.
        LABELS: Dictionary mapping labels to indices.
        sample_rate: Sample rate for audio processing.
        n_mels: Number of Mel spectrogram bins.
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 1. Load full waveform
    waveform, _ = librosa.load(audio_path, sr=sample_rate)

    # 2. Create Mel spectrogram
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    mel_spec = mel_transform(torch.tensor(waveform).unsqueeze(0))
    mel_spec = torch.log1p(mel_spec)  # log-scale for stability

    mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension

    # 3. Forward pass
    with torch.no_grad():
        outputs = model(mel_spec.to(device))
        probs = torch.sigmoid(outputs).cpu().squeeze().numpy()

    # 4. Visualize saliency for relevant classes (excluding "no_music")
    for idx, prob in enumerate(probs):
        class_name = list(LABELS.keys())[idx]
        if class_name == "no_music":
            continue
        if prob >= 0.5:
            saliency = generate_saliency_map(
                model, mel_spec.clone(), idx, device=device
            )
            show_saliency(mel_spec, saliency, class_name=class_name)


def load_and_run_model(
    path_of_model: str,
    data_dict: dict[str, list[int]],
    LABELS: dict[str, int],
    interpret: bool = True,
) -> tuple[float, list[str]]:
    """
    Load the model and run it on the provided dataset.
    Parameters:
        path_of_model: Path to the saved model.
        data_dict: Dictionary containing audio file paths and their corresponding labels.
        LABELS: Dictionary mapping labels to indices.
        interpret: Boolean indicating whether to interpret the audio files.
    Returns:
        accuracy: Accuracy of the model on the dataset.
        pred_labels: List of predicted labels for the audio files.
    """
    dataset = BabySlakhDataset(data_dict, num_classes=len(LABELS), LABELS = LABELS)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = InstrumentClassifier(num_classes=len(LABELS))
    model.load_state_dict(torch.load(path_of_model))

    accuracy, predictions = evaluate(
        model, dataloader, label="Predict", display_conf_matrix=False
    )
    class_counts = np.sum(predictions, axis=0)

    # Classify a label as True if it appears more than the threshold
    song_label = (class_counts > 10).astype(int)

    pred_labels = []
    for label, num in LABELS.items():
        if song_label[num] == 1:
            pred_labels.append(label)

    if interpret:
        for audio_path in data_dict.keys():
            print(f"Interpreting audio: {audio_path}")
            interpret_full_audio(model, audio_path, LABELS)

    return accuracy, pred_labels


if __name__ == "__main__":
    model_path = "models/instrument_classification/saved_model.pth"
    model_arch_path = "models/instrument_classification/model_v1.pth"
    save_model = False
    train_model = False
    interpret = True
    
    # Configure hyperparameters
    hyperparameters = {
        "epochs": 15,
        "batch_size": 8,
        "learning_rate": 0.001,
        "num_classes": 14,
        "padding": 1,
        "dropout": 0.3,
        "kernel_size": 3,
        "segment_duration": 2.0,
        "sample_rate": 22050,
        "n_mels": 128,
        "energy_threshold": 0.01,
        "patience": 5,
        "scheduler_type": "plateau",
        "scheduler_gamma": 0.1,
        "test_split": 0.3,
        "cross_validation": False,
    }

    if train_model:
        # Load and prepare data
        inst_dict, LABELS = get_data(percent=0.8, seed=42)
        hyperparameters["num_classes"] = len(LABELS)
        
        print(f"Training on {len(inst_dict)} audio files with {len(LABELS)} instrument classes")
        
        # Create dataset
        full_dataset = BabySlakhDataset(
            inst_dict, 
            LABELS,
            segment_duration=hyperparameters["segment_duration"],
            sample_rate=hyperparameters["sample_rate"],
            n_mels=hyperparameters["n_mels"],
            energy_threshold=hyperparameters["energy_threshold"],
            num_classes=hyperparameters["num_classes"]
        )
        
        print(f"Created dataset with {len(full_dataset)} segments")

        # Split into train and test
        train_data, train_len, test_data = get_train_test_split(
            full_dataset, test_split=hyperparameters["test_split"]
        )
        
        print(f"Split into {train_len} training and {len(test_data)} test samples")
        
        # Train model
        final_model, training_info = get_model(
            train_data,
            train_len,
            LABELS,
            cross_validation=hyperparameters["cross_validation"],
            k_fold_splits=5,
            fold_epochs=5,
            final_epochs=hyperparameters["epochs"],
            padding=hyperparameters["padding"],
            dropout=hyperparameters["dropout"],
            kernel_size=hyperparameters["kernel_size"],
            patience=hyperparameters["patience"],
            scheduler_type=hyperparameters["scheduler_type"],
            lr=hyperparameters["learning_rate"]
        )

        # Save model
        if save_model:
            # Save just the model weights
            torch.save(final_model.state_dict(), model_path)
            print(f"Model weights saved to {model_path}")
            
            # Save the complete model with architecture for easier loading
            model_info = {
                "model_state_dict": final_model.state_dict(),
                "hyperparameters": hyperparameters,
                "labels": LABELS,
                "training_info": training_info
            }
            torch.save(model_info, model_arch_path)
            print(f"Complete model saved to {model_arch_path}")

        # Test the model
        print("\n--- Final Test Evaluation ---")
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
        test_accuracy, _ = evaluate(final_model, test_loader, label="Test", display_conf_matrix=True)
        print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")

    else:
        # Load a pre-trained model for inference
        new_dict, LABELS = get_data(percent=0.1, seed=1)
        accuracy, pred_labels = load_and_run_model(
            model_path, new_dict, LABELS, interpret=True
        )
        print(f"Model loaded from {model_path}")
        print(f"Predicted instruments: {', '.join(pred_labels)}")
        print(f"Accuracy: {accuracy*100:.2f}%")

    # Print Model Summary
    model = InstrumentClassifier(num_classes=hyperparameters["num_classes"])
    dummy_input = torch.randn(1, 1, hyperparameters["n_mels"], 221)  # Example input size
    summary(model, input_size=(1, 1, hyperparameters["n_mels"], 221))