import torch
import librosa
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as metrics
import os
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split, Subset
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


class InstrumentClassifier(nn.Module):
    def __init__(self, num_classes=14, padding=1, dropout=0.3, kernel_size=3):
        super(InstrumentClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Adaptive pooling to 1x1

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)  # 30% dropout
        self.fc2 = nn.Linear(64, num_classes)

    def forward(
        self, x: Float[Tensor, "batch 1 mel time"]
    ) -> Float[Tensor, "batch num_classes"]:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten before FC layers

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x


class BabySlakhDataset(Dataset):
    def __init__(
        self,
        stem_dict,
        segment_duration=2.0,
        sample_rate=22050,
        n_mels=128,
        energy_threshold=0.01,
        num_classes=5,
    ):
        """
        Parameters:
            stem_dict (dict): Dictionary mapping file paths to lists of labels.
            segment_duration (float): Duration (in seconds) of each segment.
            sample_rate (int): Audio sample rate.
            n_mels (int): Number of Mel spectrogram bins.
            energy_threshold (float): Threshold to determine if audio segment has music.
            num_classes (int): Total number of instrument classes.
        """
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.segments = []
        self.num_classes = num_classes

        for path, labels in stem_dict.items():
            waveform, sr = librosa.load(path, sr=self.sample_rate)
            total_samples = len(waveform)

            for start in range(0, total_samples, self.segment_samples):
                end = start + self.segment_samples
                segment = waveform[start:end]
                if len(segment) < self.segment_samples:
                    continue  # Skip incomplete segment

                energy = np.mean(np.abs(segment))
                if energy < energy_threshold:
                    multi_hot = torch.zeros(num_classes)  # No music → all zeros
                    multi_hot[LABELS["no_music"]] = 1
                else:
                    multi_hot = torch.zeros(num_classes)
                    for label in labels:  # Set 1s for multiple labels
                        multi_hot[label] = 1

                self.segments.append((segment, multi_hot))

    def __len__(self):
        return len(self.segments)

    def __getitem__(
        self, idx: int
    ) -> tuple[Float[Tensor, "1 mel time"], Float[Tensor, "num_classes"]]:
        waveform_segment, multi_hot_label = self.segments[idx]
        mel_spec = self.mel_transform(torch.tensor(waveform_segment).unsqueeze(0))
        mel_spec = torch.log1p(mel_spec)  # Convert to log-scale for stability
        return (
            mel_spec,
            multi_hot_label,
        )  # Multi-hot labels for multi-class classification


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
) -> float | None:
    """
    Train the model with validation.
    Parameters:
        model: Model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of epochs to train.
        lr: Learning rate for the optimizer.
        num_classes: Number of classes for multi-label classification.
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # Multi-label loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        display_cm = False

        for mel_spec, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        ):
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)  # Now labels are multi-hot

            optimizer.zero_grad()
            outputs = model(mel_spec)  # Get raw logits

            loss = criterion(outputs, labels.float())  # Compute loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # **Fix accuracy calculation**
            predicted = torch.sigmoid(outputs)  # Convert logits to probabilities
            predicted = (predicted >= 0.5).float()  # Thresholding

            correct = (
                (predicted == labels).sum().item()
            )  # Compare entire multi-hot labels
            total_correct += correct
            total_samples += (
                labels.numel()
            )  # Total elements (all classes for all samples)

        # Calculate accuracy
        train_acc = 100 * total_correct / total_samples
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Train Accuracy: {train_acc:.2f}%"
        )
        if epoch == epochs - 1:
            display_cm = True
        if val_loader is not None:
            val_accuracy, _ = evaluate(model, val_loader, "Validation", display_cm)
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
            return val_accuracy

    return None


def run_cross_validation(LABELS, train_val_len, train_val_data, k=5, fold_epochs=5):
    """
    Run k-fold cross-validation on the dataset.
    Parameters:
        LABELS: Dictionary mapping labels to indices.
        train_val_len: Length of the training+validation dataset.
        train_val_data: Dataset for training and validation.
        k: Number of folds for cross-validation.
        fold_epochs: Number of epochs for each fold.
    Returns:
        None"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(train_val_len))):
        print(f"--- Fold {fold+1} ---")

        # Create train and validation subsets
        train_subset = Subset(train_val_data, train_idx)
        val_subset = Subset(train_val_data, val_idx)

        # Data loaders
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        # Initialize model
        model = InstrumentClassifier(num_classes=len(LABELS))

        # Train model
        fold_accuracy = train_with_val(
            model, train_loader, val_loader, epochs=fold_epochs
        )

        # Store fold metrics
        fold_metrics.append({"fold": fold + 1, "accuracy": fold_accuracy})

        print(f"Fold {fold+1} Validation Accuracy: {fold_accuracy:.4f}")
        print("\n--- Fold End ---\n")

    # Compute average validation accuracy
    avg_val_accuracy = sum([m["accuracy"] for m in fold_metrics]) / len(fold_metrics)
    print(f"\n--- Cross-validation Completed ---")
    print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")


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
    final_epochs: int = 15,
    padding: int = 1,
    dropout: float = 0.3,
    kernel_size: int = 3,
) -> nn.Module:
    """
    Get the model and train it on the dataset.
    Parameters:
        train_val_data: Dataset for training and validation.
        train_val_length: Length of the training and validation dataset.
        LABELS: Dictionary mapping labels to indices.
        cross_validation: Boolean indicating whether to perform cross-validation.
        k_fold_splits: Number of splits for cross-validation.
        fold_epochs: Number of epochs for each fold in cross-validation.
        final_epochs: Number of epochs for final training.
        padding: Padding for convolutional layers.
        dropout: Dropout rate for the model.
        kernel_size: Kernel size for convolutional layers.
    Returns:
        final_model: Trained model.
    """
    if cross_validation:
        # Cross-validation setup
        run_cross_validation(
            LABELS, train_val_length, train_val_data, k_fold_splits, fold_epochs
        )
        # --- Final Model Training on Full Train+Val Dataset ---
        print("\n--- Retraining on Full Train+Val Dataset ---")
    else:
        print("\n--- Training on Full Train+Val Dataset - NO CROSS VAL ---")

    final_model = InstrumentClassifier(num_classes=len(LABELS))
    final_train_loader = DataLoader(train_val_data, batch_size=8, shuffle=True)

    train_with_val(
        final_model, final_train_loader, None, epochs=final_epochs
    )  # Train for longer

    return final_model


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
    dataset = BabySlakhDataset(data_dict, num_classes=len(LABELS))
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
    save_model = False
    train_model = False
    hyperparameters = {
        "epochs": 10,
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
    }

    if train_model:
        inst_dict, LABELS = get_data(percent=0.05, seed=42)
        hyperparameters["num_classes"] = len(LABELS)
        # Load dataset
        full_dataset = BabySlakhDataset(
            inst_dict, num_classes=hyperparameters["num_classes"]
        )

        train_data, train_len, test_data = get_train_test_split(
            full_dataset, test_split=0.3
        )
        final_model = get_model(
            train_data,
            train_len,
            LABELS,
            cross_validation=False,
            k_fold_splits=5,
            fold_epochs=5,
            final_epochs=10,
            padding=hyperparameters["padding"],
            dropout=hyperparameters["dropout"],
            kernel_size=hyperparameters["kernel_size"],
        )

        # save model
        if save_model:
            torch.save(final_model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # --- Final Test Evaluation ---
        print("\n--- Final Test Evaluation ---")
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
        evaluate(final_model, test_loader, label="Test", display_conf_matrix=True)
    else:
        new_dict, LABELS = get_data(percent=0.02, seed=1)
        _, _ = load_and_run_model(
            model_path, new_dict, LABELS, interpret=False
        )
        print(f"Model loaded from {model_path}")