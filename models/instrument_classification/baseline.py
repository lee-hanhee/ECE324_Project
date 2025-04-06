import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from processing import get_data
from identification import BabySlakhDataset


class BaselineLinearClassifier(nn.Module):
    """A baseline model with a single linear layer over flattened input."""

    def __init__(self, input_shape, num_classes):
        """Initialize the model.

        Args:
            input_shape (torch.Size): Shape of the input spectrogram (1, mel, time).
            num_classes (int): Number of instrument classes.
        """
        super().__init__()
        input_dim = input_shape[1] * input_shape[2]  # mel x time
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, mel, time).

        Returns:
            torch.Tensor: Raw logits of shape (batch, num_classes).
        """
        x = x.view(x.size(0), -1)  # Flatten
        return self.linear(x)


def train_baseline(model, dataloader, num_epochs=5, lr=1e-3):
    """Train the baseline model.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for mel_spec, labels in dataloader:
            mel_spec, labels = mel_spec.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss:.4f}")


def evaluate_baseline(model, dataloader, label_names):
    """Evaluate the baseline model and print metrics.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for evaluation.
        label_names (list[str]): List of class label names.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for mel_spec, labels in dataloader:
            mel_spec, labels = mel_spec.to(device), labels.to(device)
            outputs = model(mel_spec)
            preds = (torch.sigmoid(outputs) >= 0.5).int()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    if y_true.shape[1] == 1:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Baseline)")
        plt.show()


if __name__ == "__main__":
    # === Load dataset ===
    data_dict, LABELS = get_data(percent=0.2, seed=42)
    full_dataset = BabySlakhDataset(data_dict, LABELS=LABELS, num_classes=len(LABELS))

    # === Split into train and test ===
    test_ratio = 0.2
    test_len = int(len(full_dataset) * test_ratio)
    train_len = len(full_dataset) - test_len
    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # === Init model ===
    sample_x, _ = full_dataset[0]
    input_shape = sample_x.shape
    model = BaselineLinearClassifier(input_shape, num_classes=len(LABELS))

    # === Train on train set ===
    print("\nTraining baseline model...\n")
    train_baseline(model, train_loader, num_epochs=5)

    # === Evaluate on test set ===
    print("\nEvaluating baseline model on TEST set...\n")
    evaluate_baseline(model, test_loader, list(LABELS.keys()))