import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    precision_recall_fscore_support,
)
import numpy as np
from jaxtyping import Float, Bool, Int
from typing import List, Optional
import os

# Ensure results directory exists
SAVE_DIR = "models/instrument_classification/results"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_confusion_matrix(
    all_labels: Bool[np.ndarray, "batch num_classes"],
    all_preds: Bool[np.ndarray, "batch num_classes"],
    class_names: List[str],
    label: str = "Test",
) -> None:
    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)
    num_classes = len(class_names)
    num_cols = int(np.ceil(num_classes / 2))

    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 3, 6))
    fig.suptitle(f"{label} Confusion Matrices", fontsize=16)

    axes = axes.flatten()
    for i, class_name in enumerate(class_names):
        cm = conf_matrices[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=axes[i],
        )
        axes[i].set_title(class_name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    save_path = os.path.join(SAVE_DIR, f"{label.lower()}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()


def plot_confusion_bar(
    all_labels: Bool[np.ndarray, "batch num_classes"],
    all_preds: Bool[np.ndarray, "batch num_classes"],
    class_names: List[str],
    label: str = "Validation",
) -> None:
    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)

    TP = [cm[1, 1] for cm in conf_matrices]
    FP = [cm[0, 1] for cm in conf_matrices]
    FN = [cm[1, 0] for cm in conf_matrices]
    TN = [cm[0, 0] for cm in conf_matrices]

    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, TN, label="True Negatives", color="darkseagreen")
    ax.bar(x, TP, bottom=np.array(TN), label="True Positives", color="seagreen")
    ax.bar(
        x, FP, bottom=np.array(TN) + np.array(TP), label="False Positives", color="salmon"
    )
    ax.bar(
        x, FN, bottom=np.array(TN) + np.array(TP) + np.array(FP), label="False Negatives", color="red"
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} Set Confusion Metrics Per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"{label.lower()}_confusion_bar.png")
    plt.savefig(save_path)
    plt.close()


def plot_metrics(
    all_labels: Bool[np.ndarray, "batch num_classes"],
    all_preds: Bool[np.ndarray, "batch num_classes"],
    class_names: List[str],
    label: str = "Test",
) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    print_metrics(precision, recall, f1, support, class_names)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", color="skyblue")
    ax.bar(x, recall, width, label="Recall", color="orange")
    ax.bar(x + width, f1, width, label="F1-score", color="green")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(f"{label} Set Metrics Per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"{label.lower()}_metrics.png")
    plt.savefig(save_path)
    plt.close()


def print_metrics(
    precision: Float[np.ndarray, "num_classes"],
    recall: Float[np.ndarray, "num_classes"],
    f1: Float[np.ndarray, "num_classes"],
    support: Int[np.ndarray, "num_classes"],
    class_names: List[str],
) -> None:
    for i, name in enumerate(class_names):
        print(f"{name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-score: {f1[i]:.4f}")
        print(f"  Support: {support[i]}")
        print()

    total_support = np.sum(support)
    weighted_precision = np.sum(precision * support) / total_support
    weighted_recall = np.sum(recall * support) / total_support
    weighted_f1 = np.sum(f1 * support) / total_support
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")


def show_saliency(
    mel_spec: Float[np.ndarray, "freq time"],
    saliency: Float[np.ndarray, "freq time"],
    class_name: str,
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].imshow(
        mel_spec.squeeze().cpu().numpy(), origin="lower", aspect="auto", cmap="viridis"
    )
    ax[0].set_title("Mel Spectrogram")

    ax[1].imshow(saliency, origin="lower", aspect="auto", cmap="hot")
    ax[1].set_title(f"Saliency Map ({class_name})")

    plt.tight_layout()
    if not save_path:
        save_path = os.path.join(SAVE_DIR, f"saliency_{class_name}.png")
    plt.savefig(save_path)
    plt.close()
