import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, precision_recall_fscore_support
import numpy as np


def plot_confusion_matrix(all_labels, all_preds, class_names, label="Test"):
    """
    Plots a heatmap for multi-label confusion matrices in two rows.

    all_labels: Ground truth labels (multi-hot encoded)
    all_preds: Predicted labels (multi-hot encoded)
    class_names: List of class names corresponding to indices
    label: String label for the dataset (Validation/Test)
    """
    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)
    num_classes = len(class_names)
    num_cols = int(np.ceil(num_classes / 2))  # Number of columns (split into two rows)
    
    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 3, 6))
    fig.suptitle(f"{label} Confusion Matrices", fontsize=16)
    
    axes = axes.flatten()  # Flatten to iterate easily
    for i, class_name in enumerate(class_names):
        cm = conf_matrices[i]  # Get the confusion matrix for class i
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], 
                    ax=axes[i])
        axes[i].set_title(class_name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    
    # Hide unused subplots if class count is odd
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def plot_confusion_bar(all_labels, all_preds, class_names, label="Validation"):
    """
    Plots a stacked bar chart of True Positives (TP), False Positives (FP),
    False Negatives (FN), and True Negatives (TN) for multi-label classification.

    all_labels: Ground truth labels (multi-hot encoded)
    all_preds: Predicted labels (multi-hot encoded)
    class_names: List of class names corresponding to indices
    label: String label for the dataset (Validation/Test)
    """
    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)
    
    TP = [cm[1, 1] for cm in conf_matrices]  # True Positives
    FP = [cm[0, 1] for cm in conf_matrices]  # False Positives
    FN = [cm[1, 0] for cm in conf_matrices]  # False Negatives
    TN = [cm[0, 0] for cm in conf_matrices]  # True Negatives
    
    x = np.arange(len(class_names))  # Class indices
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, TN, label='True Negatives', color='darkseagreen')
    ax.bar(x, TP, bottom=np.array(TN), label='True Positives', color='seagreen')
    ax.bar(x, FP, bottom=np.array(TN) + np.array(TP), label='False Positives', color='salmon')
    ax.bar(x, FN, bottom=np.array(TN) + np.array(TP) + np.array(FP), label='False Negatives', color='red')
    
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} Set Confusion Metrics Per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_metrics(all_labels, all_preds, class_names, label="Test"):
    """
    Plots a bar chart of precision, recall, and F1-score for multi-label classification.

    all_labels: Ground truth labels (multi-hot encoded)
    all_preds: Predicted labels (multi-hot encoded)
    class_names: List of class names corresponding to indices
    label: String label for the dataset 
    """
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    x = np.arange(len(class_names))  # Class indices
    width = 0.25  # Bar width
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='orange')
    ax.bar(x + width, f1, width, label='F1-score', color='green')
    
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(f"{label} Set Metrics Per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
