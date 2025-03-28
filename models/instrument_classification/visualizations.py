import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score
import numpy as np


def plot_confusion_matrix(all_labels, all_preds, class_names, label="Validation"):
    """
    Plots a heatmap for multi-label confusion matrices.

    all_labels: Ground truth labels (multi-hot encoded)
    all_preds: Predicted labels (multi-hot encoded)
    class_names: List of class names corresponding to indices
    label: String label for the dataset (Validation/Test)
    """
    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)

    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 4))
    fig.suptitle(f"{label} Confusion Matrices", fontsize=16)

    for i, class_name in enumerate(class_names):
        cm = conf_matrices[i]  # Get the confusion matrix for class i

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[i])
        axes[i].set_title(class_name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    plt.show()


def plot_f1_score(all_labels, all_preds, class_names):
    """
    Plot F1-Score for each class and overall.
    
    all_labels: Ground truth labels (multi-hot encoded)
    all_preds: Predicted labels (binary)
    class_names: List of class names corresponding to indices
    """
    f1_scores = []
    
    for i, class_name in enumerate(class_names):
        f1 = f1_score(all_labels[:, i], all_preds[:, i])
        f1_scores.append(f1)
    
    # Plot F1 Score
    plt.figure(figsize=(4, 3))
    plt.bar(class_names, f1_scores, color='lightgreen')
    plt.axhline(y=np.mean(f1_scores), color='r', linestyle='-', label = "Mean F1 score")
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Multi-label Classification')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_average_precision(all_labels, all_preds, class_names):
    """
    Plot Average Precision (AP) for each class and mean Average Precision (mAP).
    
    all_labels: Ground truth labels (multi-hot encoded)
    all_preds: Predicted labels (probabilities or scores)
    class_names: List of class names corresponding to indices
    """
    ap_scores = []
    
    for i, class_name in enumerate(class_names):
        ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        ap_scores.append(ap)
    
    # Plot Average Precision scores
    plt.figure(figsize=(4, 3))
    plt.bar(class_names, ap_scores, color='skyblue')
    plt.axhline(y=np.mean(ap_scores), color='r', linestyle='-', label = "Mean average precision")
    plt.xlabel('Class')
    plt.ylabel('Average Precision')
    plt.title('Average Precision for Multi-label Classification')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()