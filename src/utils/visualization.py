import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve


def plot_training_history(
    ssl_loss_history: Optional[List[float]] = None,
    train_loss_history: Optional[List[float]] = None,
    val_loss_history: Optional[List[float]] = None,
    train_acc_history: Optional[List[float]] = None,
    val_acc_history: Optional[List[float]] = None,
    warmup_epochs: int = 5,
    save_path: Optional[str] = None
) -> None:
    """Plots training histories for SSL pretraining and Multi-Task Fine-Tuning."""
    if ssl_loss_history is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(ssl_loss_history, label='NT-Xent Loss', color='teal', linewidth=2)
        plt.title("Self-Supervised Pretraining Loss (SimCLR)", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        if save_path:
            plt.savefig(save_path.replace(".png", "_ssl.png"), bbox_inches='tight', dpi=300)
        plt.show()

    if train_loss_history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Multi-Task Fine-Tuning Training History', fontsize=16, fontweight='bold')

        # Loss Plot
        ax1.plot(train_loss_history, label='Training Loss', color='crimson', linewidth=2)
        if val_loss_history:
            ax1.plot(val_loss_history, label='Validation Loss', color='darkorange', linestyle='--', linewidth=2)
        ax1.set_title("Loss Over Epochs", fontsize=13)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Combined Loss", fontsize=11)
        ax1.axvline(x=warmup_epochs - 1, color='gray', linestyle=':', label='Fine-Tuning Stage 2')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Accuracy Plot
        if train_acc_history:
            ax2.plot(train_acc_history, label='Training Accuracy', color='royalblue', linewidth=2)
        if val_acc_history:
            ax2.plot(val_acc_history, label='Validation Accuracy', color='forestgreen', linestyle='--', linewidth=2)
        ax2.set_title("Accuracy Over Epochs", fontsize=13)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.axvline(x=warmup_epochs - 1, color='gray', linestyle=':', label='Fine-Tuning Stage 2')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def plot_confusion_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None
) -> None:
    """Plots and saves confusion matrix on test set."""
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title("Confusion Matrix on Test Set", fontsize=14, fontweight='bold')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_reliability_diagram(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """Plots calibration reliability diagram comparing confidence vs accuracy."""
    y_true_binary = (all_preds == all_labels)
    pred_confidences = np.max(all_probs, axis=1)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_binary, pred_confidences, n_bins=n_bins, strategy='uniform'
    )
    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model Calibration", color="teal", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    plt.xlabel("Average Confidence (in bin)", fontsize=12)
    plt.ylabel("Accuracy (in bin)", fontsize=12)
    plt.title("Reliability Diagram (Model Calibration)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_roc_curves(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None
) -> None:
    """Computes and plots One-vs-All multi-class ROC-AUC curves."""
    y_true_binarized = label_binarize(all_labels, classes=range(len(classes)))
    n_classes = y_true_binarized.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['teal', 'darkorange', 'dodgerblue', 'crimson'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve for {classes[i]} (AUC = {roc_auc[i]:0.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='No-Skill (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class ROC-AUC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
