import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics.classification import MulticlassCalibrationError


def tta_predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Applies Test-Time Augmentation (TTA) using safe affine flips
    and returns averaged classification logits.
    """
    views = [
        lambda t: t,
        lambda t: torch.flip(t, dims=[-1]),  # Horizontal flip
        lambda t: torch.flip(t, dims=[-2]),  # Vertical flip
    ]
    logits = []
    with torch.inference_mode():
        for f in views:
            lg, _ = model(f(x))
            logits.append(lg)
    return torch.stack(logits, dim=0).mean(0)


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[torch.nn.Parameter, float]:
    """
    Finds optimal temperature parameter T on validation set using L-BFGS optimizer
    to minimize Negative Log-Likelihood (Cross-Entropy).
    """
    model.eval()
    T = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            lg, _ = model(x)
            val_logits_list.append(lg)
            val_labels_list.append(y)

    val_logits = torch.cat(val_logits_list)
    val_labels = torch.cat(val_labels_list)

    def _closure():
        optimizer.zero_grad()
        scaled = val_logits / T.clamp_min(1e-3)
        loss = F.cross_entropy(scaled, val_labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return T, T.item()


def compute_ece(all_probs: np.ndarray, all_labels: np.ndarray, num_classes: int = 4, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) via TorchMetrics MulticlassCalibrationError."""
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm='l1')
    ece_metric.update(torch.from_numpy(all_probs), torch.from_numpy(all_labels))
    return float(ece_metric.compute().item())


def compute_classification_report(all_labels: np.ndarray, all_preds: np.ndarray, classes: list) -> str:
    """Generates standard sklearn classification performance report."""
    return classification_report(all_labels, all_preds, target_names=classes, digits=4)
