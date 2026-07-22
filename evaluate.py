"""
Phase 4: Comprehensive Model Evaluation (TTA, Temperature Calibration, ECE, Confusion Matrix, ROC-AUC)
Usage:
    python evaluate.py --config configs/default_config.yaml --save-plots
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from tqdm.auto import tqdm

from src.utils.helpers import load_config, set_seed, get_device
from src.dataset import prepare_dataloaders
from src.models.multitask import MultiTaskTumorModel
from src.utils.metrics import tta_predict, calibrate_temperature, compute_ece, compute_classification_report
from src.utils.visualization import plot_confusion_matrix, plot_reliability_diagram, plot_roc_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Test Evaluation & Reliability Metrics")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--model-path", type=str, default=None, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation (TTA)")
    parser.add_argument("--save-plots", action="store_true", help="Save figure plots to outputs directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save evaluation plots")
    return parser.parse_args()


def run_evaluation(config: dict, use_tta: bool = True, save_plots: bool = False, output_dir: str = "./outputs"):
    set_seed(config["seed"])
    device = get_device(config.get("device", "auto"))
    print(f"Initializing Evaluation Pipeline on device: {device}")

    _, _, val_loader, test_loader, _ = prepare_dataloaders(config)

    # Load model
    final_model_path = config["final_model_path"]
    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f"Final model checkpoint not found at: {final_model_path}")

    base_encoder = resnet18(weights=None)
    base_encoder.fc = nn.Identity()
    eval_model = MultiTaskTumorModel(pretrained_encoder=base_encoder, num_classes=len(config["classes"]))
    eval_model.load_state_dict(torch.load(final_model_path, map_location=device))
    eval_model.to(device)
    eval_model.eval()
    print(f"Loaded model weights from '{final_model_path}'")

    # Step 1: Temperature Calibration on Validation Set
    print("\n--- Finding Optimal Temperature Parameter (T) on Validation Set ---")
    T_param, temp_val = calibrate_temperature(eval_model, val_loader, device)
    print(f"Optimal Temperature (T): {temp_val:.4f}")

    def apply_temperature(logits: torch.Tensor) -> torch.Tensor:
        return logits / T_param.clamp_min(1e-3)

    # Step 2: Evaluation on Test Set
    all_labels, all_preds, all_probs = [], [], []

    print(f"\n--- Running Test Set Inference (TTA Enabled: {use_tta}) ---")
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)

            if use_tta:
                class_logits = tta_predict(eval_model, images)
            else:
                class_logits, _ = eval_model(images)

            calibrated_logits = apply_temperature(class_logits)
            probs = F.softmax(calibrated_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Performance Report
    classes = config["classes"]
    report_str = compute_classification_report(all_labels, all_preds, classes)

    print("\n" + "="*60)
    print("CLASSIFICATION PERFORMANCE REPORT")
    print("="*60)
    print(report_str)

    # Calibration Metric (ECE)
    ece_val = compute_ece(all_probs, all_labels, num_classes=len(classes), n_bins=config.get("calibration_bins", 15))
    print("="*60)
    print(f"Expected Calibration Error (ECE): {ece_val:.4f}")
    print("="*60)

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        rel_path = os.path.join(output_dir, "reliability_diagram.png")
        roc_path = os.path.join(output_dir, "roc_curves.png")
    else:
        cm_path, rel_path, roc_path = None, None, None

    # Visualization
    plot_confusion_matrix(all_labels, all_preds, classes, save_path=cm_path)
    plot_reliability_diagram(all_labels, all_preds, all_probs, save_path=rel_path)
    plot_roc_curves(all_labels, all_probs, classes, save_path=roc_path)

    if save_plots:
        print(f"\nEvaluation plots saved to directory '{output_dir}'")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.model_path: config["final_model_path"] = args.model_path
    use_tta = not args.no_tta
    run_evaluation(config, use_tta=use_tta, save_plots=args.save_plots, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
