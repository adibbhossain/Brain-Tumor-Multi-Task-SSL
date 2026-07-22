"""
Phase 3: Multi-Task Fine-Tuning (Classification + Confidence/Uncertainty Learning)
Usage:
    python train_multitask.py --config configs/default_config.yaml
"""

import argparse
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.optim import Adam
from torch.amp import GradScaler, autocast

from src.utils.helpers import load_config, set_seed, get_device, ensure_dir
from src.dataset import prepare_dataloaders
from src.models.multitask import MultiTaskTumorModel


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Multi-Task Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--encoder-path", type=str, default=None, help="Path to pretrained encoder checkpoint")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Stage 1 head warmup epochs")
    parser.add_argument("--finetune-epochs", type=int, default=None, help="Stage 2 end-to-end finetuning epochs")
    return parser.parse_args()


def train_one_epoch(model, loader, class_criterion, confidence_criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if device.type == 'cuda':
            with autocast('cuda'):
                class_logits, confidence_logit = model(images)
                class_loss = class_criterion(class_logits, labels)
                preds = torch.argmax(class_logits, dim=1)
                is_correct = (preds == labels).float()
                confidence_loss = confidence_criterion(confidence_logit, is_correct)
                loss = class_loss + confidence_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            class_logits, confidence_logit = model(images)
            class_loss = class_criterion(class_logits, labels)
            preds = torch.argmax(class_logits, dim=1)
            is_correct = (preds == labels).float()
            confidence_loss = confidence_criterion(confidence_logit, is_correct)
            loss = class_loss + confidence_loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_correct += is_correct.sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, class_criterion, confidence_criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            class_logits, confidence_logit = model(images)

            class_loss = class_criterion(class_logits, labels)
            preds = torch.argmax(class_logits, dim=1)
            is_correct = (preds == labels).float()
            confidence_loss = confidence_criterion(confidence_logit, is_correct)
            loss = class_loss + confidence_loss

            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def train_multitask(config: dict):
    set_seed(config["seed"])
    device = get_device(config.get("device", "auto"))
    print(f"Initializing Multi-Task Fine-Tuning on device: {device}")

    _, train_loader, val_loader, _, _ = prepare_dataloaders(config)

    # Initialize backbone and load pretrained weights
    encoder_path = config["encoder_path"]
    base_encoder = resnet18(weights=None)
    base_encoder.fc = nn.Identity()

    if os.path.exists(encoder_path):
        print(f"Loading SSL Pretrained Encoder weights from '{encoder_path}'")
        base_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    else:
        print(f"Pretrained encoder not found at '{encoder_path}'. Training from scratch!")

    model = MultiTaskTumorModel(pretrained_encoder=base_encoder, num_classes=len(config["classes"])).to(device)

    class_criterion = nn.CrossEntropyLoss()
    confidence_criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    final_model_path = config["final_model_path"]
    ensure_dir(final_model_path)
    best_val_acc = 0.0

    print("\n" + "="*60)
    print("PHASE 3: MULTI-TASK FINE-TUNING")
    print("="*60)

    # --- Stage 1: Head Warm-up ---
    warmup_epochs = config.get("warmup_epochs", 5)
    print(f"\n--- Stage 1: Head Warm-up ({warmup_epochs} Epochs) ---")
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = Adam(model.parameters(), lr=config.get("warmup_lr", 1e-3))
    for epoch in range(warmup_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, class_criterion, confidence_criterion, optimizer, scaler, device)
        v_loss, v_acc = evaluate(model, val_loader, class_criterion, confidence_criterion, device)
        print(f"Warm-up Epoch {epoch+1}/{warmup_epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")

    # --- Stage 2: Full Fine-Tuning ---
    finetune_epochs = config.get("finetune_epochs", 35)
    print(f"\n--- Stage 2: Full End-to-End Fine-Tuning ({finetune_epochs} Epochs) ---")
    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer = Adam(model.parameters(), lr=config.get("finetune_lr", 1e-4))
    for epoch in range(finetune_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, class_criterion, confidence_criterion, optimizer, scaler, device)
        v_loss, v_acc = evaluate(model, val_loader, class_criterion, confidence_criterion, device)

        is_best = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), final_model_path)
            is_best = f" - Best model saved! ({best_val_acc:.2f}%)"

        print(f"Finetune Epoch {epoch+1}/{finetune_epochs} | Train Acc: {tr_acc:.2f}% | Val Acc: {v_acc:.2f}%{is_best}")

    print(f"\n Multi-Task Fine-Tuning complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved final classifier model to '{final_model_path}'")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.encoder_path: config["encoder_path"] = args.encoder_path
    if args.warmup_epochs: config["warmup_epochs"] = args.warmup_epochs
    if args.finetune_epochs: config["finetune_epochs"] = args.finetune_epochs
    train_multitask(config)

if __name__ == "__main__":
    main()
