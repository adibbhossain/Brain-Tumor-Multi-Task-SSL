"""
Phase 2: Contrastive Self-Supervised Pre-training (SimCLR)
Usage:
    python train_ssl.py --config configs/default_config.yaml
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightly.loss import NTXentLoss
from tqdm.auto import tqdm

from src.utils.helpers import load_config, set_seed, get_device, ensure_dir
from src.dataset import prepare_dataloaders
from src.models.simclr import SimCLRModel


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: SimCLR Contrastive Pretraining")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML file")
    parser.add_argument("--epochs", type=int, default=None, help="Override SSL epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Override encoder output checkpoint path")
    return parser.parse_args()


def train_ssl(config: dict):
    set_seed(config["seed"])
    device = get_device(config.get("device", "auto"))
    print(f"Initializing SimCLR Pretraining on device: {device}")

    # Prepare data
    ssl_loader, _, _, _, splits_info = prepare_dataloaders(config)
    print(f"Dataset Size: {splits_info['total']} images | Training Split (SSL): {splits_info['train']} images")

    # Build SimCLR model
    simclr = SimCLRModel(
        pretrained_weights=True,
        input_dim=512,
        hidden_dim=config.get("projection_hidden_dim", 2048),
        output_dim=config.get("projection_output_dim", 128)
    ).to(device)

    # Optimization
    criterion = NTXentLoss(temperature=config.get("temperature", 0.1)).to(device)
    optimizer = torch.optim.AdamW(simclr.parameters(), lr=config["ssl_lr"], weight_decay=1e-4)
    epochs = config["ssl_epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    accum_steps = config.get("accum_steps", 4)
    encoder_path = config["encoder_path"]
    ensure_dir(encoder_path)

    ssl_loss_history = []
    start_time = time.time()

    print("\n" + "="*60)
    print("PHASE 2: SELF-SUPERVISED PRETRAINING (SimCLR)")
    print("="*60)

    for epoch in range(epochs):
        simclr.train()
        total_loss = 0.0
        progress_bar = tqdm(ssl_loader, desc=f"SSL Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, (views, _) in enumerate(progress_bar):
            v1, v2 = views
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            if device.type == 'cuda':
                with autocast('cuda'):
                    z1 = simclr(v1)
                    z2 = simclr(v2)
            else:
                z1 = simclr(v1)
                z2 = simclr(v2)

            z1 = F.normalize(z1.float(), dim=1)
            z2 = F.normalize(z2.float(), dim=1)
            loss = criterion(z1, z2) / accum_steps

            if scaler:
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix({'loss': f'{(loss.item() * accum_steps):.4f}'})
            total_loss += loss.item() * accum_steps

        # Catch remaining gradients
        if (step + 1) % accum_steps != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        avg_loss = total_loss / len(ssl_loader)
        ssl_loss_history.append(avg_loss)
        print(f"SSL Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Sanity checks
    simclr.eval()
    with torch.no_grad():
        (views, _) = next(iter(ssl_loader))
        v1, v2 = views
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        z1 = simclr(v1)
        z2 = simclr(v2)
        z1_norm = F.normalize(z1.float(), dim=1)
        z2_norm = F.normalize(z2.float(), dim=1)
        id_loss = criterion(z1_norm, z1_norm).item()
        pos_sim = (z1_norm * z2_norm).sum(dim=1).mean().item()
        sim_matrix = z1_norm @ z2_norm.T
        sim_matrix.fill_diagonal_(0.0)
        neg_sim = sim_matrix.mean().item()

    print("\nSanity Check Results:")
    print(f"Identity Loss (Self vs Self): {id_loss:.4f}")
    print(f"Positive Pair Similarity:    {pos_sim:.3f}")
    print(f"Negative Pair Similarity:    {neg_sim:.3f}")

    elapsed_min = (time.time() - start_time) / 60
    print(f"\nPretraining completed in {elapsed_min:.2f} minutes.")

    torch.save(simclr.get_encoder().state_dict(), encoder_path)
    print(f"Pretrained backbone saved to '{encoder_path}'")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs: config["ssl_epochs"] = args.epochs
    if args.lr: config["ssl_lr"] = args.lr
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.output_dir: config["encoder_path"] = args.output_dir
    train_ssl(config)

if __name__ == "__main__":
    main()
