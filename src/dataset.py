import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


class BrainMRIDataset(Dataset):
    """Custom PyTorch dataset for loading and transforming brain MRI images."""
    def __init__(self, image_paths: list, labels: list, transform=None, is_ssl: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_ssl = is_ssl

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        label = self.labels[idx]

        if self.is_ssl:
            view1 = self.transform(image=image)['image']
            view2 = self.transform(image=image)['image']
            return (view1, view2), label
        else:
            if self.transform:
                image = self.transform(image=image)['image']
            return image, label


def make_random_resized_crop(img_size: int, scale=(0.7, 1.0)):
    """
    Tries several RandomResizedCrop signatures to handle different Albumentations versions.
    Falls back to Resize+RandomCrop if none work.
    """
    trials = [
        {"size": img_size, "scale": scale},                    # v2 style (int)
        {"size": (img_size, img_size), "scale": scale},        # v2 style (tuple)
        {"height": img_size, "width": img_size, "scale": scale},  # v1 with scale
        {"size": img_size},
        {"size": (img_size, img_size)},
        {"height": img_size, "width": img_size},
    ]
    for kwargs in trials:
        try:
            return A.RandomResizedCrop(**kwargs)
        except Exception:
            continue
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.RandomCrop(height=img_size, width=img_size),
    ])


def get_transforms(img_size: int = 256):
    """Returns contrastive (SSL) and validation/test transformation pipelines."""
    contrastive_transform = A.Compose([
        make_random_resized_crop(img_size, scale=(0.7, 1.0)),
        A.Rotate(limit=15, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(11, 23), sigma_limit=(0.1, 2.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return contrastive_transform, val_test_transform


def load_dataset_paths(data_path: str, classes: List[str], raw_prefix: str = "512") -> Tuple[List[str], List[int]]:
    """Scans dataset directory for images matching target class subfolders."""
    raw_classes = [f"{raw_prefix}{cls}" if raw_prefix and not cls.startswith(raw_prefix) else cls for cls in classes]
    image_paths, labels = [], []
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    for i, cls_folder in enumerate(raw_classes):
        cls_path = os.path.join(data_path, cls_folder)
        if not os.path.isdir(cls_path):
            # Try without prefix as fallback
            cls_path = os.path.join(data_path, classes[i])
        
        if os.path.isdir(cls_path):
            for img_name in os.listdir(cls_path):
                if os.path.splitext(img_name)[1].lower() in exts:
                    image_paths.append(os.path.join(cls_path, img_name))
                    labels.append(i)

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No images found in '{data_path}'. Please check directory structure or configs."
        )

    return image_paths, labels


def prepare_dataloaders(config: dict):
    """Loads image paths, performs stratified train/val/test split, and constructs DataLoaders."""
    data_path = config["data_path"]
    classes = config["classes"]
    raw_prefix = config.get("raw_prefix", "512")
    seed = config["seed"]
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 2)
    pin_memory = config.get("pin_memory", True)
    img_size = config["img_size"]

    image_paths, labels = load_dataset_paths(data_path, classes, raw_prefix)

    # Perform stratified splits
    test_size = config.get("test_size", 0.15)
    val_size_ratio = config.get("val_size_of_train_val", 0.17647)

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size_ratio, random_state=seed, stratify=train_val_labels
    )

    contrastive_transform, val_test_transform = get_transforms(img_size)

    ssl_dataset = BrainMRIDataset(train_paths, train_labels, transform=contrastive_transform, is_ssl=True)
    train_dataset = BrainMRIDataset(train_paths, train_labels, transform=contrastive_transform, is_ssl=False)
    val_dataset = BrainMRIDataset(val_paths, val_labels, transform=val_test_transform, is_ssl=False)
    test_dataset = BrainMRIDataset(test_paths, test_labels, transform=val_test_transform, is_ssl=False)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0 and os.name != 'nt':
        loader_kwargs['persistent_workers'] = True

    ssl_loader = DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    splits_info = {
        "total": len(image_paths),
        "train": len(train_paths),
        "val": len(val_paths),
        "test": len(test_paths),
        "train_paths": train_paths, "train_labels": train_labels,
        "val_paths": val_paths, "val_labels": val_labels,
        "test_paths": test_paths, "test_labels": test_labels,
        "all_paths": image_paths, "all_labels": labels
    }

    return ssl_loader, train_loader, val_loader, test_loader, splits_info
