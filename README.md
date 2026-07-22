# Multi-Task Self-Supervised Learning for Brain MRI Classification and Uncertainty Estimation

PyTorch implementation of Self-Supervised Learning (SimCLR) and Multi-Task Fine-Tuning for Brain Tumor Classification and Uncertainty Estimation on MRI scans.

## Overview

This repository contains the implementation of the methodology described in our paper. The framework consists of:
1. **Self-Supervised Pretraining (SimCLR)**: Pretraining a ResNet-18 backbone using NT-Xent contrastive loss on MRI images.
2. **Multi-Task Fine-Tuning**: Jointly training a 4-class classification head (Glioma, Meningioma, Pituitary, Normal) and a confidence/uncertainty estimation head.
3. **Model Calibration & Evaluation**: Post-hoc Temperature Scaling, Expected Calibration Error (ECE) measurement, and Test-Time Augmentation (TTA).

---

## Repository Structure

```
.
├── configs/
│   └── default_config.yaml      # Hyperparameters and path configurations
├── src/
│   ├── dataset.py               # Dataset loader, Albumentations augmentations, and splits
│   ├── models/
│   │   ├── simclr.py            # SimCLR encoder architecture
│   │   └── multitask.py         # Multi-task model architecture
│   └── utils/
│       ├── helpers.py           # Utility functions and seed setting
│       ├── metrics.py           # Evaluation metrics, TTA, and calibration
│       └── visualization.py     # Plotting utilities
├── notebooks/
│   └── multi-task-mri-ssl.ipynb # Standalone execution notebook
├── train_ssl.py                 # Self-supervised pretraining script
├── train_multitask.py           # Multi-task fine-tuning script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Dependency requirements
└── LICENSE                      # License file
```

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Multi_Task_MRI_SSL.git
cd Multi_Task_MRI_SSL
pip install -r requirements.txt
```

---

## Dataset Preparation

This implementation evaluates on the **PMRAM Bangladeshi Brain Cancer MRI Dataset**.

1. Download the dataset from Kaggle or your source.
2. Structure the dataset directory as follows:

```
data/
└── Raw Data/
    └── Raw/
        ├── 512Glioma/
        ├── 512Meningioma/
        ├── 512Normal/
        └── 512Pituitary/
```

3. Update the `data_path` field in `configs/default_config.yaml` to point to your local dataset path.

---

## Usage

### 1. Self-Supervised Pretraining (SimCLR)

To pretrain the ResNet-18 encoder using SimCLR contrastive learning:

```bash
python train_ssl.py --config configs/default_config.yaml
```

The pretrained encoder weights will be saved to `./checkpoints/pretrained_encoder.pth`.

### 2. Multi-Task Fine-Tuning

To fine-tune the model with classification and confidence heads:

```bash
python train_multitask.py --config configs/default_config.yaml
```

The fine-tuned model will be saved to `./checkpoints/final_tumor_classifier.pth`.

### 3. Evaluation

To evaluate the model on the test set with TTA and Temperature Scaling calibration:

```bash
python evaluate.py --config configs/default_config.yaml --save-plots
```

---

## Citation

If you find this codebase useful for your research, please cite our paper:

```bibtex
@article{multitask_mri_ssl_2026,
  title={Multi-Task Self-Supervised Learning and Predictive Uncertainty Estimation for Brain Cancer MRI Classification},
  author={Author Names},
  journal={Accepted Publication Venue},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
