# A Multi-Task Framework for Brain Tumor Classification with Uncertainty Estimation Using Self-Supervised Learning

PyTorch implementation of **"A Multi-Task Framework for Brain Tumor Classification with Uncertainty Estimation Using Self-Supervised Learning"** (Accepted for publication in a Taylor & Francis Book Chapter, 2025).

> **Authors:** Md. Adib Hossain, Md. Mehedi Hasan*, Md. Shehabub Mobin Siam, Mohiuddin Showrov, Md. Zahid Hasan.

---

## Overview

This repository contains the official implementation of our proposed deep learning framework for brain MRI scan analysis. The workflow consists of:
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

Clone the repository and install dependencies:

```bash
git clone https://github.com/adibbhossain/Brain-Tumor-Multi-Task-SSL.git
cd Brain-Tumor-Multi-Task-SSL
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

If you find this codebase or paper useful for your research, please cite:

```bibtex
@incollection{hossain2025multitask,
  title={A Multi-Task Framework for Brain Tumor Classification with Uncertainty Estimation Using Self-Supervised Learning},
  author={Hossain, Md. Adib and Hasan, Md. Mehedi and Siam, Md. Shehabub Mobin and Showrov, Mohiuddin and Hasan, Md. Zahid},
  booktitle={Taylor \& Francis Book Chapter},
  year={2025},
  publisher={Taylor \& Francis},
  note={Accepted for publication}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
