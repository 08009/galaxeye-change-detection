# Binary Change Detection on EO-SAR Image Pairs
### GalaxEye Space — AI Research Intern Assignment

## Description
Pixel-level binary change detection using a U-Net with ResNet34 
backbone on co-registered EO and SAR image pairs across multiple 
disaster events. The model takes a 4-channel input (3-channel EO 
+ 1-channel SAR) and produces a binary change mask where 1 = 
Change and 0 = No-Change.

## Requirements
- Python 3.10
- CUDA-compatible GPU recommended
- See requirements.txt for all dependencies

## Environment Setup
    conda create -n galaxeye python=3.10 -y
    conda activate galaxeye
    pip install -r requirements.txt

## Dataset Structure
    data/
    ├── train/
    │   ├── pre-event/
    │   ├── post-event/
    │   └── target/
    ├── val/
    │   └── val/
    │       ├── pre-event/
    │       ├── post-event/
    │       └── target/
    └── test/
        └── test/
            ├── pre-event/
            ├── post-event/
            └── target/

## Training
    python train.py

## Evaluation
    python eval.py

## Model Weights
Download best model checkpoint (93.4 MB):
[best_model.pth](https://drive.google.com/file/d/1KOJinHy1Cx4euezo4AzqHPMsZ-xeb6e5/view?usp=drive_link)

## Results

| Metric    | Validation | Test   |
|-----------|------------|--------|
| IoU       | 0.4935     | 0.1821 |
| Precision | 0.5257     | 0.2430 |
| Recall    | 0.8896     | 0.4211 |
| F1 Score  | 0.6609     | 0.3082 |

## References
- Bandara & Patel (2022) — ChangeFormer
- Chen et al. (2021) — BIT
- Daudt et al. (2018) — Fully Convolutional Siamese Networks
- Fang et al. (2021) — SNUNet-CD
- segmentation-models-pytorch
- albumentations
