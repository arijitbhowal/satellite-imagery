# UNet-LSTM Model for River Dataset and Sentinel-2 NDWI Extraction

This repository implements a pipeline for river segmentation using UNet-LSTM and includes a script to export Sentinel-2 NDWI masks and RGB + NIR imagery for a defined Area of Interest (AOI).

---
## Guide
1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup Training and Inference](#setup-training-and-inference)
4. [Results](#results)
5. [Links](#links)
6. [Acknowledgments](#acknowledgments)
---

## Introduction
This repository provides a full pipeline for river segmentation using a combination of UNet and LSTM networks. It also includes scripts to extract NDWI from Sentinel-2 imagery and generate necessary masks and image patches for training.

## Features

1. **Sentinel-2 NDWI Extraction**:
   - Define an AOI using geographical coordinates.
   - Mask clouds in Sentinel-2 imagery using the QA60 band.
   - Compute the Normalized Difference Water Index (NDWI)
   - Export NDWI binary masks and RGB + NIR images for each month over a specified time range.
   
     ### Example Code
     For hands-on implementation and further customization, you can refer to this [Google Earth Engine script](https://code.earthengine.google.com/3cce22b6f8ff4aa0aafe1fb496f98d71).

2. **River Dataset Segmentation**:
   - Preprocessing river datasets by generating `rivers1.tif`, `rivers2.tif`, and `change.tif`.
   - Sliding window patch generation for training and validation.
   - Training a UNet-LSTM model for segmentation tasks.
   - Metrics evaluation and results visualization.
---

## Setup Training and Inference

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/UNet-LSTM-River-Dataset.git](https://github.com/arijitbhowal/satellite-imagery.git
cd satellite-imagery
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Preprocess Dataset
```bash
python3 process_river_dataset.py
```
This script processes the dataset and creates:
 - `rivers1.tif` and `rivers2.tif` in the `rivers/` directory
 - `change.tif` in the `change/` directory.
### 4. Split the Dataset
```bash
python3 Fsplit.py
```
### 5. Patch Extraction
Use a sliding window technique to extract training and validation patches:
```bash
python3 create_xys.py
```
Outputs:
- `xys/myxys_train.csv`: Contains training patch coordinates.
- `xys/myxys_val.csv`: Contains validation patch coordinates.

### 6. Training the model
To train the UNet-LSTM model, run:
```bash
python3 main.py --Fsplit /Fsplit/ --xys /xys/ --patch_size 32 --nb_dates 19
```
Outputs:
- Trained model after each epoch saved in the `models/` directory.

### 7. Inference
The `inf.py` script is used to perform inference using the trained UNet-LSTM model. It takes input images and generates segmentation masks or change detection outputs. This script is useful for deploying the model to test on new datasets or unseen regions
```bash
python3 inf.py

```
## Results
![Screenshot 2024-11-20 122241](https://github.com/user-attachments/assets/211bb20b-da4b-461e-911d-0635a88daa28)
![Screenshot 2024-11-20 122753](https://github.com/user-attachments/assets/cbc58fbe-c328-499a-87ec-5b1d53261949)
![Screenshot 2024-11-20 122831](https://github.com/user-attachments/assets/a1df1b8c-51b5-4675-98c7-acc8b432d3fa)

## Links
**Dataset Link**:https://www.kaggle.com/datasets/arijitbhowal/river-dataset

**Trained Model**:https://www.kaggle.com/models/arijitbhowal/unet_lstm

## Acknowledgments

This project is inspired by the work presented in [multi-task-L-UNet](https://github.com/mpapadomanolaki/multi-task-L-UNet/tree/main) and the paper *"A Deep Multi-Task Learning Framework Coupling Semantic Segmentation and Fully Convolutional LSTM Networks for Urban Change Detection"* (https://ieeexplore.ieee.org/abstract/document/9352207). The model architecture and data processing steps were adapted from these sources.

We would like to thank the authors for making their code available and for the insightful paper that guided our approach.
