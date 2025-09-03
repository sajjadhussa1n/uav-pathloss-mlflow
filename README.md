# UAV Pathloss Prediction with MSFF-UNet and MLflow

This repository provides code and resources for training and evaluating a **Multi-Scale Feature Fusion U-Net (MSFF-UNet)** model for UAV-assisted mmWave communications.  
It includes:
- Model architecture implementation in TensorFlow/Keras.
- Dataset preparation and loading utilities.
- Training and evaluation pipeline with MLflow integration.
- Pre-trained weights and processed dataset for reproducibility.

---

## 📖 Overview

Accurate pathloss prediction is critical for UAV-assisted wireless communication systems.  
We propose a **UNet-based architecture with multi-scale feature fusion (MSFF-UNet)** to predict pathloss using 3D geometry, line-of-sight (LOS) masks, and building layout information.  

<p align="center">
  <img src="docs/architecture.png" alt="Model Architecture" width="600"/>
</p>

---

## 📂 Dataset

- The dataset is made publicly available via **Google Drive**.  
- It contains pre-processed CSV files for **training** and **testing**, ready to be loaded by our dataset class.  

To download automatically, see instructions below.  

---

## 🚀 Getting Started

You can run the pipeline either:
- On **Google Colab** (quick start, recommended for testing), or
- Locally via **CLI** (VS Code, terminal, etc.).

---

### 1. Clone Repository

```bash
git clone https://github.com/sajjadhussa1n/uav-pathloss-mlflow.git
cd uav-pathloss-mlflow
