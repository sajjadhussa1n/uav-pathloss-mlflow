# UAV Pathloss Prediction with MSFF-UNet and MLflow

This repository provides code and resources for training and evaluating a **Multi-Scale Feature Fusion U-Net (MSFF-UNet)** model for UAV-assisted mmWave communications.  
It includes:
- Model architecture implementation in TensorFlow/Keras.
- Dataset preparation and loading utilities.
- Training and evaluation pipeline with MLflow integration.
- Pre-trained weights and processed dataset for reproducibility.

---

## Overview

Accurate pathloss prediction is critical for UAV-assisted wireless communication systems.  
We propose a **UNet-based architecture with multi-scale feature fusion (MSFF-UNet)** to predict pathloss using 3D geometry, line-of-sight (LOS) masks, and building layout information.  

<p align="center">
  <img src="docs/architecture.png" alt="Model Architecture" width="600"/>
</p>

---

## Dataset

- The dataset is made publicly available via **Google Drive**.  
- It contains pre-processed CSV files for **training** and **testing**, ready to be loaded by our dataset class.  

To download automatically, see instructions below.  

### Overview
This dataset contains UAV-assisted mmWave path loss simulated across **five diverse urban environments**:
- Munich-01
- Munich-02
- Helsinki
- Manhattan
- London

For each environment, ray-traced simulations were performed at:
- **4 UAV transmitter (TX) locations**
- **3 UAV altitudes**: 25m, 35m, and 45m

Each CSV file corresponds to a unique combination of environment, TX location, and altitude.


### File Naming Convention
Files follow the format:

averaged_path_loss_dataset_{CITY}_tx_loc_{NUMBER}{ALTITUDE_CODE}


- **CITY**: {munich01, munich02, helsinki, manhattan, london}  
- **NUMBER**: {1, 2, 3, 4} → TX location index  
- **ALTITUDE_CODE**:  
  - a = 25m altitude  
  - b = 35m altitude  
  - c = 45m altitude  

**Example**:  
`averaged_path_loss_dataset_london_tx_loc_1a` → London, TX Location 1, altitude 25m


### Dataset Columns
Each CSV file contains the following columns:

- `RX_X` : Receiver X-coordinate (meters)  
- `RX_Y` : Receiver Y-coordinate (meters)  
- `TX_X` : Transmitter X-coordinate (meters)  
- `TX_Y` : Transmitter Y-coordinate (meters)  
- `TX_Z` : Transmitter altitude (meters)  
- `Phi` : Azimuth angle between TX and RX (degrees)  
- `Distance_3d` : 3D distance between TX and RX (meters)  
- `LOS_mask` : Line-of-sight indicator (1 = LOS, 0 = NLOS)  
- `Is_building` : Building penetration indicator (1 = RX inside building, 0 = outdoors)  
- `Path_loss` : Averaged path loss (dB)  


---


## Getting Started

You can run the pipeline either:
- On **Google Colab** (quick start, recommended for testing), or
- Locally via **CLI** (VS Code, terminal, etc.).

---

### 1. Clone Repository

```bash
!git clone https://github.com/sajjadhussa1n/uav-pathloss-mlflow.git
%cd uav-pathloss-mlflow
```
### 2. Download Dataset and Pre-trained Weights

The dataset used in this research and the best model weights are publicly available on Google Drive. We use gdown to fetch public Google Drive folders.

```bash
!pip install gdown
```

Download the dataset in the `./dataset` directory. It is also recommended to download the dataset manually from this [link](https://drive.google.com/drive/folders/1ooH4jxP_qk3OriNYzt8vPr-waYiR6qy0?usp=sharing) in case the following bash commands fail. 

```bash
# Dataset Folder ID
folder_id="1ooH4jxP_qk3OriNYzt8vPr-waYiR6qy0"
target_dir="./dataset"

# Download all dataset files
!gdown --folder https://drive.google.com/drive/folders/$folder_id -O $target_dir
```

Download the pre-trained model weights in the `./artifacts` directory. Again, you can manually download the pre-trained model weights from this [link](https://drive.google.com/drive/folders/1g7PgvSqooMttOzmRsplTDocRMLeKdOZS?usp=sharing).

```bash
# Pretrained weights Folder ID
folder_id="1g7PgvSqooMttOzmRsplTDocRMLeKdOZS"
target_dir="./artifacts"

# Download pretrained weights
!gdown --folder https://drive.google.com/drive/folders/$folder_id -O $target_dir
```

### 3. Install Dependencies

```bash
!pip install -r requirements.txt
```

If Colab asks to restart the runtime after install, please restart and make sure you are in the root directory.

```bash
%cd uav-pathloss-mlflow
```

### 4. Run Training and Evaluation

```bash
!python main.py
```
This script will automatically create the model instance, prepare train and test datasets, train the model, and evaluate it. The results will be reported in the CLI. This code also uses MLFlow framework to automatically track experiments and log results in ./mlruns directory. 

### 5. Adjust Training Parameters

Modify configs/config.yaml to change training parameters including epochs, learning rate, batch size etc.

---

### 📑 Citation 

If you use this repository or dataset, please cite our work:

```bibtex
@article{hussain2025uavpathloss,
  author={S. Hussain},
  title={A Multi-Scale Feature Extraction and Fusion UNet for Pathloss Prediction in UAV-Assisted mmWave Radio Networks},
  journal={IEEE Transactions on Wireless Communications},
  year={2025},
  note={Submitted, September 2025}
}
```
---

### 📬 Contact 

For questions or collaborations, feel free to reach out:

- 📧 [Sajjad Hussain](https://github.com/sajjadhussa1n) sajjad.hussain2@seecs.edu.pk

---

### ⭐ Acknowledgements 

- TensorFlow/Keras for deep learning.

- MLflow for experiment tracking.

- Google Drive for dataset hosting.


