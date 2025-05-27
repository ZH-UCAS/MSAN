

# MSAN

This repository provides a complete pipeline for preprocessing, training, and evaluating models on the [CHB-MIT Scalp EEG Database](http://archive.physionet.org/physiobank/database/chbmit/).

## ⚙️ 1. Data Preparation

Please download the CHB-MIT EEG dataset from:

**[Download Link](http://archive.physionet.org/physiobank/database/chbmit/)**

## 🧹 2. Data Preprocessing

* **To preprocess data for a single patient**, run:

  ```bash
  make preprocess
  ```

* **To preprocess data for all patients**, run:

  ```bash
  make preprocess_chb
  ```

## 🧠 3. Model Training

* **To train a model on a single patient's data**, run:

  ```bash
  make train
  ```

* **To train on all patients' data**, run:

  ```bash
  make train_chb
  ```

## 📊 4. Model Evaluation

* **To evaluate a model trained on a single patient**, run:

  ```bash
  make eval
  ```

* **To evaluate models trained on all patients**, run:

  ```bash
  make eval_chb
  ```

## 📚 Citation & Acknowledgment

This project utilizes techniques and concepts described in the following publication. Please cite this work if you use the associated codebase or methodologies:

> Q. Dong, H. Zhang, J. Xiao, and J. Sun, "Multi-Scale Spatio-Temporal Attention Network for Epileptic Seizure Prediction," *IEEE Journal of Biomedical and Health Informatics*, 2025. doi: [10.1109/JBHI.2025.3545265](https://doi.org/10.1109/JBHI.2025.3545265)

**Keywords**: Feature extraction; Electroencephalography; Epilepsy; Seizure prediction; Multi-scale spatio-temporal attention.

## 🔧 Note

If you encounter any missing dependencies or configuration issues, please don’t hesitate to contact me.

