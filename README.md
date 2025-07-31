# Federated Binary Firefly Algorithm (FBFA) for Feature Selection

This repository implements a federated feature selection and classification framework using:
- 🔥 **Federated Binary Firefly Algorithm (FBFA)**
- 🌲 **Federated Representative Hierarchical Clustering (FRHC)**
- 🤖 **Federated Multi-layer Perceptron (MLP)**

It supports client-wise data partitioning and federated learning with feature selection for high-dimensional tabular datasets like IoT-IDS or spam classification.

---

## 🧠 Key Features

- File-based CSV loading with user-selected label column
- Standard preprocessing with label encoding, scaling, and constant feature removal
- Non-IID partitioning using Dirichlet distribution
- Two feature selection modes: FBFA and FRHC
- Federated training using client-local MLPs and model averaging
- Test accuracy and confusion matrix reporting

---

## 📁 Project Structure

```
FBFA-Federated-Feature-Selection/
├── run_pipeline.py         # Main execution script
├── data/                   # Cleaned data files for training/testing
├── notebooks/              # Original raw notebooks and datasets used during development
├── src/
│   ├── preprocessing.py    # Data loading & preprocessing
│   ├── partitioning.py     # Dirichlet-based data partitioning for clients
│   ├── model.py            # Tabular MLP architecture
│   ├── train.py            # Training, evaluation, and federated averaging
│   ├── fbfa.py             # Federated Binary Firefly Algorithm (FBFA)
│   └── frhc.py             # Federated Representative Hierarchical Clustering (FRHC)
```

---

## 🚀 Getting Started

### 1️⃣ Clone this repository

git clone https://github.com/<your-username>/FBFA-Federated-Feature-Selection.git
cd FBFA-Federated-Feature-Selection

### 2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

### 3 Run the pipeline
python run_pipeline.py

Follow on-screen prompts to:
Select your dataset (CSV)
Specify label column name
Choose feature selection mode: full / FBFA / FRHC


📊 Supported Feature Selection Modes
Full Feature Set: Uses all available features (baseline)
FBFA: Uses federated Binary Firefly Optimization across clients
FRHC: Uses local clustering + global intersection for representative feature selection


# Notebooks

This folder contains original Jupyter notebooks and associated datasets used during the development of the FBFA Federated Feature Selection framework.

---

## 📘 Notebooks

| Notebook Name                | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| `spambase-fed-bfa.ipynb`    | Federated BFA testing on the Spambase dataset (binary)       |
| `ACI-fed-bfa-Bin.ipynb`     | FBFA experiment on ACI IoT 2023 dataset (binary classification) |
| `ACI-fed-bfa-Multi.ipynb`   | FBFA on ACI dataset for multiclass classification            |
| `WUSTL-fed-bfa-Bin.ipynb`   | Binary classification using FBFA on WUSTL-IIoT dataset       |
| `WUSTL-fed-bfa-Multi.ipynb` | Multiclass FBFA training on WUSTL-IIoT dataset               |

---

## 📂 Data Files

| CSV File              | Description                        |
|-----------------------|------------------------------------|
| `spambase.csv`        | UCI Spambase dataset               |
| `ACI-IoT-2023.csv`    | ACI IoT Intrusion Detection dataset|
| `wustl_corrected.csv` | Cleaned WUSTL-IIoT dataset         |

---

> These notebooks were essential during the initial experimentation phase. The finalized codebase is modularized under `src/` and controlled via `run_pipeline.py`.



📘 Citation
If you use this work in your research, please cite:
S. Kumar Reddy Mallidi et al., "Distributed Federated Feature Selection via Adaptive Consensus and Binary Firefly Algorithm for Large-Scale IoT Network Intrusion Detection", 2025.

📬 Contact
For questions or collaborations, reach out at: satya.cnis@gmail.com
