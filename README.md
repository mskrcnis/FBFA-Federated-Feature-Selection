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

FBFA-Federated-Feature-Selection/
├── run_pipeline.py # Main execution script
├── data/ # Put your CSV files here
├── notebooks/ # For any demo notebooks
├── src/
│ ├── preprocessing.py # Data loading & preprocessing
│ ├── partitioning.py # Dirichlet-based data split
│ ├── model.py # Tabular MLP architecture
│ ├── train.py # Training, evaluation, and aggregation
│ ├── fbfa.py # FBFA federated feature selection
│ └── frhc.py # FRHC-based feature selection


---

## 🚀 Getting Started

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/FBFA-Federated-Feature-Selection.git
cd FBFA-Federated-Feature-Selection

###2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

###3️⃣ Run the pipeline
python run_pipeline.py

Follow on-screen prompts to:
Select your dataset (CSV)
Specify label column name
Choose feature selection mode: full / FBFA / FRHC


📊 Supported Feature Selection Modes
Full Feature Set: Uses all available features (baseline)
FBFA: Uses federated Binary Firefly Optimization across clients
FRHC: Uses local clustering + global intersection for representative feature selection

📘 Citation
If you use this work in your research, please cite:
S. Kumar Reddy Mallidi et al., "Distributed Federated Feature Selection via Adaptive Consensus and Binary Firefly Algorithm for Large-Scale IoT Network Intrusion Detection", 2025.

📬 Contact
For questions or collaborations, reach out at: satya.cnis@gmail.com
