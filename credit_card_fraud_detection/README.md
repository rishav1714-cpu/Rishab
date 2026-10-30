# Credit Card Fraud Detection Using Artificial Neural Networks (ANN)

A production-grade, end-to-end Machine Learning and Deep Learning system designed to detect fraudulent credit card transactions in real-time. Built with **TensorFlow / Keras**, **Scikit-learn**, and **Streamlit**, this project addresses the challenges of extreme class imbalance (0.17% fraud prevalence) in high-throughput banking environments.

---

## 1. Project Introduction & Problem Statement

Financial fraud presents severe risks to payment processors, commercial banks, and cardholders globally, resulting in billions of dollars in unauthorized transactions annually. 

Detecting fraudulent transactions introduces three critical machine learning challenges:
1. **Extreme Class Imbalance:** Fraudulent transactions account for only **0.1727%** of the dataset (492 fraud cases out of 284,807 total transactions).
2. **Asymmetric Misclassification Costs:** In financial surveillance, missing a fraud case (**False Negative**) results in direct financial loss and regulatory liability, whereas flagging a genuine cardholder (**False Positive**) causes minor customer friction.
3. **Data Leakage Vulnerability:** Standardizing continuous variables prior to train-test splitting leaks test distribution parameters into training, leading to overly optimistic results that fail in production.

---

## 2. Project Objectives

- Construct a fully modular, reproducible Artificial Neural Network (ANN) trained on real European credit card transactions.
- Eliminate data leakage by fitting preprocessing scalers strictly on training partitions.
- Address severe class imbalance using inverse-frequency class weights ($w_{\text{fraud}} \approx 289$).
- Evaluate performance using non-deceptive metrics (**PR-AUC**, **ROC-AUC**, **Recall**, and **Precision**) rather than raw accuracy.
- Provide interactive decision threshold calibration to allow risk teams to balance fraud capture vs. false alarms.
- Deploy an intuitive, banking-grade Streamlit web dashboard for real-time single-transaction scoring and high-volume batch CSV inference.

---

## 3. Dataset Description

The project uses the benchmark **Kaggle Credit Card Fraud Detection Dataset** (European cardholder transactions):
- **Source:** [Kaggle MLG - ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807
- **Genuine Transactions (Class 0):** 284,315 (99.827%)
- **Fraudulent Transactions (Class 1):** 492 (0.1727%)
- **Features (30 Numerical Inputs):**
  - `Time`: Elapsed seconds between each transaction and the first transaction in the dataset (spans 48 hours / 172,800 seconds).
  - `V1` to `V28`: 28 principal component analysis (PCA) numerical latent representations (transformed for cardholder privacy).
  - `Amount`: Transaction value in Euros (€).
- **Target Variable:** `Class` (0 = Genuine, 1 = Fraudulent).

---

## 4. Technology Stack

- **Deep Learning Framework:** TensorFlow 2.x & Keras 3.x
- **Machine Learning & Preprocessing:** Scikit-learn
- **Data Manipulation:** Pandas, NumPy
- **Interactive Web Application:** Streamlit
- **Visualizations:** Plotly Express & Graph Objects, Matplotlib, Seaborn
- **Serialization:** Joblib

---

## 5. Project Structure

```
credit_card_fraud_detection/
│
├── app.py                      # Interactive Streamlit Financial Dashboard
├── train_model.py              # Modular ANN Training Pipeline
├── preprocessing.py            # Data loading, validation, scaling & splitting
├── evaluate_model.py           # Independent test evaluation & curve generator
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation
│
├── data/
│   ├── creditcard.csv                 # Core Kaggle dataset (150 MB)
│   ├── sample_batch_transactions.csv  # 50 mixed test transactions for batch scoring
│   └── test_data_cache.npz            # Cached held-out test arrays (56,962 rows)
│
├── models/
│   ├── fraud_detection_ann.keras      # Saved trained Keras model (native format)
│   └── scaler.pkl                     # Fitted StandardScaler (Time, Amount)
│
└── outputs/
    ├── confusion_matrix.png           # Formatted Confusion Matrix plot
    ├── training_history.png           # Loss, Accuracy, and PR-AUC training trajectories
    ├── precision_recall_curve.png     # Precision-Recall Curve (AUC = 0.6763)
    ├── roc_curve.png                  # Receiver Operating Characteristic (AUC = 0.9546)
    ├── metrics.json                   # Serialized scalar evaluation metrics
    └── threshold_analysis.csv         # Calibration table across all decision cutoffs
```

---

## 6. Artificial Neural Network Architecture

The model is built as a sequential deep feedforward neural network optimized for tabular anomaly detection:

```
Input Vector (30 Features: Time, V1-V28, Amount)
       │
       ▼
[Dense Layer 1: 64 Neurons, ReLU Activation]
       │
       ▼
[Dropout Regularization: Rate = 0.30]
       │
       ▼
[Dense Layer 2: 32 Neurons, ReLU Activation]
       │
       ▼
[Dropout Regularization: Rate = 0.20]
       │
       ▼
[Dense Layer 3: 16 Neurons, ReLU Activation]
       │
       ▼
[Output Layer: 1 Neuron, Sigmoid Activation]
       │
       ▼
P(Fraud | X) ∈ [0, 1]
```

### Compiler Configuration:
- **Optimizer:** Adam ($\text{learning\_rate} = 0.001$)
- **Loss Function:** Binary Crossentropy
- **Optimization Objective:** Precision-Recall AUC (`val_pr_auc`)
- **Early Stopping:** `patience = 5`, restores best weights automatically
- **Batch Size:** 2,048 (vectorized for efficient tabular mini-batching)
- **Class Weights:** Inverse-frequency class weighting ($w_0 = 0.5009$, $w_1 = 289.33$)

---

## 7. Model Evaluation & Performance

Evaluation was conducted strictly on a held-out test split of **56,962 transactions** (including **98 true fraud incidents**), unseen during training or hyperparameter tuning.

### Key Held-Out Test Metrics:

| Evaluation Metric | Baseline (Random / Dummy) | ANN Model Performance |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.5000 | **0.9546** |
| **PR-AUC (Primary Metric)** | 0.0017 (0.17%) | **0.6763** (397× improvement) |
| **Recall (Detection Rate)** | 0.00% | **89.80%** (88 / 98 Frauds Caught) |
| **Operating Accuracy** | 99.83% | **97.20%** (at 0.50 threshold) |

---

## 8. Why Accuracy Alone is Misleading

In an imbalanced dataset where 99.83% of transactions are legitimate:
- A naive dummy baseline that predicts **Class 0 (Genuine)** for *every single transaction* achieves **99.827% accuracy**.
- However, that model detects **0% of fraudulent attacks**, resulting in catastrophic financial loss for the card issuer.
- Accuracy is dominated by the massive pool of True Negatives. Therefore, financial risk teams rely on **Precision**, **Recall**, and **PR-AUC**, which focus specifically on the minority positive class.

---

## 9. Decision Threshold Calibration

Because class weighting shifts the model's loss landscape toward catching frauds, the raw sigmoid output represents calibrated risk scores. Adjusting the classification threshold enables precise alignment with institutional risk preferences:

| Threshold | Precision | Recall (Caught / Total) | False Positives | Operating Regime |
| :---: | :---: | :---: | :---: | :--- |
| **0.30** | 0.78% | **93.88%** (92 / 98) | 11,749 | Ultra-Conservative (Maximum Detection) |
| **0.50** | 5.26% | **89.80%** (88 / 98) | 1,585 | Default Balanced Surveillance |
| **0.85** | 40.48% | **86.73%** (85 / 98) | 125 | High Precision (Reduced Analyst Workload) |
| **0.90** | **59.57%** | **85.71%** (84 / 98) | **57** | **Recommended Banking Operating Point** |
| **0.95** | **67.77%** | **83.67%** (82 / 98) | **39** | Ultra-Low False Alarm Production Gateway |

---

## 10. Installation & Setup

### Prerequisites
- Python 3.10 – 3.13
- Git

### 1. Clone or Navigate to the Repository
```bash
cd credit_card_fraud_detection
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 11. Execution Instructions

### Step 1: Train the ANN Model
Train the deep neural network and save the fitted scaler, model weights, and training history:
```bash
python train_model.py
```

### Step 2: Evaluate the Model & Generate Graphs
Evaluate on the held-out test dataset and produce all diagnostic graphs in `outputs/`:
```bash
python evaluate_model.py
```

### Step 3: Launch the Streamlit Financial Dashboard
Launch the interactive web platform:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 12. Streamlit Dashboard Features

The dashboard is structured into 5 core modules:
1. **🏠 Home:** Executive overview, key dataset KPIs, architecture specifications, and workflow.
2. **📊 Data Exploration:** Ingest CSV or default data, inspect missing values and duplicates, explore interactive distributions (Donut, Boxplot, 24-hour Time Density, Correlation Heatmap).
3. **📈 Model Performance:** Dynamic threshold slider, live updating Confusion Matrix, Precision-Recall & ROC curves, training history viewer, and the Accuracy Paradox analysis.
4. **⚡ Fraud Prediction:**
   - **Single Transaction Inference:** Real-time form with one-click **Genuine** and **Fraudulent** preset buttons, interactive risk gauge, and clear colored warning badges.
   - **Batch CSV Scoring:** Upload multi-row CSV files, execute high-throughput vectorized scoring, view risk score histograms, and download annotated CSV results.
5. **📘 About the Project:** Deep technical breakdown of objectives, methodology, network architecture, and compliance guidelines.

---

## 13. Limitations & Future Scope

### Limitations
- **PCA Anonymization:** Raw transactional features (e.g., merchant ID, merchant category code (MCC), cardholder location, IP address) are masked by PCA, restricting domain-specific feature engineering.
- **Concept Drift:** Fraudsters continuously evolve tactics (e.g., changing ticket sizes, switching from physical card-present to card-not-present fraud). Static models require automated re-training pipelines.
- **Time Window:** The dataset spans a 48-hour window; seasonal, month-end, and holiday behaviors are not captured.

### Future Scope
- **Graph Neural Networks (GNNs):** Model complex cardholder-merchant interaction graphs to identify coordinated syndicate fraud rings.
- **Autoencoder Hybridization:** Combine supervised ANNs with unsupervised autoencoder reconstruction error for zero-day fraud pattern detection.
- **Streaming Pipeline Integration:** Deploy using Apache Kafka and FastAPI to achieve sub-10ms inference per payment payload.
- **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) to provide compliance-compliant feature attribution on every rejected transaction.

---

## 14. Regulatory & Experimental Notice

> **⚠️ Disclaimer:** The models, risk scores, and classifications generated by this application are experimental and developed for academic and engineering demonstration purposes. They are not a substitute for comprehensive bank fraud investigation, anti-money laundering (AML) protocols, or professional compliance scrutiny.
