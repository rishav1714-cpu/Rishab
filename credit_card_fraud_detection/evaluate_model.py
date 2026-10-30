"""
Model Evaluation and Diagnostic Module for Credit Card Fraud ANN.

This module evaluates the trained model on an independent, held-out test dataset:
- Calculates: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC.
- Generates and persists:
  1. outputs/confusion_matrix.png
  2. outputs/training_history.png
  3. outputs/precision_recall_curve.png
  4. outputs/roc_curve.png
  5. outputs/metrics.json
  6. outputs/threshold_analysis.csv
- Sweeps decision thresholds (0.01 to 0.99) to show precision vs recall trade-offs.
- Explains why accuracy alone is misleading for imbalanced fraud datasets.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for reliable server execution
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import keras

import preprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
TEST_CACHE_PATH = os.path.join(BASE_DIR, "data", "test_data_cache.npz")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
HISTORY_PATH = os.path.join(OUTPUTS_DIR, "training_history.json")


def load_test_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the held-out test dataset partition.
    Uses cached test file if available, or deterministically reconstructs it.
    """
    if os.path.exists(TEST_CACHE_PATH):
        cached = np.load(TEST_CACHE_PATH)
        return cached["X_test"], cached["y_test"]
    
    # Deterministic reconstruction
    df = preprocessing.load_dataset(DATA_PATH)
    X, y = preprocessing.separate_features_target(df)
    X_train, X_test, y_train, y_test = preprocessing.split_data(X, y, test_size=0.20, random_state=42)
    scaler = preprocessing.load_fitted_scaler(SCALER_PATH)
    _, X_test_scaled, _ = preprocessing.standardize_features(X_train, X_test)
    return X_test_scaled.values.astype(np.float32), y_test.values.astype(np.int32)


def evaluate_at_threshold(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Calculates comprehensive performance metrics at a specific decision threshold.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1_score": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist()
    }


def compute_threshold_curve(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Computes performance metrics across a spectrum of decision thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
        
    records = []
    for th in thresholds:
        metrics = evaluate_at_threshold(y_true, y_pred_proba, threshold=th)
        records.append({
            "threshold": th,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
            "tp": metrics["true_positives"],
            "fp": metrics["false_positives"],
            "tn": metrics["true_negatives"],
            "fn": metrics["false_negatives"]
        })
    return pd.DataFrame(records)


def plot_confusion_matrix(cm: np.ndarray, save_path: str, threshold: float = 0.50):
    """
    Plots and saves styled Confusion Matrix.
    """
    plt.figure(figsize=(7, 5.5), dpi=300)
    labels = ["Genuine (0)", "Fraud (1)"]
    
    # Custom annotations with counts and percentages
    cm_sum = np.sum(cm)
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / cm_sum * 100
            annot[i, j] = f"{count:,}\n({pct:.2f}%)"
            
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1.5,
        linecolor="#E2E8F0",
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix (Decision Threshold = {threshold:.2f})", fontsize=14, pad=15, weight="bold")
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Actual True Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrix to: {save_path}")


def plot_training_history(history_path: str, save_path: str):
    """
    Plots training and validation Loss, Accuracy, and PR-AUC over epochs.
    """
    if not os.path.exists(history_path):
        print(f"[Warning] Training history file not found at {history_path}")
        return
        
    with open(history_path, "r") as f:
        history = json.load(f)
        
    epochs = range(1, len(history["loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # Loss plot
    axes[0].plot(epochs, history["loss"], "o-", color="#2563EB", label="Train Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "s--", color="#DC2626", label="Val Loss", linewidth=2)
    axes[0].set_title("Binary Crossentropy Loss", fontsize=13, weight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Accuracy plot
    axes[1].plot(epochs, history["accuracy"], "o-", color="#10B981", label="Train Accuracy", linewidth=2)
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], "s--", color="#F59E0B", label="Val Accuracy", linewidth=2)
    axes[1].set_title("Model Accuracy", fontsize=13, weight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    # PR-AUC plot
    pr_key = "pr_auc" if "pr_auc" in history else [k for k in history.keys() if "pr" in k or "auc" in k][0]
    val_pr_key = f"val_{pr_key}"
    axes[2].plot(epochs, history[pr_key], "o-", color="#8B5CF6", label="Train PR-AUC", linewidth=2)
    if val_pr_key in history:
        axes[2].plot(epochs, history[val_pr_key], "s--", color="#EC4899", label="Val PR-AUC", linewidth=2)
    axes[2].set_title("Precision-Recall AUC (Primary Metric)", fontsize=13, weight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("PR-AUC", fontsize=11)
    axes[2].legend(frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("ANN Model Training & Validation Trajectory", fontsize=16, weight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved Training History to: {save_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    pr_auc: float, 
    save_path: str
):
    """
    Plots styled Precision-Recall curve with baseline reference.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    baseline = np.sum(y_true) / len(y_true)
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall, precision, color="#1E40AF", lw=2.5, label=f"ANN PR Curve (AUC = {pr_auc:.4f})")
    plt.axhline(y=baseline, color="#DC2626", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline*100:.2f}%)")
    
    plt.title("Precision-Recall (PR) Curve", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Recall (Sensitivity / Detection Rate)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved PR Curve to: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    roc_auc: float, 
    save_path: str
):
    """
    Plots styled Receiver Operating Characteristic (ROC) curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color="#059669", lw=2.5, label=f"ANN ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, weight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.xlim([-0.01, 1.02])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved ROC Curve to: {save_path}")


def get_misleading_accuracy_explanation() -> str:
    """
    Educational explanation of the accuracy paradox in extreme imbalanced scenarios.
    """
    return (
        "### The Accuracy Paradox in Fraud Detection\n\n"
        "In the Kaggle European credit card fraud dataset, **99.83%** of all transactions are genuine "
        "and only **0.17%** are fraudulent (492 out of 284,807 transactions).\n\n"
        "If a naive baseline model simply classifies *every single transaction as Genuine (0)*, it achieves an "
        "astonishing **99.83% accuracy** without identifying even one fraudulent transaction. In a banking scenario, "
        "such a model would be catastrophic, resulting in 100% financial loss from undetected frauds.\n\n"
        "Therefore, **Accuracy is fundamentally misleading** for highly imbalanced anomaly detection. "
        "Production financial institutions prioritize:\n"
        "- **Recall (Sensitivity):** The fraction of total fraudulent attempts caught.\n"
        "- **Precision:** The probability that a flagged transaction is genuinely fraudulent (reducing false alerts).\n"
        "- **PR-AUC (Precision-Recall Area Under Curve):** Evaluates precision vs recall trade-offs across all operating thresholds, unaffected by massive true negative volumes.\n"
        "- **F1-Score:** Harmonic mean of precision and recall."
    )


def run_full_evaluation(threshold: float = 0.50) -> Dict[str, Any]:
    """
    Runs the complete evaluation suite on the held-out test dataset:
    - Calculates metrics at the target threshold.
    - Generates all 4 plot figures.
    - Produces threshold analysis curve table.
    - Saves metrics.json.
    """
    print("=" * 60)
    print(f"Running Full Model Evaluation (Decision Threshold: {threshold:.2f})")
    print("=" * 60)
    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # 1. Load test data and model
    X_test, y_test = load_test_dataset()
    print(f"Loaded held-out test dataset: {len(y_test):,} samples (Fraud cases: {int(np.sum(y_test))})")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}. Train the model first.")
        
    model = keras.models.load_model(MODEL_PATH)
    print("Loaded ANN model successfully.")
    
    # 2. Predict probabilities
    y_pred_proba = model.predict(X_test, batch_size=2048).ravel()
    
    # 3. Calculate metrics
    metrics_summary = evaluate_at_threshold(y_test, y_pred_proba, threshold=threshold)
    
    print("\n--- Model Evaluation Summary (Threshold = {:.2f}) ---".format(threshold))
    print(f"Accuracy  : {metrics_summary['accuracy']*100:.3f}%")
    print(f"Precision : {metrics_summary['precision']*100:.3f}%")
    print(f"Recall    : {metrics_summary['recall']*100:.3f}%")
    print(f"F1-Score  : {metrics_summary['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics_summary['roc_auc']:.4f}")
    print(f"PR-AUC    : {metrics_summary['pr_auc']:.4f}")
    print(f"True Positives (Frauds Caught): {metrics_summary['true_positives']} / {int(np.sum(y_test))}")
    print(f"False Negatives (Frauds Missed): {metrics_summary['false_negatives']}")
    print(f"False Positives (False Alarms): {metrics_summary['false_positives']}")
    
    # 4. Generate visualisations
    cm = np.array(metrics_summary["confusion_matrix"])
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path, threshold=threshold)
    
    history_plot_path = os.path.join(OUTPUTS_DIR, "training_history.png")
    plot_training_history(HISTORY_PATH, history_plot_path)
    
    pr_curve_path = os.path.join(OUTPUTS_DIR, "precision_recall_curve.png")
    plot_precision_recall_curve(y_test, y_pred_proba, metrics_summary["pr_auc"], pr_curve_path)
    
    roc_curve_path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
    plot_roc_curve(y_test, y_pred_proba, metrics_summary["roc_auc"], roc_curve_path)
    
    # 5. Threshold trade-off curve
    threshold_df = compute_threshold_curve(y_test, y_pred_proba)
    threshold_csv_path = os.path.join(OUTPUTS_DIR, "threshold_analysis.csv")
    threshold_df.to_csv(threshold_csv_path, index=False)
    print(f"[Evaluation] Saved Threshold Trade-off table to: {threshold_csv_path}")
    
    # 6. Save metrics json
    metrics_json_path = os.path.join(OUTPUTS_DIR, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[Evaluation] Saved Evaluation Metrics JSON to: {metrics_json_path}")
    
    print("=" * 60)
    print("Full Evaluation Completed Successfully!")
    print("=" * 60)
    return metrics_summary


if __name__ == "__main__":
    run_full_evaluation()
