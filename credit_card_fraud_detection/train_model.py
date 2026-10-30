"""
Artificial Neural Network (ANN) Training Module for Credit Card Fraud Detection.

Architecture:
- Input layer: 30 features (Time, V1 to V28, Amount)
- Hidden Layer 1: 64 neurons, ReLU activation
- Dropout: 0.3
- Hidden Layer 2: 32 neurons, ReLU activation
- Dropout: 0.2
- Hidden Layer 3: 16 neurons, ReLU activation
- Output Layer: 1 neuron, Sigmoid activation

Compiler:
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Metrics: Accuracy, Precision, Recall, PR-AUC, ROC-AUC

Training Parameters:
- Epochs: 20
- Batch size: 2048
- Class weights: Inversely proportional to class frequencies to handle severe imbalance
- Early Stopping: Monitor validation PR-AUC, patience=5, restore_best_weights=True
- Validation Split: 20% stratified split of training data
"""

import os
import json
import random
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, metrics, callbacks

import preprocessing

# Reproducible random seed
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "outputs", "training_history.json")


def build_fraud_ann(input_dim: int = 30, learning_rate: float = 0.001) -> keras.Model:
    """
    Constructs the specified 3-hidden-layer feedforward ANN.
    
    Args:
        input_dim: Number of input features (default: 30).
        learning_rate: Initial Adam learning rate.
        
    Returns:
        keras.Model: Compiled Keras Sequential model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input_layer"),
        layers.Dense(64, activation="relu", name="dense_1_64"),
        layers.Dropout(0.3, name="dropout_1_0.3"),
        layers.Dense(32, activation="relu", name="dense_2_32"),
        layers.Dropout(0.2, name="dropout_2_0.2"),
        layers.Dense(16, activation="relu", name="dense_3_16"),
        layers.Dense(1, activation="sigmoid", name="output_fraud_prob")
    ], name="Credit_Card_Fraud_ANN")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(curve="PR", name="pr_auc"),
            metrics.AUC(curve="ROC", name="roc_auc")
        ]
    )
    return model


def train_model(
    data_path: str = DATA_PATH,
    model_save_path: str = MODEL_PATH,
    scaler_save_path: str = SCALER_PATH,
    history_save_path: str = HISTORY_PATH,
    epochs: int = 20,
    batch_size: int = 2048,
    validation_ratio: float = 0.20
) -> Tuple[keras.Model, Dict[str, list]]:
    """
    Executes end-to-end model training:
    1. Loads dataset and validates features.
    2. Performs stratified train/test split (80% train, 20% test).
    3. Fits StandardScaler ONLY on training features (Time, Amount).
    4. Computes class weights to handle extreme imbalance.
    5. Performs internal validation split for Early Stopping.
    6. Trains ANN and persists the model and history.
    """
    print("=" * 60)
    print("Starting Credit Card Fraud ANN Model Training Pipeline")
    print("=" * 60)
    
    # 1. Load data
    df = preprocessing.load_dataset(data_path)
    print(f"[1/6] Dataset loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # 2. Separate features and target
    X, y = preprocessing.separate_features_target(df)
    
    # 3. Stratified split into Train (80%) and Test (20%)
    X_train_full, X_test, y_train_full, y_test = preprocessing.split_data(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    print(f"[2/6] Stratified train/test split: Train={len(X_train_full):,}, Test={len(X_test):,}")
    
    # 4. Fit scaler strictly on X_train_full
    X_train_scaled, X_test_scaled, scaler = preprocessing.standardize_features(
        X_train_full, X_test, scaler_path=scaler_save_path
    )
    print(f"[3/6] Fitted StandardScaler on training data & saved to: {scaler_save_path}")
    
    # Sub-split training set into train and validation for early stopping
    X_train, X_val, y_train, y_val = preprocessing.split_data(
        X_train_scaled, y_train_full, test_size=validation_ratio, random_state=RANDOM_SEED
    )
    print(f"      Sub-split: Train={len(X_train):,}, Validation={len(X_val):,}")
    
    # 5. Compute class weights to counteract imbalance
    class_weights = preprocessing.compute_imbalance_weights(y_train)
    print(f"[4/6] Class Weights computed: Class 0 = {class_weights[0]:.4f}, Class 1 = {class_weights[1]:.4f}")
    
    # 6. Build ANN
    model = build_fraud_ann(input_dim=X_train.shape[1], learning_rate=0.001)
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Convert data to numpy float32 arrays for optimal performance
    X_tr_arr = X_train.values.astype(np.float32)
    y_tr_arr = y_train.values.astype(np.float32)
    X_val_arr = X_val.values.astype(np.float32)
    y_val_arr = y_val.values.astype(np.float32)
    
    print(f"[5/6] Training ANN for up to {epochs} epochs (Batch size: {batch_size})...")
    history = model.fit(
        X_tr_arr,
        y_tr_arr,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        validation_data=(X_val_arr, y_val_arr),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 7. Persist trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[6/6] Trained model successfully saved to: {model_save_path}")
    
    # Persist training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    with open(history_save_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"      Training history persisted to: {history_save_path}")
    
    # Save test dataset arrays for fast, deterministic evaluation
    test_data_path = os.path.join(BASE_DIR, "data", "test_data_cache.npz")
    np.savez_compressed(
        test_data_path,
        X_test=X_test_scaled.values.astype(np.float32),
        y_test=y_test.values.astype(np.int32)
    )
    print(f"      Cached test partition to: {test_data_path}")
    
    print("=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)
    return model, history_dict


if __name__ == "__main__":
    train_model()
