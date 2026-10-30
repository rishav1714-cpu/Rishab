"""
Data Preprocessing Module for Credit Card Fraud Detection.

This module handles:
- Dataset loading and validation.
- Missing-value and duplicate inspection.
- Class distribution analysis.
- Feature and target separation (Time, V1-V28, Amount -> Class).
- Stratified train-test splitting (strictly avoiding data leakage).
- Standardisation of Time and Amount features using StandardScaler.
- Class weight computation to handle severe imbalance.
- Saving and loading of the fitted scaler object.
- Safe preprocessing for single-row and batch inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Standard feature order: 30 numerical features
FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
TARGET_COL = 'Class'


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads credit card dataset from CSV file.
    
    Args:
        filepath: Absolute or relative path to CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate expected columns
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects dataset for missing values, duplicates, and class balance.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dict containing summary statistics and distribution figures.
    """
    total_records = len(df)
    missing_values = df.isnull().sum().to_dict()
    total_missing = sum(missing_values.values())
    duplicate_count = int(df.duplicated().sum())
    duplicate_percent = (duplicate_count / total_records * 100) if total_records > 0 else 0.0
    
    class_counts = df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {}
    count_genuine = int(class_counts.get(0, 0))
    count_fraud = int(class_counts.get(1, 0))
    fraud_percentage = (count_fraud / total_records * 100) if total_records > 0 else 0.0
    
    return {
        "total_records": total_records,
        "total_features": len(FEATURE_COLS),
        "total_missing": total_missing,
        "missing_by_column": missing_values,
        "duplicate_count": duplicate_count,
        "duplicate_percent": round(duplicate_percent, 4),
        "count_genuine": count_genuine,
        "count_fraud": count_fraud,
        "fraud_percentage": round(fraud_percentage, 4),
        "genuine_percentage": round(100.0 - fraud_percentage, 4),
        "amount_stats_genuine": df[df[TARGET_COL] == 0]['Amount'].describe().to_dict() if TARGET_COL in df.columns else {},
        "amount_stats_fraud": df[df[TARGET_COL] == 1]['Amount'].describe().to_dict() if TARGET_COL in df.columns else {}
    }


def separate_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the 30 input features from the target Class column.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy().astype(int)
    return X, y


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train-test split to preserve fraud proportion in both splits.
    
    Args:
        X: Feature DataFrame.
        y: Target series.
        test_size: Fraction of samples reserved for testing. Default 0.20 (80/20 split).
        random_state: Seed for reproducibility.
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def standardize_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    scaler_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Standardises Time and Amount features using StandardScaler.
    Crucial: The scaler is fitted strictly on X_train to avoid data leakage.
    
    Args:
        X_train: Training feature DataFrame.
        X_test: Optional test feature DataFrame to transform.
        scaler_path: Optional file path to persist the fitted scaler.
        
    Returns:
        Tuple: (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    
    cols_to_scale = ['Time', 'Amount']
    
    # Fit strictly on train
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[Preprocessing] Fitted StandardScaler saved to: {scaler_path}")
        
    return X_train_scaled, X_test_scaled, scaler


def compute_imbalance_weights(y_train: pd.Series) -> Dict[int, float]:
    """
    Computes balanced class weights for training the ANN:
    weight_j = total_samples / (n_classes * samples_j)
    
    Args:
        y_train: Training target labels.
        
    Returns:
        Dict[int, float]: Class weight mapping {0: weight_0, 1: weight_1}.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    return class_weight_dict


def load_fitted_scaler(scaler_path: str) -> StandardScaler:
    """
    Loads saved StandardScaler for inference.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    return joblib.load(scaler_path)


def preprocess_inference_data(
    df: pd.DataFrame, 
    scaler: StandardScaler
) -> np.ndarray:
    """
    Prepares input dataframe for model prediction:
    - Validates presence of all 30 features.
    - Preserves correct column order: ['Time', 'V1'..'V28', 'Amount'].
    - Applies the saved fitted scaler to Time and Amount.
    
    Args:
        df: DataFrame containing the 30 features.
        scaler: Fitted StandardScaler.
        
    Returns:
        np.ndarray: Scaled feature array of shape (N, 30) ready for model.predict().
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required features: {missing}")
        
    df_clean = df[FEATURE_COLS].copy()
    cols_to_scale = ['Time', 'Amount']
    df_clean[cols_to_scale] = scaler.transform(df_clean[cols_to_scale])
    
    return df_clean.values.astype(np.float32)
