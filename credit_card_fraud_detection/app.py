"""
Credit Card Fraud Detection Dashboard - Streamlit Web Application
A modern financial analytics and real-time fraud prediction system powered by Artificial Neural Networks (ANN).
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import keras
import preprocessing
import evaluate_model

# -----------------------------------------------------------------------------
# Configuration and Global Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection | ANN Deep Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard_sample.csv")
SAMPLE_BATCH_PATH = os.path.join(BASE_DIR, "data", "sample_batch_transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")
THRESHOLD_CSV_PATH = os.path.join(OUTPUTS_DIR, "threshold_analysis.csv")
HISTORY_PATH = os.path.join(OUTPUTS_DIR, "training_history.json")

# Sample Transaction Presets
SAMPLE_GENUINE = {
    'Time': 10.0, 'V1': 1.449, 'V2': -1.176, 'V3': 0.913, 'V4': -1.375,
    'V5': -1.971, 'V6': -0.629, 'V7': -1.423, 'V8': 0.048, 'V9': -1.720,
    'V10': 1.626, 'V11': 1.199, 'V12': -0.671, 'V13': -0.513, 'V14': -0.095,
    'V15': 0.230, 'V16': 0.031, 'V17': 0.253, 'V18': 0.854, 'V19': -0.221,
    'V20': -0.387, 'V21': -0.009, 'V22': 0.313, 'V23': 0.027, 'V24': 0.500,
    'V25': 0.251, 'V26': -0.129, 'V27': 0.042, 'V28': 0.016, 'Amount': 7.80
}

SAMPLE_FRAUD = {
    'Time': 7672.0, 'V1': 0.7027, 'V2': 2.426, 'V3': -5.234, 'V4': 4.416,
    'V5': -2.170, 'V6': -2.667, 'V7': -3.878, 'V8': 0.911, 'V9': -0.166,
    'V10': -5.009, 'V11': 4.675, 'V12': -8.167, 'V13': 0.638, 'V14': -6.763,
    'V15': 1.296, 'V16': -3.811, 'V17': -3.754, 'V18': -1.049, 'V19': 1.554,
    'V20': 0.422, 'V21': 0.551, 'V22': -0.009, 'V23': 0.721, 'V24': 0.473,
    'V25': -1.959, 'V26': 0.319, 'V27': 0.600, 'V28': 0.129, 'Amount': 1.00
}

# -----------------------------------------------------------------------------
# Custom Banking Theme CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Layout Aesthetics */
    .reportview-container, .main {
        background-color: #F8FAFC;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header Card */
    .bank-header {
        background: linear-gradient(135deg, #0B192C 0%, #1E3E62 100%);
        color: #FFFFFF;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(11, 25, 44, 0.15);
    }
    .bank-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
    }
    .bank-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #94A3B8;
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.06);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .kpi-subtitle {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 4px;
    }
    
    /* Status Badges */
    .badge-genuine {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #10B981;
        padding: 16px 20px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .badge-fraud {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #EF4444;
        padding: 16px 20px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #0B192C;
    }
    
    /* Disclaimer Card */
    .disclaimer-card {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 13px;
        color: #1E3A8A;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Loads trained Keras ANN model."""
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return None

@st.cache_resource
def get_scaler():
    """Loads fitted StandardScaler."""
    if os.path.exists(SCALER_PATH):
        return preprocessing.load_fitted_scaler(SCALER_PATH)
    return None

@st.cache_data
def get_dataset(filepath: str = DATA_PATH):
    """Loads dataset with caching. Falls back to sample dataset if full dataset is absent."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    if os.path.exists(SAMPLE_DATA_PATH):
        return pd.read_csv(SAMPLE_DATA_PATH)
    return None

@st.cache_data
def get_threshold_curve_data():
    """Loads threshold analysis table."""
    if os.path.exists(THRESHOLD_CSV_PATH):
        return pd.read_csv(THRESHOLD_CSV_PATH)
    return None

@st.cache_data
def get_metrics_summary():
    """Loads precomputed evaluation metrics."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h2 style="color: #0B192C; margin: 0; font-size: 20px; font-weight: 800;">FINSEC ANALYTICS</h2>
    <span style="font-size: 11px; letter-spacing: 1px; color: #64748B; font-weight: 600;">FRAUD INTELLIGENCE ANN</span>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 A. Home",
        "📊 B. Data Exploration",
        "📈 C. Model Performance",
        "⚡ D. Fraud Prediction",
        "📘 E. About the Project"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Status")
model_loaded = get_model() is not None
scaler_loaded = get_scaler() is not None
data_available = os.path.exists(DATA_PATH)

st.sidebar.markdown(f"**ANN Model:** {'🟢 Active (.keras)' if model_loaded else '🔴 Missing'}")
st.sidebar.markdown(f"**StandardScaler:** {'🟢 Loaded (.pkl)' if scaler_loaded else '🔴 Missing'}")
st.sidebar.markdown(f"**Core Dataset:** {'🟢 Available (150MB)' if data_available else '🔴 Upload Required'}")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Adjust decision thresholds in Section C to balance precision and recall according to risk tolerance.")


# =============================================================================
# SECTION A: HOME
# =============================================================================
if "A. Home" in nav_selection:
    st.markdown("""
    <div class="bank-header">
        <h1>Credit Card Fraud Detection Using Artificial Neural Networks (ANN)</h1>
        <p>Production-grade deep learning architecture for high-precision European cardholder transaction surveillance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview Description
    st.markdown("""
    ### Executive Overview
    This platform provides an end-to-end Machine Learning intelligence engine engineered to detect fraudulent credit card transactions in real-time.
    Powered by a **3-hidden-layer Deep Artificial Neural Network (ANN)** trained on European cardholder transactions, the system resolves extreme class imbalance (**0.17% fraud rate**) using **inverse-frequency class weighting**, **stratified partitioning**, and **PR-AUC objective optimization**.
    """)
    
    # Key Dataset and Performance KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-value">284,807</div>
            <div class="kpi-subtitle">European cardholders</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Fraud Incidence</div>
            <div class="kpi-value" style="color: #EF4444;">492 (0.17%)</div>
            <div class="kpi-subtitle">Severe class imbalance</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Held-out Recall</div>
            <div class="kpi-value" style="color: #10B981;">89.8%</div>
            <div class="kpi-subtitle">88 / 98 Frauds Caught</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">PR-AUC Score</div>
            <div class="kpi-value" style="color: #2563EB;">0.6763</div>
            <div class="kpi-subtitle">Baseline: 0.0017 (397x)</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">ROC-AUC Score</div>
            <div class="kpi-value" style="color: #8B5CF6;">0.9546</div>
            <div class="kpi-subtitle">Overall separability</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Architecture Overview & Workflow
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.subheader("Deep ANN Pipeline Flow")
        st.markdown("""
        1. **Feature Input (30-D):** `Time`, 28 PCA-transformed numerical features (`V1`–`V28`), and transaction `Amount`.
        2. **Standardization:** Robust scaling fitted *strictly* on training partitions to prevent data leakage.
        3. **Imbalance Handling:** Balanced class weighting (~289x gradient weight for fraud class).
        4. **Deep Network Topology:**
           - **Hidden Layer 1:** 64 units, ReLU, Dropout (0.3)
           - **Hidden Layer 2:** 32 units, ReLU, Dropout (0.2)
           - **Hidden Layer 3:** 16 units, ReLU
           - **Output:** 1 unit, Sigmoid activation in [0, 1]
        5. **Operational Decisioning:** Dynamic threshold calibration for balancing False Positives vs Detection Recall.
        """)
        
    with col_right:
        st.subheader("Core Architectural Specifications")
        spec_df = pd.DataFrame({
            "Component": [
                "Input Dimensions", "Layer 1", "Regularization 1", 
                "Layer 2", "Regularization 2", "Layer 3", 
                "Output Layer", "Loss Function", "Optimizer"
            ],
            "Specification": [
                "30 Features", "64 Neurons (ReLU)", "Dropout (p=0.3)",
                "32 Neurons (ReLU)", "Dropout (p=0.2)", "16 Neurons (ReLU)",
                "1 Neuron (Sigmoid)", "Binary Crossentropy", "Adam (lr=0.001)"
            ]
        })
        st.dataframe(spec_df, use_container_width=True, hide_index=True)
        
    # Quick Navigation Call-to-action
    st.markdown("---")
    st.markdown("### Next Steps")
    c1, c2, c3 = st.columns(3)
    if c1.button("📊 Explore Dataset & Distributions", use_container_width=True):
        st.query_params["nav"] = "B. Data Exploration"
        st.rerun()
    if c2.button("📈 Inspect Model Evaluation & Curves", use_container_width=True):
        st.query_params["nav"] = "C. Model Performance"
        st.rerun()
    if c3.button("⚡ Run Fraud Prediction on Transactions", use_container_width=True):
        st.query_params["nav"] = "D. Fraud Prediction"
        st.rerun()


# =============================================================================
# SECTION B: DATA EXPLORATION
# =============================================================================
elif "B. Data Exploration" in nav_selection:
    st.markdown("""
    <div class="bank-header">
        <h1>Exploratory Data Analysis & Surveillance</h1>
        <p>Interactive statistical inspection of European credit card transactions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Loading Interface
    st.markdown("### 1. Data Ingestion")
    data_source_mode = st.radio(
        "Select Data Source:",
        ["Use Default Packaged Dataset (data/creditcard.csv)", "Upload Custom CSV Dataset"],
        horizontal=True
    )
    
    df = None
    if data_source_mode == "Upload Custom CSV Dataset":
        uploaded_file = st.file_uploader("Upload Credit Card CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully uploaded: {len(df):,} records.")
            except Exception as e:
                st.error(f"Error loading uploaded CSV: {e}")
        else:
            st.info("Please upload a CSV file with columns: Time, V1..V28, Amount, and optionally Class.")
    else:
        df = get_dataset()
        if df is None:
            st.warning("Default dataset not found in `data/creditcard.csv`. Please upload your dataset.")
            
    if df is not None:
        # Data Quality & Missing Value Inspection
        info = preprocessing.inspect_dataset(df)
        
        st.markdown("### 2. Dataset Quality & Integrity Inspection")
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        qc1.metric("Total Records", f"{info['total_records']:,}")
        qc2.metric("Total Features", info['total_features'])
        qc3.metric("Missing Values", f"{info['total_missing']:,}")
        qc4.metric("Duplicate Rows", f"{info['duplicate_count']:,} ({info['duplicate_percent']}%)")
        qc5.metric("Fraud Proportion", f"{info['fraud_percentage']}%", delta="-99.66% vs balanced")
        
        # Expandable preview
        with st.expander("🔍 Dataset Preview & Schema Inspection", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown(f"**Shape:** `{df.shape[0]:,}` rows $\\times$ `{df.shape[1]}` columns.")
            if info["duplicate_count"] > 0:
                st.caption(f"Note: {info['duplicate_count']} duplicated rows identified. In financial fraud analytics, duplicate payloads with identical attributes often represent replay transactions or identical synthetic queries.")
                
        # Interactive Visualizations
        st.markdown("### 3. Interactive Visualizations")
        
        tab_dist, tab_amount, tab_time, tab_corr = st.tabs([
            "Class Distribution", 
            "Transaction Amount Analysis", 
            "Transaction Time Patterns", 
            "Feature Correlation Heatmap"
        ])
        
        # TAB 1: Class Distribution
        with tab_dist:
            c_left, c_right = st.columns(2)
            with c_left:
                fig_donut = go.Figure(data=[go.Pie(
                    labels=["Genuine (Class 0)", "Fraudulent (Class 1)"],
                    values=[info["count_genuine"], info["count_fraud"]],
                    hole=0.6,
                    marker=dict(colors=["#10B981", "#EF4444"]),
                    textinfo="label+percent",
                    insidetextorientation="radial"
                )])
                fig_donut.update_layout(
                    title="Transaction Class Distribution (Extreme Imbalance)",
                    title_font=dict(size=15, color="#0B192C"),
                    showlegend=True,
                    height=380,
                    margin=dict(t=50, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                
            with c_right:
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=["Genuine (0)", "Fraud (1)"],
                        y=[info["count_genuine"], info["count_fraud"]],
                        text=[f"{info['count_genuine']:,}<br>(99.83%)", f"{info['count_fraud']:,}<br>(0.17%)"],
                        textposition="outside",
                        marker_color=["#10B981", "#EF4444"]
                    )
                ])
                fig_bar.update_layout(
                    title="Absolute Transaction Counts (Logarithmic Scale)",
                    yaxis_type="log",
                    yaxis_title="Count (Log Scale)",
                    title_font=dict(size=15, color="#0B192C"),
                    height=380,
                    margin=dict(t=50, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
        # TAB 2: Transaction Amount Analysis
        with tab_amount:
            a_col1, a_col2 = st.columns(2)
            
            with a_col1:
                # Boxplot comparing amounts
                fig_box = px.box(
                    df,
                    x="Class",
                    y="Amount",
                    color="Class",
                    color_discrete_map={0: "#10B981", 1: "#EF4444"},
                    labels={"Class": "Transaction Type (0: Genuine, 1: Fraud)", "Amount": "Amount (€)"},
                    title="Transaction Amount Comparison (Boxplot with Log Scale)",
                    log_y=True
                )
                fig_box.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
                
            with a_col2:
                # Statistical summary comparison table
                st.markdown("#### Transaction Amount Statistical Profile")
                gen_s = info["amount_stats_genuine"]
                fr_s = info["amount_stats_fraud"]
                
                stats_df = pd.DataFrame({
                    "Statistic": ["Count", "Mean (€)", "Std Dev (€)", "Min (€)", "25% (€)", "Median (€)", "75% (€)", "Max (€)"],
                    "Genuine (0)": [
                        f"{int(gen_s.get('count', 0)):,}",
                        f"€{gen_s.get('mean', 0):.2f}",
                        f"€{gen_s.get('std', 0):.2f}",
                        f"€{gen_s.get('min', 0):.2f}",
                        f"€{gen_s.get('25%', 0):.2f}",
                        f"€{gen_s.get('50%', 0):.2f}",
                        f"€{gen_s.get('75%', 0):.2f}",
                        f"€{gen_s.get('max', 0):.2f}"
                    ],
                    "Fraudulent (1)": [
                        f"{int(fr_s.get('count', 0)):,}",
                        f"€{fr_s.get('mean', 0):.2f}",
                        f"€{fr_s.get('std', 0):.2f}",
                        f"€{fr_s.get('min', 0):.2f}",
                        f"€{fr_s.get('25%', 0):.2f}",
                        f"€{fr_s.get('50%', 0):.2f}",
                        f"€{fr_s.get('75%', 0):.2f}",
                        f"€{fr_s.get('max', 0):.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                st.caption("Key Insight: Notice that while genuine transactions reach up to €25,691, fraud transactions cap at €2,125 with a median of €9.25, demonstrating small-amount probing behavior.")

        # TAB 3: Transaction Time Patterns
        with tab_time:
            # Time is seconds elapsed from first transaction over 48 hours (172,800 s)
            df_time = df.copy()
            df_time["Hour"] = (df_time["Time"] / 3600) % 24
            
            fig_time = px.histogram(
                df_time,
                x="Hour",
                color="Class",
                barmode="overlay",
                nbins=48,
                histnorm="probability density",
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={"Hour": "Time of Day (24-Hour Cycle)", "Class": "Type"},
                title="Normalized Transaction Activity Across 24-Hour Cycle"
            )
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
            st.caption("Temporal Dynamics: Fraud activity maintains steady elevated density during nocturnal hours (2 AM – 6 AM) when legitimate transactions experience diurnal dips.")

        # TAB 4: Correlation Heatmap
        with tab_corr:
            # Compute top correlations with target Class
            corrs = df.corr()[TARGET_COL].drop(TARGET_COL).sort_values()
            top_neg = corrs.head(6)
            top_pos = corrs.tail(6)
            top_corr_features = pd.concat([top_neg, top_pos])
            
            fig_corr = px.bar(
                x=top_corr_features.values,
                y=top_corr_features.index,
                orientation="h",
                color=top_corr_features.values,
                color_continuous_scale="RdBu_r",
                labels={"x": "Pearson Correlation Coefficient with Class", "y": "Feature"},
                title="Top Features Correlated with Transaction Fraud"
            )
            fig_corr.update_layout(height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Features `V17`, `V14`, `V12`, and `V10` exhibit the strongest inverse correlation with fraudulent activity, serving as primary discriminating signals for the ANN.")


# =============================================================================
# SECTION C: MODEL PERFORMANCE
# =============================================================================
elif "C. Model Performance" in nav_selection:
    st.markdown("""
    <div class="bank-header">
        <h1>Model Evaluation & Calibration Analytics</h1>
        <p>Comprehensive performance benchmarks on 56,962 held-out European credit card test transactions</p>
    </div>
    """, unsafe_allow_html=True)
    
    metrics_summary = get_metrics_summary()
    threshold_df = get_threshold_curve_data()
    
    if metrics_summary is None or threshold_df is None:
        st.warning("Model evaluation outputs not yet found. Running evaluation pipeline now...")
        metrics_summary = evaluate_model.run_full_evaluation()
        threshold_df = get_threshold_curve_data()
        
    # Interactive Threshold Tuning Slider
    st.markdown("### 1. Interactive Decision Threshold Calibration")
    st.markdown("""
    Because class weights scale fraud gradients to counteract extreme class imbalance, the model's continuous sigmoid output 
    represents calibrated risk probabilities. Use the slider below to inspect how shifting the decision threshold impacts 
    **Recall (detection rate)** vs **Precision (false alert rate)** in real time.
    """)
    
    calib_col1, calib_col2 = st.columns([2, 1])
    with calib_col1:
        current_threshold = st.slider(
            "Select Operating Decision Threshold:",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Higher threshold reduces false alarms (increases precision), lower threshold catches more fraud (increases recall)."
        )
    with calib_col2:
        st.markdown(f"""
        <div style="background: #F1F5F9; border-radius: 8px; padding: 12px 16px; margin-top: 10px;">
            <div style="font-size: 11px; color: #64748B; font-weight: 600;">RECOMMENDED OPERATING POINT</div>
            <div style="font-size: 14px; font-weight: 700; color: #1E293B;">Threshold: 0.90 – 0.95</div>
            <div style="font-size: 11px; color: #475569;">F1: ~0.70–0.75 | 84+ Frauds Caught | False Alarms &lt; 60</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Find closest row in threshold table
    idx = (threshold_df["threshold"] - current_threshold).abs().idxmin()
    cur_metrics = threshold_df.iloc[idx]
    
    # Dynamic Metric Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Operating Accuracy", f"{cur_metrics['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{cur_metrics['precision']*100:.2f}%")
    m3.metric("Recall (Sensitivity)", f"{cur_metrics['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{cur_metrics['f1_score']:.4f}")
    m5.metric("Frauds Caught", f"{int(cur_metrics['tp'])} / 98")
    m6.metric("False Alarms (FP)", f"{int(cur_metrics['fp']):,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Plots
    col_cm, col_tradeoff = st.columns(2)
    
    with col_cm:
        st.subheader("Dynamic Confusion Matrix")
        cm_matrix = np.array([
            [int(cur_metrics["tn"]), int(cur_metrics["fp"])],
            [int(cur_metrics["fn"]), int(cur_metrics["tp"])]
        ])
        
        cm_labels = ["Genuine (0)", "Fraud (1)"]
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=cm_labels,
            y=cm_labels,
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(
            title=f"Test Set Confusion Matrix (Threshold = {current_threshold:.2f})",
            height=380
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_tradeoff:
        st.subheader("Threshold vs Metric Trade-off")
        fig_to = go.Figure()
        fig_to.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["precision"], mode="lines", name="Precision", line=dict(color="#2563EB", width=2)))
        fig_to.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["recall"], mode="lines", name="Recall", line=dict(color="#10B981", width=2)))
        fig_to.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["f1_score"], mode="lines", name="F1-Score", line=dict(color="#F59E0B", width=2.5)))
        fig_to.add_vline(x=current_threshold, line_dash="dash", line_color="#EF4444", annotation_text=f"Current ({current_threshold:.2f})")
        fig_to.update_layout(
            title="Precision, Recall & F1 Across All Thresholds",
            xaxis_title="Decision Threshold",
            yaxis_title="Score",
            height=380,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_to, use_container_width=True)
        
    # Deep Dive: The Accuracy Paradox
    st.markdown("---")
    st.markdown("### 2. The Accuracy Paradox in Highly Imbalanced Fraud Analytics")
    st.markdown(evaluate_model.get_misleading_accuracy_explanation())
    
    # Static High-Resolution Diagnostic Plots Tabs
    st.markdown("---")
    st.markdown("### 3. Deep Learning Diagnostic Suite (Persisted Plots)")
    plot_tabs = st.tabs(["Precision-Recall Curve", "ROC Curve", "Training History", "Confusion Matrix (Saved)"])
    
    with plot_tabs[0]:
        pr_img = os.path.join(OUTPUTS_DIR, "precision_recall_curve.png")
        if os.path.exists(pr_img):
            st.image(pr_img, use_container_width=True, caption="Precision-Recall Curve on Held-Out Test Set (PR-AUC = 0.6763 vs 0.0017 Baseline)")
            
    with plot_tabs[1]:
        roc_img = os.path.join(OUTPUTS_DIR, "roc_curve.png")
        if os.path.exists(roc_img):
            st.image(roc_img, use_container_width=True, caption="Receiver Operating Characteristic Curve (ROC-AUC = 0.9546)")
            
    with plot_tabs[2]:
        hist_img = os.path.join(OUTPUTS_DIR, "training_history.png")
        if os.path.exists(hist_img):
            st.image(hist_img, use_container_width=True, caption="Training vs Validation Trajectory (Loss, Accuracy, and PR-AUC over epochs)")
            
    with plot_tabs[3]:
        cm_img = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(cm_img, use_container_width=True, caption="Pre-saved Confusion Matrix at Default Threshold (0.50)")


# =============================================================================
# SECTION D: FRAUD PREDICTION
# =============================================================================
elif "D. Fraud Prediction" in nav_selection:
    st.markdown("""
    <div class="bank-header">
        <h1>Real-Time Fraud Inference Engine</h1>
        <p>Execute instant transaction scoring with deep neural network risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = get_model()
    scaler = get_scaler()
    
    if model is None or scaler is None:
        st.error("Model or Scaler not found! Please verify that `train_model.py` has been executed.")
        st.stop()
        
    pred_tab1, pred_tab2 = st.tabs(["⚡ Single Transaction Scoring", "📁 Batch CSV Scoring"])
    
    # -------------------------------------------------------------------------
    # TAB 1: Single Transaction Scoring
    # -------------------------------------------------------------------------
    with pred_tab1:
        st.markdown("### 1. Single Transaction Attribute Input")
        st.markdown("Select a sample transaction preset for rapid testing or manually specify values:")
        
        # Preset buttons
        p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
        if "preset_data" not in st.session_state:
            st.session_state.preset_data = SAMPLE_GENUINE.copy()
            
        if p_col1.button("💳 Load Genuine Transaction Preset", use_container_width=True):
            st.session_state.preset_data = SAMPLE_GENUINE.copy()
            st.rerun()
            
        if p_col2.button("🚨 Load Fraudulent Transaction Preset", use_container_width=True):
            st.session_state.preset_data = SAMPLE_FRAUD.copy()
            st.rerun()
            
        current_data = st.session_state.preset_data
        
        # Input Form
        with st.form("single_prediction_form"):
            st.markdown("#### Primary Transaction Parameters")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                val_time = st.number_input(
                    "Transaction Time (Seconds)", 
                    min_value=0.0, 
                    max_value=200000.0, 
                    value=float(current_data["Time"]),
                    step=10.0,
                    help="Seconds elapsed since the initial transaction in the dataset"
                )
            with col_t2:
                val_amount = st.number_input(
                    "Transaction Amount (€)", 
                    min_value=0.0, 
                    max_value=100000.0, 
                    value=float(current_data["Amount"]),
                    step=1.0,
                    help="Transaction value in Euros"
                )
            with col_t3:
                decision_threshold = st.slider(
                    "Inference Decision Threshold",
                    min_value=0.10,
                    max_value=0.95,
                    value=0.85,
                    step=0.05,
                    help="Custom cutoff for classifying as fraudulent"
                )
                
            st.markdown("#### PCA Transformed Latent Features (V1 to V28)")
            with st.expander("Expand to adjust V1 – V28 feature vectors", expanded=False):
                v_cols = st.columns(4)
                v_inputs = {}
                for i in range(1, 29):
                    col_target = v_cols[(i - 1) % 4]
                    v_key = f"V{i}"
                    default_v = float(current_data.get(v_key, 0.0))
                    v_inputs[v_key] = col_target.number_input(
                        f"Feature {v_key}",
                        value=default_v,
                        format="%.4f"
                    )
            
            submit_btn = st.form_submit_button("⚡ Evaluate Transaction Risk", use_container_width=True)
            
        if submit_btn:
            # Construct row
            row_dict = {"Time": val_time, "Amount": val_amount}
            for i in range(1, 29):
                v_key = f"V{i}"
                row_dict[v_key] = v_inputs.get(v_key, float(current_data.get(v_key, 0.0)))
                
            row_df = pd.DataFrame([row_dict])
            
            # Safe preprocessing
            X_infer = preprocessing.preprocess_inference_data(row_df, scaler)
            
            # Predict
            fraud_probability = float(model.predict(X_infer, verbose=0)[0][0])
            is_fraud = fraud_probability >= decision_threshold
            
            st.markdown("### Risk Evaluation Assessment")
            
            # Risk Gauge Meter
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(fraud_probability * 100, 2),
                number={'suffix': "%", 'font': {'size': 32, 'color': '#0F172A'}},
                title={'text': "Calculated Fraud Probability", 'font': {'size': 18, 'color': '#0B192C'}},
                delta={'reference': decision_threshold * 100, 'increasing': {'color': '#EF4444'}, 'decreasing': {'color': '#10B981'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                    'bar': {'color': "#EF4444" if is_fraud else "#10B981"},
                    'bgcolor': "#F1F5F9",
                    'borderwidth': 1,
                    'bordercolor': "#CBD5E1",
                    'steps': [
                        {'range': [0, 30], 'color': "#ECFDF5"},
                        {'range': [30, 70], 'color': "#FEF3C7"},
                        {'range': [70, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {
                        'line': {'color': "#0B192C", 'width': 3},
                        'thickness': 0.75,
                        'value': decision_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Status Badge Output
            if is_fraud:
                st.markdown(f"""
                <div class="badge-fraud">
                    <span style="font-size: 24px;">🚨</span>
                    <div>
                        <div style="font-size: 18px; font-weight: 800;">CRITICAL FRAUD ALERT FLAGGED</div>
                        <div>Estimated Fraud Probability: <strong>{fraud_probability*100:.2f}%</strong> (Exceeds {decision_threshold*100:.0f}% Threshold)</div>
                        <div style="font-size: 13px; margin-top: 4px; opacity: 0.9;">Action: Transaction blocked pending automated cardholder verification and AML analyst review.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="badge-genuine">
                    <span style="font-size: 24px;">✅</span>
                    <div>
                        <div style="font-size: 18px; font-weight: 800;">TRANSACTION CLEARED AS GENUINE</div>
                        <div>Estimated Fraud Probability: <strong>{fraud_probability*100:.2f}%</strong> (Below {decision_threshold*100:.0f}% Threshold)</div>
                        <div style="font-size: 13px; margin-top: 4px; opacity: 0.9;">Action: Cleared for real-time payment authorization.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: Batch CSV Scoring
    # -------------------------------------------------------------------------
    with pred_tab2:
        st.markdown("### 2. High-Throughput Batch Transaction Scoring")
        st.markdown("Upload a CSV file containing transaction batches to process in bulk with the ANN inference engine.")
        
        batch_source = st.radio(
            "Batch Input Mode:",
            ["Load Pre-Generated Sample Batch (50 Mixed Transactions)", "Upload Custom CSV File"],
            horizontal=True
        )
        
        batch_df = None
        if batch_source == "Upload Custom CSV File":
            b_file = st.file_uploader("Upload Batch CSV (Must contain Time, V1..V28, Amount)", type=["csv"], key="batch_uploader")
            if b_file is not None:
                try:
                    batch_df = pd.read_csv(b_file)
                except Exception as e:
                    st.error(f"Error parsing batch CSV: {e}")
        else:
            if os.path.exists(SAMPLE_BATCH_PATH):
                batch_df = pd.read_csv(SAMPLE_BATCH_PATH)
            else:
                st.warning("Pre-generated sample batch not found.")
                
        if batch_df is not None:
            st.markdown(f"**Loaded Batch:** `{len(batch_df):,}` rows $\\times$ `{batch_df.shape[1]}` columns.")
            with st.expander("View Input Batch Sample", expanded=False):
                st.dataframe(batch_df.head(5), use_container_width=True)
                
            b_thresh = st.slider("Batch Classification Decision Threshold:", 0.10, 0.95, 0.85, 0.05, key="b_thresh")
            
            if st.button("🚀 Process Batch Predictions", use_container_width=True):
                with st.spinner("Processing batch vectorization and ANN inference..."):
                    # Preprocess
                    X_batch_scaled = preprocessing.preprocess_inference_data(batch_df, scaler)
                    
                    # Vectorized predict
                    probs = model.predict(X_batch_scaled, batch_size=2048, verbose=0).ravel()
                    
                    # Attach outputs
                    res_df = batch_df.copy()
                    res_df["Fraud_Probability"] = probs.round(6)
                    res_df["Risk_Score_Pct"] = (probs * 100).round(2)
                    res_df["Predicted_Class"] = (probs >= b_thresh).astype(int)
                    res_df["Verdict"] = np.where(res_df["Predicted_Class"] == 1, "FRAUD ALERT", "GENUINE")
                    
                    # Summary metrics
                    total_proc = len(res_df)
                    flagged_fraud = int(np.sum(res_df["Predicted_Class"] == 1))
                    flagged_gen = total_proc - flagged_fraud
                    avg_risk = float(np.mean(probs * 100))
                    
                    st.markdown("#### Batch Surveillance Summary")
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Processed Transactions", f"{total_proc:,}")
                    bc2.metric("Flagged Fraud Alerts", f"{flagged_fraud:,}", delta=f"{flagged_fraud/total_proc*100:.1f}%")
                    bc3.metric("Cleared Genuine", f"{flagged_gen:,}")
                    bc4.metric("Mean Risk Score", f"{avg_risk:.2f}%")
                    
                    # Distribution Plot of batch scores
                    fig_batch_dist = px.histogram(
                        res_df,
                        x="Risk_Score_Pct",
                        color="Verdict",
                        nbins=30,
                        color_discrete_map={"GENUINE": "#10B981", "FRAUD ALERT": "#EF4444"},
                        labels={"Risk_Score_Pct": "Calculated Fraud Risk Score (%)"},
                        title="Batch Risk Score Probability Distribution"
                    )
                    fig_batch_dist.add_vline(x=b_thresh * 100, line_dash="dash", line_color="#0B192C", annotation_text=f"Threshold ({b_thresh*100:.0f}%)")
                    st.plotly_chart(fig_batch_dist, use_container_width=True)
                    
                    # Highlighted Table
                    st.markdown("#### Detailed Transaction Annotations")
                    st.dataframe(
                        res_df[["Verdict", "Risk_Score_Pct", "Amount", "Time"] + [f"V{i}" for i in range(1, 6)]].head(25),
                        use_container_width=True
                    )
                    
                    # Download CSV
                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Annotated Batch Predictions (CSV)",
                        data=csv_export,
                        file_name="annotated_fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# =============================================================================
# SECTION E: ABOUT THE PROJECT
# =============================================================================
elif "E. About the Project" in nav_selection:
    st.markdown("""
    <div class="bank-header">
        <h1>About the Project & Technical Documentation</h1>
        <p>Architectural design, methodology, theoretical framework, and regulatory disclosure</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Problem Statement
    Global financial institutions encounter billions in annual losses due to unauthorized credit card transactions. 
    Detecting card fraud presents extreme data-science challenges:
    - **Extreme Class Imbalance:** Fraudulent transactions typically represent less than **0.2%** of all transaction streams.
    - **Asymmetric Misclassification Costs:** Missing a fraudulent transaction (False Negative) results in direct monetary theft and regulatory fines, whereas misclassifying a genuine transaction (False Positive) creates cardholder friction.
    - **Latency Constraints:** Fraud surveillance algorithms must compute inference scores in sub-50 milliseconds during merchant payment gateway handshakes.
    
    ### 2. Project Objectives
    1. Construct a fully reproducible, modular Deep Artificial Neural Network (ANN) in TensorFlow / Keras.
    2. Prevent data leakage by fitting preprocessing scalers strictly on training partitions.
    3. Mitigate severe class imbalance using inverse-frequency loss weighting.
    4. Provide financial officers with actionable decision-threshold adjustment capabilities.
    5. Deliver a banking-grade web surveillance dashboard.
    
    ### 3. End-to-End Methodology
    """)
    
    meth_col1, meth_col2, meth_col3 = st.columns(3)
    with meth_col1:
        st.markdown("""
        **A. Ingestion & Split**
        - Ingestion of 284,807 European cardholder transactions.
        - Stratified 80/20 train-test partition preserving exact class ratios.
        - Training sub-split into train and validation sets for early stopping.
        """)
    with meth_col2:
        st.markdown("""
        **B. Preprocessing**
        - `Time` and `Amount` standardized via `StandardScaler`.
        - Scaler fitted *only* on training data and persisted to `scaler.pkl`.
        - PCA features `V1`–`V28` maintained without synthetic distortion.
        """)
    with meth_col3:
        st.markdown("""
        **C. Training & Evaluation**
        - Dynamic class weighting ($w_1 \\approx 289$).
        - Optimization with Adam and Early Stopping on validation PR-AUC.
        - Evaluation on 56,962 unseen test records.
        """)
        
    st.markdown("""
    ### 4. Neural Network Topology Breakdown
    """)
    
    arch_df = pd.DataFrame([
        {"Stage": "Input Layer", "Neurons / Units": "30", "Activation": "Linear", "Regularization": "None", "Purpose": "Accepts 30 continuous financial features"},
        {"Stage": "Hidden Layer 1", "Neurons / Units": "64", "Activation": "ReLU", "Regularization": "Dropout (0.30)", "Purpose": "Extracts non-linear latent patterns & prevents co-adaptation"},
        {"Stage": "Hidden Layer 2", "Neurons / Units": "32", "Activation": "ReLU", "Regularization": "Dropout (0.20)", "Purpose": "Compresses representation into higher-order abstractions"},
        {"Stage": "Hidden Layer 3", "Neurons / Units": "16", "Activation": "ReLU", "Regularization": "None", "Purpose": "Final dense feature integration before classification"},
        {"Stage": "Output Layer", "Neurons / Units": "1", "Activation": "Sigmoid", "Regularization": "None", "Purpose": "Outputs continuous risk probability P(Fraud | x)"}
    ])
    st.dataframe(arch_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### 5. Production Limitations
    - **Feature Anonymization:** The dataset uses PCA-transformed features (`V1`–`V28`) due to privacy regulations, preventing direct merchant category code (MCC) or geolocation analysis.
    - **Concept Drift:** Fraud syndicates continuously alter their behavioral patterns; static models require automated periodic retraining pipelines.
    - **Temporal Window:** The dataset spans a 48-hour recording window; longer-term seasonal or holiday trends require extended historical logs.
    
    ### 6. Future Scope
    - **Graph Neural Networks (GNNs):** Model cardholder-merchant transaction graphs to identify coordinated syndicate rings.
    - **Unsupervised Autoencoders:** Complement supervised ANNs with reconstruction error anomaly scoring for zero-day fraud tactics.
    - **Streaming Kafka Ingestion:** Deploy containerized inference endpoints processing 10,000+ transactions per second.
    - **Explainable AI (SHAP / LIME):** Generate real-time feature attribution explanations for banking compliance audits.
    """)
    
    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚖️ Regulatory & Experimental Disclaimer:</strong><br>
        The predictive models, risk scores, and visualizations generated within this application are developed strictly for academic, research, and technical evaluation purposes. 
        Model outputs should not be used as the sole determinant for financial transaction blocking or legal determinations without human oversight, compliance verification, and adherence to applicable Anti-Money Laundering (AML) / Counter-Financing of Terrorism (CFT) legal frameworks.
    </div>
    """, unsafe_allow_html=True)

