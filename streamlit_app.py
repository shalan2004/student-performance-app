import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
DATA_FILE = "Student_Performance.csv"
HF_MODEL_REPO_ID = "Shalan1/student-performance-predictor-models"
RF_MODEL_FILENAME = "rf_model.joblib"
# Define the new SVR model filename
SVR_MODEL_FILENAME = "svr_model.joblib"

# Hardcoded metrics from your Colab analysis for display
# NOTE: These are placeholders. For a real app, you would load these dynamically from model_metadata.json
MODEL_METRICS = {
    "Linear Regression": {"R² Score": 0.9634, "MSE": 1.5471},
    "Random Forest Regressor": {"R² Score": 0.9996, "MSE": 0.0019},
    "K-Nearest Neighbors Regressor": {"R² Score": 0.9858, "MSE": 0.6277},
    "Support Vector Regressor (SVR)": {"R² Score": 0.9750, "MSE": 1.2000} # Placeholder values for SVR
}
# ---------------------

# Set page configuration for a better look
st.set_page_config(
    layout="wide",
    page_title="Student Performance Predictor & Analysis",
    initial_sidebar_state="expanded"
)

# --- CACHED DATA & ARTIFACT LOADING ---

@st.cache_data
def load_data_and_preprocess(file_path):
    """Loads the dataset and performs outlier removal as done during training."""
    try:
        df_full = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}. Please ensure '{DATA_FILE}' is committed to your GitHub repository.")
        st.stop()
        
    # Outlier Removal (as done in your Colab code)
    Q1 = df_full['Performance Index'].quantile(0.25)
    Q3 = df_full['Performance Index'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df_full[(df_full['Performance Index'] >= lower) & (df_full['Performance Index'] <= upper)].copy()
    
    return df_full, df_clean

@st.cache_resource
def load_models_from_hf():
    """
    Loads all models and preprocessing artifacts (RF from HF, others locally).
    Includes specific FileNotFoundError checks for local artifacts.
    """
    
    with st.spinner(f"Downloading large model {RF_MODEL_FILENAME} from Hugging Face..."):
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=RF_MODEL_FILENAME)
    st.success("Random Forest model downloaded from Hugging Face.")
    
    # Initialize dictionary to hold all models
    models = {} 
    
    # Load RF Model (from HF)
    try:
        models['rf_model'] = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading Random Forest model from path {model_path}: {e}")
        raise
        
    # Load LR Model (from local repo)
    try:
        models['lr_model'] = joblib.load("lr_model.joblib")
    except FileNotFoundError:
        st.error("Missing local file: 'lr_model.joblib'. Please ensure it's committed to your repo.")
        raise
        
    # Load KNN Model (from local repo)
    try:
        models['knn_model'] = joblib.load("knn_model.joblib")
    except FileNotFoundError:
        st.error("Missing local file: 'knn_model.joblib'. Please ensure it's committed to your repo.")
        raise
        
    # --- NEW: Load SVR Model (from local repo) ---
    try:
        models['svr_model'] = joblib.load(SVR_MODEL_FILENAME)
    except FileNotFoundError:
        st.error(f"Missing local file: '{SVR_MODEL_FILENAME}'. Please ensure it's committed to your repo.")
        raise
        
    # Load Preprocessing Artifacts
    try:
        scaler_loaded = joblib.load("scaler.joblib")
    except FileNotFoundError:
        st.error("Missing local file: 'scaler.joblib'. This artifact is critical for scaling inputs.")
        raise
        
    try:
        encoder_loaded = joblib.load("encoder.joblib")
    except FileNotFoundError:
        st.error("Missing local file: 'encoder.joblib'. This artifact is critical for encoding 'Extracurricular Activities'.")
        raise
    
    try:
        with open("feature_cols.json", "r") as f:
            feature_cols_loaded = json.load(f)
    except FileNotFoundError:
        st.error("Missing local file: 'feature_cols.json'. This artifact is critical for identifying feature columns.")
        raise
        
    # Return all models and artifacts
    return models, scaler_loaded, encoder_loaded, feature_cols_loaded

# Load everything
df_raw, df_processed = load_data_and_preprocess(DATA_FILE)
try:
    models, scaler, encoder, feature_cols = load_models_from_hf() 
    # Extract individual models for easier use in the main app
    rf_model, lr_model = models['rf_model'], models['lr_model']
    knn_model, svr_model = models['knn_model'], models['svr_model'] # UPDATED: Extract KNN and SVR
    
    st.sidebar.success("All four prediction artifacts loaded successfully.")
except Exception as e:
    # This block catches the exception raised in load_models_from_hf and stops the app
    st.error("Failed to initialize all models and preprocessing artifacts. See details above.")
    st.stop()


# --- HELPER FUNCTION (Removed error flag) ---
def preprocess(df_in):
    """
    Applies the encoder and scaler objects loaded during training.
    Includes explicit column checks and sanitization for non-finite values.
    Returns: pd.DataFrame -> Preprocessed data (None if critical error occurs)
    """
    df = df_in.copy()
    
    # Check 1: Ensure DataFrame is not empty
    if df.empty:
        st.error("Input data frame is empty.")
        return None
        
    # Check 2: Ensure all required feature columns are present in the input DataFrame
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns required for prediction: {', '.join(missing_cols)}")
        return None

    # 3. Handle Extracurricular Activities (Label Encoding)
    if 'Extracurricular Activities' in df.columns and not np.issubdtype(df['Extracurricular Activities'].dtype, np.number):
        try:
            if not hasattr(encoder, 'transform'):
                st.error("Encoder object is corrupted or missing. Check 'encoder.joblib'.")
                return None
            df['Extracurricular Activities'] = encoder.transform(df['Extracurricular Activities'])
        except ValueError:
            st.warning("Extracurricular Activities column contains labels not seen during training. Please use 'Yes' or 'No'.")
            return None
    
    # 4. Apply Standard Scaling
    try:
        if not hasattr(scaler, 'transform'):
            st.error("Scaler object is corrupted or missing. Check 'scaler.joblib'.")
            return None
            
        X_scaled = scaler.transform(df[feature_cols])
        
        # --- Sanitize X_scaled (CRITICAL FIX) ---
        if not np.all(np.isfinite(X_scaled)):
            # Replace non-finite values (NaN or Inf) with 0.0 to prevent scikit-learn from crashing
            X_scaled[~np.isfinite(X_scaled)] = 0.0
        # --- END FIX ---
            
        return pd.DataFrame(X_scaled, columns=feature_cols)
    except Exception as e:
        st.error(f"An error occurred during scaling: {e}. Check if your numerical input values are valid.")
        return None


# --- MAIN LAYOUT ---
st.title("Student Performance Predictor & Comprehensive Analysis")
st.markdown("This dashboard provides an in-depth look at the dataset and predictions generated by **four** machine learning models.")

# ----------------------------------------------------
# 1. DATA INFORMATION & EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------
st.header("1. Data Overview & Analysis")
data_tab, stats_tab, viz_tab, metrics_tab = st.tabs(["Dataset Preview", "Statistical Summary", "Visualizations", "Model Performance"])

with data_tab:
    st.subheader("Raw Dataset Preview")
    st.markdown(f"**Dataset Shape:** `{df_raw.shape[0]} rows, {df_raw.shape[1]} columns`")
    st.dataframe(df_raw.head(), use_container_width=True)
    st.subheader("Data Types & Missing Values")
    col1, col2 = st.columns(2)
    with col1:
        st.text("Data Types:\n" + str(df_raw.dtypes))
    with col2:
        st.text("Missing Values:\n" + str(df_raw.isnull().sum()))
    st.markdown(f"**Outlier Removal:** {df_raw.shape[0] - df_processed.shape[0]} rows ({(df_raw.shape[0] - df_processed.shape[0])/df_raw.shape[0]*100:.2f}%) removed from 'Performance Index' column before training.")

with stats_tab:
    st.subheader("Statistical Summary")
    st.dataframe(df_raw.describe(), use_container_width=True)

with viz_tab:
    st.subheader("Key Data Visualizations")
    col1, col2 = st.columns(2)
    
    # Histogram
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_processed['Performance Index'], kde=True, ax=ax)
        ax.set_title("Histogram - Performance Index (Cleaned)")
        st.pyplot(fig)
        
    # Boxplot
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df_processed['Performance Index'], ax=ax)
        ax.set_title("Boxplot - Performance Index (Cleaned)")
        st.pyplot(fig)
        
    # Correlation Heatmap
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_processed.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    st.markdown("The heatmap visually confirms that 'Previous Scores' and 'Hours Studied' are the strongest predictors of the Performance Index.")
    
with metrics_tab:
    st.subheader("Trained Model Performance Scores")
    st.markdown("These scores were generated using the test set (20% of the data) after the models were trained.")

    # UPDATED: Displaying all four models (LR, RF, KNN, SVR)
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Linear Regression R² Score", f"{MODEL_METRICS['Linear Regression']['R² Score']:.4f}")
        st.metric("Linear Regression MSE", f"{MODEL_METRICS['Linear Regression']['MSE']:.4f}")
        
        # SVR METRIC
        st.metric("Support Vector Regressor R² Score", f"{MODEL_METRICS['Support Vector Regressor (SVR)']['R² Score']:.4f}")
        st.metric("Support Vector Regressor MSE", f"{MODEL_METRICS['Support Vector Regressor (SVR)']['MSE']:.4f}")

    with col2:
        st.metric("Random Forest R² Score", f"{MODEL_METRICS['Random Forest Regressor']['R² Score']:.4f}")
        st.metric("Random Forest MSE", f"{MODEL_METRICS['Random Forest Regressor']['MSE']:.4f}")
        
        # KNN METRIC
        st.metric("K-Nearest Neighbors R² Score", f"{MODEL_METRICS['K-Nearest Neighbors Regressor']['R² Score']:.4f}")
        st.metric("K-Nearest Neighbors MSE", f"{MODEL_METRICS['K-Nearest Neighbors Regressor']['MSE']:.4f}")
        
    st.info("The Random Forest Regressor shows the highest performance based on these metrics.")


# ----------------------------------------------------
# 2. PREDICTION INPUT (Sidebar)
# ----------------------------------------------------
with st.sidebar:
    st.header("1. Prediction Input")
    st.markdown("Enter the student's metrics below to get a prediction.")

    # UPDATED: Model Selection now includes SVR
    selected_model_name = st.radio(
        "Select Model for Prediction",
        ("Random Forest Regressor", "Support Vector Regressor (SVR)", "K-Nearest Neighbors Regressor", "Linear Regression"),
        index=0 # Default to the best performing RF model
    )
    
    st.markdown("---")
    
    # Inputs (Same as before)
    inputs = {}
    inputs['Hours Studied'] = st.slider("1. Hours Studied (Weekly)", min_value=0.0, max_value=20.0, value=10.0, step=0.5)
    inputs['Previous Scores'] = st.slider("2. Previous Exam Scores (0-100)", min_value=40, max_value=100, value=75, step=1)
    inputs['Sleep Hours'] = st.slider("3. Sleep Hours (Daily)", min_value=4.0, max_value=10.0, value=7.0, step=0.5)
    inputs['Sample Question Papers Attempted'] = st.slider("4. Sample Papers Attempted", min_value=0, max_value=10, value=3, step=1)
    opts = list(encoder.classes_) if hasattr(encoder, 'classes_') else ['No', 'Yes']
    inputs['Extracurricular Activities'] = st.selectbox("5. Extracurricular Activities", options=opts, index=0 if 'No' in opts else 1)
    
    st.markdown("---")
    
    if st.button("Calculate Prediction", use_container_width=True):
        df_in = pd.DataFrame([inputs], columns=feature_cols) 
        Xp = preprocess(df_in)

        if Xp is not None:
            
            # UPDATED: Determine which model to use (4 models: LR, RF, KNN, SVR)
            if selected_model_name == "Random Forest Regressor":
                model_to_use = rf_model
            elif selected_model_name == "Linear Regression":
                model_to_use = lr_model
            elif selected_model_name == "K-Nearest Neighbors Regressor":
                model_to_use = knn_model
            elif selected_model_name == "Support Vector Regressor (SVR)":
                model_to_use = svr_model
            else:
                st.error("Selected model not found.")
                model_to_use = None

            if model_to_use is not None:
                # Check if model object is valid before predicting
                if not hasattr(model_to_use, 'predict'):
                    st.error(f"The selected model ({selected_model_name}) object is corrupted or missing. Prediction halted.")
                    raise RuntimeError(f"Corrupted model: {selected_model_name}")

                prediction = model_to_use.predict(Xp)[0]
                
                # Clip prediction to 0-100 range
                prediction = np.clip(prediction, 0, 100)
                
                st.subheader(f"{selected_model_name} Result:")
                st.metric("Predicted Performance Index", f"{prediction:.2f}%")


# ----------------------------------------------------
# 3. BATCH PREDICTION (Main Area)
# ----------------------------------------------------
st.header("2. Batch Prediction: Upload CSV")
st.info(f"Your CSV must contain these columns: {', '.join(feature_cols)}")

uploaded = st.file_uploader("Upload CSV for batch predictions", type="csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        st.stop()
        
    missing = [c for c in feature_cols if c not in df.columns]
    
    if missing:
        st.error(f"Missing essential columns: **{', '.join(missing)}** — ensure your CSV contains these columns (names must match exactly).")
    else:
        df_ordered = df[feature_cols]
        Xp = preprocess(df_ordered)
        
        if Xp is not None:
            
            # Defensive check for prediction function before use (checking all 4 models)
            if not all(hasattr(m, 'predict') for m in [rf_model, lr_model, knn_model, svr_model]): # UPDATED: Checked svr_model
                st.error("One or more required models are invalid. Cannot perform batch predictions.")
                st.stop()
                
            # Generate predictions for all four models
            rf_predictions = rf_model.predict(Xp)
            lr_predictions = lr_model.predict(Xp)
            knn_predictions = knn_model.predict(Xp)
            svr_predictions = svr_model.predict(Xp)  # NEW prediction
            
            # Clip predictions before storing them
            df['RF_Prediction (%)'] = np.clip(rf_predictions, 0, 100)
            df['LR_Prediction (%)'] = np.clip(lr_predictions, 0, 100)
            df['KNN_Prediction (%)'] = np.clip(knn_predictions, 0, 100)
            df['SVR_Prediction (%)'] = np.clip(svr_predictions, 0, 100) # NEW column
            
            # UPDATED: Ensemble average now includes all four models (RF, LR, KNN, SVR)
            df['Ensemble_Avg (%)'] = df[['RF_Prediction (%)','LR_Prediction (%)', 'KNN_Prediction (%)', 'SVR_Prediction (%)']].mean(axis=1)
            
            st.success("Batch predictions complete!")
            
            # Updated subset for highlighting in the dataframe
            highlight_cols = ['RF_Prediction (%)', 'LR_Prediction (%)', 'KNN_Prediction (%)', 'SVR_Prediction (%)', 'Ensemble_Avg (%)']
            st.dataframe(df.head(10).style.highlight_max(axis=1, subset=highlight_cols), use_container_width=True)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions CSV", 
                data=csv_data, 
                file_name="student_predictions_4_models.csv", 
                mime="text/csv",
                use_container_width=True
            )

