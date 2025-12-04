import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
from huggingface_hub import hf_hub_download # NEW: Import for downloading large files

# --- CONFIGURATION (!!! IMPORTANT: REPLACE THESE VALUES !!!) ---
# 1. Replace <your-username> with your actual Hugging Face username.
# 2. Replace <model-repo-name> with the name of the Hugging Face repo where you stored rf_model.joblib.
HF_MODEL_REPO_ID = "Shalan1/student-performance-predictor-models" 
RF_MODEL_FILENAME = "rf_model.joblib"
# -----------------------------------------------------------------

# Set page configuration for a better look
st.set_page_config(
    layout="wide", 
    page_title="Student Performance Predictor",
    initial_sidebar_state="expanded"
)

# --- Title and Description ---
st.title("🎓 Student Performance Predictor")
st.markdown("Model artifacts loaded from GitHub and Hugging Face Hub.")
st.markdown("Select an option in the sidebar for a single prediction, or upload a CSV for batch processing.")


# Use Streamlit's caching mechanism (st.cache_resource) for efficiency.
# This function only runs once when the app is deployed or when the code changes.
@st.cache_resource
def load_models_from_hf():
    """Downloads the large RF model from Hugging Face and loads all artifacts."""
    
    # 1. Download the large RF model file
    with st.spinner(f"Downloading large model {RF_MODEL_FILENAME} from Hugging Face..."):
        # hf_hub_download saves the file locally in a Streamlit cache directory and returns the local path
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=RF_MODEL_FILENAME)
    st.success("Download complete.")
    
    # 2. Load all models and artifacts (RF from the downloaded path, others locally)
    rf_model_loaded = joblib.load(model_path)
    
    # Load small artifacts from the GitHub repo directory
    lr_model_loaded = joblib.load("lr_model.joblib")
    scaler_loaded = joblib.load("scaler.joblib")
    encoder_loaded = joblib.load("encoder.joblib")
    
    with open("feature_cols.json", "r") as f:
        feature_cols_loaded = json.load(f)
        
    return rf_model_loaded, lr_model_loaded, scaler_loaded, encoder_loaded, feature_cols_loaded

# --- Load Models + Feature List (NEW CALL BLOCK) ---
try:
    # Call the cached function to handle artifact loading
    rf_model, lr_model, scaler, encoder, feature_cols = load_models_from_hf() 
    st.sidebar.success("All prediction artifacts loaded successfully.")
    
except Exception as e:
    st.error(f"Error during model loading: {e}. Please ensure you updated the `HF_MODEL_REPO_ID` and that all small files are committed to your GitHub repo.")
    st.stop()


# --- Helper to preprocess a DataFrame of raw feature columns ---
def preprocess(df_in):
    """
    Applies the encoder and scaler objects loaded during training.
    Assumes df_in has the columns defined in feature_cols.
    """
    df = df_in.copy()
    
    # 1. Handle Extracurricular Activities (Label Encoding)
    # Check if the column is present and if it's not already numeric (i.e., it's 'Yes'/'No')
    if 'Extracurricular Activities' in df.columns and not np.issubdtype(df['Extracurricular Activities'].dtype, np.number):
        try:
            df['Extracurricular Activities'] = encoder.transform(df['Extracurricular Activities'])
        except ValueError:
            st.warning("Extracurricular Activities column contains labels not seen during training. Please use 'Yes' or 'No'.")
            return None
    
    # 2. Apply Standard Scaling
    # Ensure the columns are in the correct order for scaling
    X_scaled = scaler.transform(df[feature_cols])
    return pd.DataFrame(X_scaled, columns=feature_cols)


# --- Sidebar for single prediction (Input Form) ---
with st.sidebar:
    st.header("1. Manual Input Prediction")
    st.markdown("Enter the student's metrics below.")

    # Specific inputs based on the known features
    inputs = {}
    
    # Feature 1: Hours Studied
    inputs['Hours Studied'] = st.slider(
        "Hours Studied (Weekly)", 
        min_value=0.0, 
        max_value=100.0, 
        value=15.0, 
        step=0.5
    )
    
    # Feature 2: Sample Question Papers Attempted
    inputs['Sample Question Papers Attempted'] = st.slider(
        "Sample Papers Attempted", 
        min_value=0, 
        max_value=15, 
        value=3, 
        step=1
    )
    
    # Feature 3: Extracurricular Activities (Categorical)
    opts = list(encoder.classes_) if hasattr(encoder, 'classes_') else ['No', 'Yes']
    inputs['Extracurricular Activities'] = st.selectbox(
        "Extracurricular Activities", 
        options=opts,
        index=0 if 'No' in opts else 1
    )
    
    st.markdown("---")
    
    if st.button("Calculate Prediction", use_container_width=True):
        df_in = pd.DataFrame([inputs])
        Xp = preprocess(df_in)

        if Xp is not None:
            # Generate Predictions
            pred_rf = rf_model.predict(Xp)[0]
            pred_lr = lr_model.predict(Xp)[0]
            
            st.subheader("Prediction Results:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Random Forest (RF)", f"{pred_rf:.2f}", help="Generally higher accuracy.")
            with col2:
                st.metric("Linear Regression (LR)", f"{pred_lr:.2f}", help="Simpler, more interpretable model.")
            with col3:
                ensemble_avg = (pred_rf + pred_lr) / 2
                st.metric("Ensemble Average", f"{ensemble_avg:.2f}", delta="Recommendation")


# --- CSV upload for batch predictions ---
st.subheader("2. Batch Prediction: Upload CSV")
st.info(f"Your CSV must contain these columns: {', '.join(feature_cols)}")

uploaded = st.file_uploader("Upload CSV for batch predictions", type="csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        st.stop()
        
    # Check for missing columns
    missing = [c for c in feature_cols if c not in df.columns]
    
    if missing:
        st.error(f"Missing essential columns: **{', '.join(missing)}** — ensure your CSV contains these columns (names must match exactly).")
    else:
        # Preprocess and predict
        Xp = preprocess(df)
        
        if Xp is not None:
            # Generate Predictions
            df['RF_Prediction'] = rf_model.predict(Xp)
            df['LR_Prediction'] = lr_model.predict(Xp)
            df['Ensemble_Avg'] = df[['RF_Prediction','LR_Prediction']].mean(axis=1)
            
            st.success("Batch predictions complete!")
            
            # Display results
            st.dataframe(df.head(10).style.highlight_max(axis=1, subset=['RF_Prediction', 'LR_Prediction', 'Ensemble_Avg']), use_container_width=True)
            
            # Download button
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions CSV", 
                data=csv_data, 
                file_name="student_predictions.csv", 
                mime="text/csv",
                use_container_width=True
            )

