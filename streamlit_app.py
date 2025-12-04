%%writefile streamlit_app.py
import streamlit as st
import pandas as pd, numpy as np, joblib, json
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Student Performance Predictor")
st.title("🎓 Student Performance Predictor — lazy deploy edition")
st.markdown("Upload a student CSV or enter values on the left. App expects the same features used during training.")

# --- load models + feature list ---
rf = joblib.load("rf_model.joblib")
lr = joblib.load("lr_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")
with open("feature_cols.json", "r") as f:
    feature_cols = json.load(f)

# --- helper to preprocess a DataFrame of raw feature columns ---
def preprocess(df_in):
    df = df_in.copy()
    # handle Extracurricular Activities if present and non-numeric
    if 'Extracurricular Activities' in df.columns and not np.issubdtype(df['Extracurricular Activities'].dtype, np.number):
        df['Extracurricular Activities'] = encoder.transform(df['Extracurricular Activities'])
    X_scaled = scaler.transform(df[feature_cols])
    return pd.DataFrame(X_scaled, columns=feature_cols)

# --- Sidebar for single prediction ---
with st.sidebar:
    st.header("Single prediction")
    # If you want super-lazy, automatically populate sliders by inspecting feature types:
    # For simplicity, we assume numeric features and render generic inputs. Adjust labels below if needed.
    inputs = {}
    for col in feature_cols:
        # treat categorical extracurriculars specially
        if col == 'Extracurricular Activities':
            opts = list(encoder.classes_) if hasattr(encoder, 'classes_') else ['No', 'Yes']
            inputs[col] = st.selectbox(col, opts)
        else:
            # numeric slider with broad bounds (user can type exact value too)
            inputs[col] = st.number_input(col, value=0.0)
    if st.button("Predict single"):
        df_in = pd.DataFrame([inputs])
        Xp = preprocess(df_in)
        pred_rf = rf.predict(Xp)[0]
        pred_lr = lr.predict(Xp)[0]
        st.metric("Random Forest", f"{pred_rf:.2f}")
        st.metric("Linear Regression", f"{pred_lr:.2f}")
        st.write("Average:", (pred_rf+pred_lr)/2)

# --- CSV upload for batch predictions ---
st.write("---")
st.subheader("Batch: upload CSV with these columns (exact names & order not necessary)")
st.write(feature_cols)
uploaded = st.file_uploader("Upload CSV for batch predictions", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing} — ensure your CSV contains these columns (names must match).")
    else:
        Xp = preprocess(df)
        df['RF_Prediction'] = rf.predict(Xp)
        df['LR_Prediction'] = lr.predict(Xp)
        df['Ensemble_Avg'] = df[['RF_Prediction','LR_Prediction']].mean(axis=1)
        st.success("Predictions ready")
        st.dataframe(df.head())
        st.download_button("Download predictions CSV", df.to_csv(index=False).encode('utf-8'), "preds.csv", "text/csv")
