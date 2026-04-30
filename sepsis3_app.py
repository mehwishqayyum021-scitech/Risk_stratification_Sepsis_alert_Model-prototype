import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap  # New requirement

# ============================================================
# PAGE CONFIG & ASSETS (Same as your original)
# ============================================================
st.set_page_config(page_title="Sepsis Clinical Risk Stratifier AI", page_icon="🌡️", layout="centered")

@st.cache_resource
def load_sepsis_assets():
    model = joblib.load("sepsis_lr_safe_preproc_v03_model.joblib")
    scaler = joblib.load("sepsis_lr_safe_preproc_v03_scaler.joblib")
    model_columns = joblib.load("sepsis_lr_safe_preproc_v03_columns.joblib")
    train_medians = joblib.load("sepsis_lr_safe_preproc_v03_train_medians.joblib")
    clinical_normals = joblib.load("sepsis_lr_safe_preproc_v03_clinical_normals.joblib")
    threshold = joblib.load("sepsis_lr_safe_preproc_v03_threshold.joblib")
    return model, scaler, model_columns, train_medians, clinical_normals, threshold

try:
    model, scaler, model_columns, train_medians, clinical_normals, chosen_threshold = load_sepsis_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}"); st.stop()

# ============================================================
# HEADER & INPUTS (Keeping your existing UI layout)
# ============================================================
st.title("🛡️ Sepsis Early Alert System")
st.caption("Educational Clinical Decision Support System Prototype")

st.subheader("Vital Signs & Selected Clinical Laboratory Values")

col1, col2, col3 = st.columns(3)

raw_inputs = {}

with col1:
    raw_inputs["HR"] = st.number_input(
        "Heart Rate / HR",
        min_value=0.0,
        max_value=250.0,
        value=float(clinical_normals.get("HR", 80.0))
    )

    raw_inputs["O2Sat"] = st.number_input(
        "Oxygen Saturation / O2Sat",
        min_value=0.0,
        max_value=100.0,
        value=float(clinical_normals.get("O2Sat", 98.0))
    )

    raw_inputs["Temp"] = st.number_input(
        "Temperature °C",
        min_value=30.0,
        max_value=45.0,
        value=float(clinical_normals.get("Temp", 37.0))
    )

    raw_inputs["MAP"] = st.number_input(
        "Mean Arterial Pressure / MAP",
        min_value=0.0,
        max_value=200.0,
        value=float(clinical_normals.get("MAP", 80.0))
    )


with col2:
    raw_inputs["SBP"] = st.number_input(
        "Systolic BP / SBP",
        min_value=0.0,
        max_value=250.0,
        value=float(clinical_normals.get("SBP", 120.0))
    )

    raw_inputs["DBP"] = st.number_input(
        "Diastolic BP / DBP",
        min_value=0.0,
        max_value=150.0,
        value=float(clinical_normals.get("DBP", 70.0))
    )

    raw_inputs["Resp"] = st.number_input(
        "Respiratory Rate",
        min_value=0.0,
        max_value=100.0,
        value=float(clinical_normals.get("Resp", 16.0))
    )

    raw_inputs["Glucose"] = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=800.0,
        value=float(clinical_normals.get("Glucose", 100.0))
    )


with col3:
    raw_inputs["WBC"] = st.number_input(
        "WBC Count",
        min_value=0.0,
        max_value=100.0,
        value=float(clinical_normals.get("WBC", 7.0))
    )

    raw_inputs["Creatinine"] = st.number_input(
        "Creatinine",
        min_value=0.0,
        max_value=20.0,
        value=float(clinical_normals.get("Creatinine", 1.0))
    )

    raw_inputs["Platelets"] = st.number_input(
        "Platelets",
        min_value=0.0,
        max_value=1000.0,
        value=float(clinical_normals.get("Platelets", 250.0))
    )

    raw_inputs["Lactate"] = st.number_input(
        "Lactate",
        min_value=0.0,
        max_value=30.0,
        value=float(clinical_normals.get("Lactate", 1.0))
    )


# Add this to your sidebar
with st.sidebar:
    st.title("About the Project")
    st.info("""
    **Research Prototype**
    This model is part of a health informatics research project.
    """)

    st.warning("""
    **Legal Disclaimer:**
    This tool is **not FDA regulated** or licensed for clinical use.
    It is intended for **Educational & Research Purposes Only**.
    Do not use for patient diagnosis or treatment.
    """)
# ============================================================
# LOGIC: BUILD PATIENT ROW (Keeping your logic)
# ============================================================
def build_patient_feature_row(raw_inputs, age, gender_value, model_columns, train_medians, clinical_normals):
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    for col in input_df.columns:
        if col.lower() == "age": input_df.loc[0, col] = float(age)
        if col.lower() == "gender": input_df.loc[0, col] = float(gender_value)

    final_clinical_values = {}
    for col in set(list(clinical_normals.keys()) + list(raw_inputs.keys())):
        val = raw_inputs.get(col)
        is_missing = False
        if val is None:
            val = train_medians.get(col, clinical_normals.get(col, 0.0))
            is_missing = True
        final_clinical_values[col] = float(val)
        if col in input_df.columns: input_df.loc[0, col] = float(val)
        m_col = f"{col}_is_missing"
        if m_col in input_df.columns: input_df.loc[0, m_col] = 1.0 if is_missing else 0.0

    for col, normal_val in clinical_normals.items():
        curr_val = final_clinical_values.get(col, normal_val)
        d_col, ad_col = f"{col}_diff_from_normal", f"{col}_abs_diff_from_normal"
        if d_col in input_df.columns: input_df.loc[0, d_col] = float(curr_val) - float(normal_val)
        if ad_col in input_df.columns: input_df.loc[0, ad_col] = abs(float(curr_val) - float(normal_val))

    return input_df[model_columns].fillna(0.0)

# ============================================================
# PREDICTION & ICONIC SHAP WATERFALL
# ============================================================
if st.button("Analyze Sepsis Risk", use_container_width=True):
    input_df = build_patient_feature_row(raw_inputs, age, gender_value, model_columns, train_medians, clinical_normals)

    # Scale the inputs
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=model_columns)

    # Get Probability
    risk_probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Risk Probability", f"{risk_probability:.2%}")
    m2.metric("Status", "🚨 High Risk" if risk_probability >= chosen_threshold else "✅ Stable")

    # --- THE PINK & BLUE SHAP WATERFALL ---
    st.subheader("Patient-Specific Risk Drivers")

    try:
        # 1. Setup the SHAP Explainer
        # For Logistic Regression, we use the Linear explainer
        explainer = shap.Explainer(model, scaler.transform(input_df)) # Base it on the scaled data
        shap_values = explainer(input_scaled)

        # 2. Plotting the Waterfall (The classic Pink/Blue look)
        # We wrap it in a figure to display it in Streamlit
        fig, ax = plt.subplots(figsize=(10, 6))

        # Max_display=7 keeps it clean and avoids the 'repetition' of minor features
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)

        # Adjust layout to prevent clipping
        plt.tight_layout()
        st.pyplot(fig)

        st.info("💗 Pink bars increase probability | 💙 Blue bars decrease probability")

    except Exception as e:
        st.error(f"Could not generate Waterfall plot: {e}")
        st.info("Make sure 'shap' is installed: pip install shap")

    with st.expander("Developer View: Feature Values"):
        st.dataframe(input_df)
