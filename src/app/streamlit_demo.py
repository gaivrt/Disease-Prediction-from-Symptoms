import streamlit as st
import pandas as pd
import numpy as np
import torch
import shap
import joblib
import matplotlib.pyplot as plt
import sys
import os
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.loader import DataLoader
from src.safety.ood_detector import OODDetector

# Page Config
st.set_page_config(page_title="Intelligent Disease Predictor", page_icon="🩺", layout="wide")

@st.cache_resource
def load_resources():
    # Load Data Specs (for symptom names)
    loader = DataLoader()
    X_train, _, _, _, encoder = loader.load_data()
    symptoms = X_train.columns.tolist()
    
    # 1. Load PyTorch Model (Research Demo)
    # The OODDetector wraps the PyTorch model
    pytorch_detector = OODDetector()
    
    # 2. Load Tuned ML Models (SOTA)
    models = {
        "PyTorch MLP (Research)": pytorch_detector,
        "XGBoost (Best Speed)": joblib.load('./saved_model/xgboost_best.joblib'),
        "LightGBM (Efficiency)": joblib.load('./saved_model/lightgbm_best.joblib'),
        "CatBoost (Robustness)": joblib.load('./saved_model/catboost_best.joblib')
    }
    
    return symptoms, models, encoder.classes_, X_train

symptoms_list, models, class_names, X_train_bg = load_resources()

# --- Sidebar ---
st.sidebar.title("🩺 Patient Symptoms")
st.sidebar.markdown("Select all symptoms that apply:")

# Model Selection
model_choice = st.sidebar.selectbox(
    "Choose AI Model",
    list(models.keys()),
    index=0
)

active_model = models[model_choice]

selected_symptoms = st.sidebar.multiselect(
    "Search Symptoms",
    symptoms_list,
    default=["high_fever", "cough"] if "high_fever" in symptoms_list else []
)


# Function to convert selection to vector
def selection_to_vector(selected, all_symptoms):
    vector = np.zeros(len(all_symptoms))
    for s in selected:
        if s in all_symptoms:
            idx = all_symptoms.index(s)
            vector[idx] = 1
    return vector

# --- Main Content ---
st.title("🛡️ Research-Level Medical AI Diagnosis System")
st.markdown(f"""
**Current Backend**: `{model_choice}`
This system integrates **Deep Learning**, **Tree Boosting SOTA**, **OOD Safety Mechanisms**, and **SHAP Interpretability** 
to provide a reliable preliminary diagnosis.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Clinical Diagnosis")
    
    if st.button("Run Diagnosis", type="primary"):
        input_vector = selection_to_vector(selected_symptoms, symptoms_list)
        
        # Prepare inputs based on model type
        # PyTorch needs Tensor input (1, 132)
        # Trees need DataFrame/Numpy (1, 132)
        
        prediction_label = ""
        confidence = 0.0
        entropy = 0.0
        is_safe = True
        
        if "PyTorch" in model_choice:
            # PyTorch Logic (with OOD built-in class)
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
            result = active_model.predict_safe(input_tensor)
            
            prediction_label = result['prediction']
            confidence = result['confidence']
            entropy = result['entropy']
            is_safe = result['is_safe']
            
        else:
            # Tree Model Logic
            # Create DF for feature names match (important for XGBoost/CatBoost)
            input_df = pd.DataFrame([input_vector], columns=symptoms_list)
            
            # Predict
            probs = active_model.predict_proba(input_df)[0]
            max_idx = np.argmax(probs)
            confidence = probs[max_idx]
            
            # Calculate Entropy manually for Safety
            # H(x) = - sum(p * log(p))
            safe_probs = probs + 1e-10
            entropy = -np.sum(safe_probs * np.log(safe_probs))
            
            threshold = 1.5 # Consistent threshold
            is_safe = entropy < threshold
            
            prediction_label = class_names[max_idx] if is_safe else "Refer to Doctor (Uncertain)"

        # Display Result
        if is_safe:
            st.success(f"**Prediction:** {prediction_label}")
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.metric("Entropy (Uncertainty)", f"{entropy:.4f}")
        else:
            st.error(f"**Prediction:** {prediction_label}")
            st.warning("⚠️ High Uncertainty detected. This symptom combination is Out-of-Distribution (OOD). Please consult a specialist.")
            st.metric("Entropy", f"{entropy:.4f}", delta="Too High", delta_color="inverse")

        # 2. SHAP Explanation
        st.subheader("2. AI Reasoning (SHAP Analysis)")
        with st.spinner("Calculating feature contributions..."):
            
            # Determine Explainer and SHAP values based on model type
            class_shap = None
            pred_idx = 0 
            
            if "PyTorch" in model_choice:
                # Deep Explainer
                input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
                background = torch.FloatTensor(X_train_bg.sample(50, random_state=42).values)
                explainer = shap.DeepExplainer(active_model.model, background)
                shap_values = explainer.shap_values(input_tensor)
                
                # Get predicted index
                model_out = active_model.model(input_tensor)
                pred_idx = torch.argmax(model_out).item()
                
                # Fix DeepExplainer List/Array issue
                if isinstance(shap_values, list):
                    class_shap = shap_values[pred_idx][0]
                else: 
                     # Array (1, 132, 41) -> [0, :, pred_idx]
                     if len(shap_values.shape) == 3:
                         class_shap = shap_values[0, :, pred_idx]
                     else:
                        class_shap = shap_values[0]

            else:
                # Tree Explainer (XGB, LGB, CatBoost)
                # TreeExplainer is fast and native
                input_df = pd.DataFrame([input_vector], columns=symptoms_list)
                
                # Note: CatBoost usually needs native shap calculation or TreeExplainer with model
                explainer = shap.TreeExplainer(active_model)
                shap_values = explainer.shap_values(input_df)
                
                # TreeExplainer for multiclass returns list of arrays or array (n, features, classes)
                # Usually list of (n_samples, n_features) for each class
                
                probs = active_model.predict_proba(input_df)[0]
                pred_idx = np.argmax(probs)
                
                if isinstance(shap_values, list):
                    class_shap = shap_values[pred_idx][0]
                elif len(np.array(shap_values).shape) == 3:
                     # (n_samples, n_features, n_classes) for LightGBM sometimes
                     # But Check shape
                     sv = np.array(shap_values)
                     # If (1, 132, 41)
                     class_shap = sv[0, :, pred_idx]
                else:
                    # Generic fallback
                    # Check if binary (only 1 output)? No, multiclass.
                    # Should be handled above.
                    class_shap = shap_values[0] # Fallback
            
            # --- Visualization Logic (Shared) ---
            
            # Map values to features
            contributions = pd.DataFrame({
                'Symptom': symptoms_list,
                'Contribution': class_shap
            })
            
            # Filter and Sort for Top Features
            contributions['AbsContribution'] = contributions['Contribution'].abs()
            # Take top 15 most important features
            top_contributions = contributions.sort_values('AbsContribution', ascending=False).head(15)
            # Sort by actual contribution for plotting (so they appear ordered in the bar chart)
            top_contributions = top_contributions.sort_values('Contribution', ascending=True)
            
            # Plot
            plt.style.use('ggplot') # Make it look nicer
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ff4b4b' if x > 0 else '#1c83e1' for x in top_contributions['Contribution']] # Streamlit red/blue
            
            bars = ax.barh(top_contributions['Symptom'], top_contributions['Contribution'], color=colors)
            ax.set_title(f"Top 15 Factors for {class_names[pred_idx]}", fontsize=14)
            ax.set_xlabel("Impact on Prediction (SHAP Value)", fontsize=10)
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width if width > 0 else width
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                        va='center', fontsize=8, color='black')

            st.pyplot(fig)
            st.info(f"Analysis based on {model_choice} Logic.")

with col2:
    st.subheader("3. System Status")
    st.json({
        "Active Model": model_choice,
        "Safety Mechanism": "Entropy-based OOD Detector (Active)",
        "Threshold": 1.5,
        "Input Features": len(symptoms_list),
        "Total Classes": len(class_names)
    })
    
    st.markdown("### Selected Symptoms Vector")
    st.code(str(selected_symptoms))

st.markdown("---")
st.caption("Course Project | Designed by Tao Xilin | Research-Level Implementation")
