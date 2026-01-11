"""
Disease Prediction - Gradio Control Panel
Full workflow: Train, Evaluate, Predict & Verify
"""
import gradio as gr
import pandas as pd
import numpy as np
import os
import sys
from joblib import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.loader import DataLoader
from train import train_model
from evaluate import evaluate_model
from models import MODELS, MODEL_NAMES

# Load data once
loader = DataLoader()
X_train, y_train, X_test, y_test, encoder = loader.load_data()
SYMPTOMS = X_train.columns.tolist()
DISEASES = encoder.classes_.tolist()

# Get training data as set of tuples for checking if combination exists
TRAIN_COMBINATIONS = set(tuple(row) for row in X_train.values)


# ============== Tab 1: Train ==============
def do_train(model_name):
    """Train selected model."""
    try:
        result = train_model(model_name)
        output = f"Model: {result['model_name']}\n"
        output += f"Training samples: {result['train_samples']}\n\n"
        output += f"Model saved to: {result['save_path']}\n\n"
        output += "Training complete! Go to Evaluate tab to see test results."
        return output
    except Exception as e:
        return f"Error: {str(e)}"


# ============== Tab 2: Evaluate ==============
def do_evaluate(model_name):
    """Evaluate selected model on test set."""
    try:
        result = evaluate_model(model_name)
        output = f"Model: {result['model_name']}\n"
        output += f"Test samples: {result['test_samples']}\n\n"
        output += f"Accuracy: {result['accuracy']:.4f}\n"
        output += f"Macro F1: {result['f1_macro']:.4f}\n\n"
        output += f"Classification Report:\n{result['classification_report']}"
        return output
    except FileNotFoundError:
        return f"Error: Model not found. Please train {model_name} first."
    except Exception as e:
        return f"Error: {str(e)}"


# ============== Tab 3: Predict & Verify ==============
def do_predict(symptom_list, model_name):
    """Predict disease from symptoms."""
    if not symptom_list:
        return "Please select at least one symptom."
    
    # Create feature vector
    feature_vector = np.zeros(len(SYMPTOMS))
    for s in symptom_list:
        if s in SYMPTOMS:
            idx = SYMPTOMS.index(s)
            feature_vector[idx] = 1
    
    # Check if combination exists in training data
    combination_exists = tuple(feature_vector) in TRAIN_COMBINATIONS
    
    # Load model and predict
    model_class = MODELS[model_name]
    model_path = os.path.join('../saved_model', model_class.filename)
    
    if not os.path.exists(model_path):
        return f"Error: Model not found. Please train {model_name} first."
    
    clf = load(model_path)
    df = pd.DataFrame([feature_vector], columns=SYMPTOMS)
    prediction_encoded = clf.predict(df)[0]
    
    # Get confidence using predict_proba
    proba = clf.predict_proba(df)[0]
    confidence = max(proba) * 100
    
    # Decode prediction to disease name
    if isinstance(prediction_encoded, (int, np.integer)):
        prediction = encoder.inverse_transform([prediction_encoded])[0]
    else:
        prediction = prediction_encoded
    
    # Build output
    output = f"Model: {model_class.name}\n"
    output += f"Selected symptoms ({len(symptom_list)}): {', '.join(symptom_list)}\n\n"
    output += f"Predicted Disease: {prediction}\n"
    output += f"Confidence: {confidence:.2f}%\n"
    
    if not combination_exists:
        output += "\nNote: This symptom combination is not in the training data. Prediction may be unreliable."
    
    return output


# ============== Build Interface ==============
with gr.Blocks(title="Disease Prediction Control Panel") as demo:
    gr.Markdown("# Disease Prediction from Symptoms")
    gr.Markdown("A machine learning system for predicting diseases based on symptoms.")
    
    with gr.Tabs():
        # Tab 1: Train
        with gr.TabItem("Train"):
            gr.Markdown("### Train a Model")
            train_model_select = gr.Radio(
                choices=MODEL_NAMES, 
                value="random_forest",
                label="Select Model"
            )
            train_btn = gr.Button("Start Training", variant="primary")
            train_output = gr.Textbox(label="Training Results", lines=12)
            train_btn.click(do_train, inputs=train_model_select, outputs=train_output)
        
        # Tab 2: Evaluate
        with gr.TabItem("Evaluate"):
            gr.Markdown("### Evaluate on Test Set")
            eval_model_select = gr.Radio(
                choices=MODEL_NAMES,
                value="random_forest", 
                label="Select Model"
            )
            eval_btn = gr.Button("Run Evaluation", variant="primary")
            eval_output = gr.Textbox(label="Evaluation Results", lines=20)
            eval_btn.click(do_evaluate, inputs=eval_model_select, outputs=eval_output)
        
        # Tab 3: Predict
        with gr.TabItem("Predict"):
            gr.Markdown("### Predict Disease")
            with gr.Row():
                with gr.Column():
                    symptom_select = gr.CheckboxGroup(
                        choices=SYMPTOMS,
                        label="Select Symptoms"
                    )
                with gr.Column():
                    predict_model_select = gr.Radio(
                        choices=MODEL_NAMES,
                        value="random_forest",
                        label="Select Model"
                    )
                    predict_btn = gr.Button("Predict", variant="primary")
                    predict_output = gr.Textbox(label="Prediction Results", lines=8)
            
            predict_btn.click(
                do_predict, 
                inputs=[symptom_select, predict_model_select],
                outputs=predict_output
            )
    
    gr.Markdown("---")
    gr.Markdown("*Disclaimer: This is for educational purposes only. Please consult a Doctor for actual medical advice.*")


if __name__ == "__main__":
    demo.launch(share=True)
