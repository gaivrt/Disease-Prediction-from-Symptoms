"""Evaluation module for disease prediction models."""
import os
import sys
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.loader import DataLoader
from models import MODELS


def evaluate_model(model_name):
    """
    Evaluate a trained model on the test dataset.
    
    Args:
        model_name: Key from MODELS dict
        
    Returns:
        dict: Evaluation results
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name]
    
    # Load model
    model_path = os.path.join('../saved_model', model_class.filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please train first.")
    
    clf = load(model_path)
    
    # Load test data
    loader = DataLoader()
    _, _, X_test, y_test, encoder = loader.load_data()
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Calculate metrics using sklearn
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0)
    
    results = {
        'model_name': model_class.name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'classification_report': report,
        'test_samples': len(X_test),
    }
    
    return results


def evaluate_all_models():
    """Evaluate all trained models."""
    results = {}
    for name in MODELS.keys():
        try:
            print(f"\nEvaluating {name}...")
            results[name] = evaluate_model(name)
            print(f"  Accuracy: {results[name]['accuracy']:.4f}")
            print(f"  Macro F1: {results[name]['f1_macro']:.4f}")
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
    return results


if __name__ == "__main__":
    evaluate_all_models()
