"""Training module for disease prediction models."""
import os
import sys
from joblib import dump

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.loader import DataLoader
from models import MODELS


def train_model(model_name):
    """
    Train a model on full training data and save it.
    
    Args:
        model_name: Key from MODELS dict ('naive_bayes', 'decision_tree', 'random_forest')
        
    Returns:
        dict: Training results
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name]
    
    # Load data
    loader = DataLoader()
    X_train, y_train, _, _, _ = loader.load_data()
    
    # Train on full training data
    clf = model_class.get_model()
    clf.fit(X_train, y_train)
    
    # Save model
    save_dir = '../saved_model'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_class.filename)
    dump(clf, save_path)
    
    results = {
        'model_name': model_class.name,
        'train_samples': len(X_train),
        'save_path': save_path,
    }
    
    return results


def train_all_models():
    """Train all available models."""
    results = {}
    for name in MODELS.keys():
        print(f"Training {name}...")
        results[name] = train_model(name)
        print(f"  Saved to: {results[name]['save_path']}")
    return results


if __name__ == "__main__":
    train_all_models()