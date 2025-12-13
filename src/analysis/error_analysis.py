import torch
import numpy as np
import pandas as pd
import sys
import os
import joblib

# Ensure imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import DataLoader
from src.models.symptom_net import SymptomNet
from src.safety.ood_detector import OODDetector

def analyze_errors(model_type='pytorch'):
    # 1. Load Data
    loader = DataLoader()
    _, _, X_test, y_test, encoder = loader.load_data()
    classes = encoder.classes_
    
    print(f"Analyzing errors on {len(X_test)} test samples...")
    
    # 2. Load Model & Predict
    misclassified = []
    
    if model_type == 'pytorch':
        # Use the OOD Detector class as a wrapper for convenience
        detector = OODDetector()
        
        # Convert test data to tensor
        X_test_tensor = torch.FloatTensor(X_test.values)
        
        with torch.no_grad():
            outputs = detector.model(X_test_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            # Identify Bad Cases
            preds_np = preds.numpy()
            
            for i in range(len(y_test)):
                true_label_idx = y_test[i]
                pred_label_idx = preds_np[i]
                
                if true_label_idx != pred_label_idx:
                    # Get active symptoms
                    sample = X_test.iloc[i]
                    active_symptoms = sample[sample == 1].index.tolist()
                    
                    misclassified.append({
                        'Index': i,
                        'True Label': classes[true_label_idx],
                        'Predicted': classes[pred_label_idx],
                        'Confidence': probs[i][pred_label_idx].item(),
                        'Active Symptoms': active_symptoms
                    })
    
    # 3. Report
    if len(misclassified) == 0:
        print("\n🎉 Amazing! Zero errors found on the test set. (Accuracy 100%)")
        print("Note: In a real world scenario, verify if test set is too small or leaked.")
    else:
        print(f"\nFound {len(misclassified)} misclassified samples (Error Rate: {len(misclassified)/len(y_test):.2%}):")
        print("-" * 60)
        for case in misclassified:
            print(f"Case #{case['Index']}")
            print(f"  True Disease : {case['True Label']}")
            print(f"  AI Predicted : {case['Predicted']} (Conf: {case['Confidence']:.2f})")
            print(f"  Symptoms     : {', '.join(case['Active Symptoms'])}")
            print("-" * 60)
            
        # Optional: Save to CSV for report
        df_errors = pd.DataFrame(misclassified)
        os.makedirs('./output', exist_ok=True)
        df_errors.to_csv('./output/error_analysis.csv', index=False)
        print("Detailed error report saved to ./output/error_analysis.csv")

if __name__ == "__main__":
    analyze_errors()
