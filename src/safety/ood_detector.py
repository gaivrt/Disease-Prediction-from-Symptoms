import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.symptom_net import SymptomNet
from src.data.loader import DataLoader

class OODDetector:
    def __init__(self, model_path='./saved_model/symptom_net.pth', threshold=1.5):
        self.threshold = threshold
        self.device = torch.device('cpu')
        
        # Load Data Specs (needed to init model architecture)
        loader = DataLoader() # We just need the encoder classes really
        _, _, _, _, self.encoder = loader.load_data()
        self.classes = self.encoder.classes_
        
        # Initialize and Load Model
        self.model = SymptomNet(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_safe(self, feature_vector):
        """
        Predicts disease with safety check.
        Args:
            feature_vector (np.array or torch.Tensor): Input symptoms (1, 132)
        Returns:
            dict: {
                'prediction': str (Disease Name or 'Refer to Doctor'),
                'confidence': float,
                'entropy': float,
                'is_safe': bool
            }
        """
        if isinstance(feature_vector, np.ndarray):
            feature_vector = torch.FloatTensor(feature_vector)
            
        with torch.no_grad():
            logits = self.model(feature_vector)
            probs = F.softmax(logits, dim=1)
            
            # Calculate Entropy
            # H(x) = - sum(p(x) * log(p(x)))
            log_probs = torch.log(probs + 1e-10) # avoid log(0)
            entropy = -torch.sum(probs * log_probs, dim=1).item()
            
            # Get Prediction
            max_prob, predicted_idx = torch.max(probs, 1)
            predicted_label = self.classes[predicted_idx.item()]
            
            is_safe = entropy < self.threshold
            
            result = {
                'prediction': predicted_label if is_safe else "Refer to Doctor (Uncertain)",
                'confidence': max_prob.item(),
                'entropy': entropy,
                'is_safe': is_safe
            }
            
            return result

if __name__ == "__main__":
    # Test Code
    detector = OODDetector()
    
    # Case 1: Known Case (from Training Data - vaguely)
    # We don't have real data loaded here easily, so let's make a dummy vector
    # Ideally use a real test vector
    
    print("--- Safety Check Test ---")
    
    # Mocking a "High Confidence" vector (simulated)
    # Just random noise is likely OOD, let's see
    noise_vector = torch.randn(1, 132) 
    # Sigmoid to make it look like 0-1 features roughly after thresholding if we wanted, 
    # but let's just pass raw noise as the model expects float input.
    # Actually model expects 0/1 binary features mostly.
    
    res = detector.predict_safe(noise_vector)
    print(f"Random Noise Input -> Prediction: {res['prediction']} | Entropy: {res['entropy']:.4f}")
