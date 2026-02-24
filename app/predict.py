"""
Prediction utilities for breast cancer classification
"""

import joblib
import numpy as np
from typing import List, Dict, Tuple


class BreastCancerPredictor:
    """Wrapper class for breast cancer prediction"""
    
    def __init__(self, model_path: str = "models/model.pkl", 
                 scaler_path: str = "models/scaler.pkl"):
        """
        Initialize predictor with model and scaler
        
        Args:
            model_path: Path to saved model file
            scaler_path: Path to saved scaler file
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.labels = {0: "malignant", 1: "benign"}
    
    def predict(self, features: List[float]) -> Dict:
        """
        Make prediction on input features
        
        Args:
            features: List of 30 feature values
            
        Returns:
            Dictionary with prediction results
        """
        # Validate input
        if len(features) != 30:
            raise ValueError(f"Expected 30 features, got {len(features)}")
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return {
            "prediction": int(prediction),
            "prediction_label": self.labels[int(prediction)],
            "confidence": float(probabilities[prediction]),
            "probabilities": {
                "malignant": float(probabilities[0]),
                "benign": float(probabilities[1])
            }
        }
    
    def predict_batch(self, features_list: List[List[float]]) -> List[Dict]:
        """
        Make predictions on multiple samples
        
        Args:
            features_list: List of feature lists
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(features) for features in features_list]


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = BreastCancerPredictor()
    
    # Example features (malignant case)
    malignant_features = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
        0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
        0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    
    # Make prediction
    result = predictor.predict(malignant_features)
    
    print("Prediction Results:")
    print(f"  Class: {result['prediction']} ({result['prediction_label']})")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    print(f"    Malignant: {result['probabilities']['malignant']:.4f}")
    print(f"    Benign: {result['probabilities']['benign']:.4f}")