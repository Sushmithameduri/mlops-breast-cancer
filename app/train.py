import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
from datetime import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


def train():
    """
    Train breast cancer classification model with MLflow tracking
    """
    print("=" * 60)
    print("BREAST CANCER CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Set MLflow tracking URI to a local SQLite database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    print("\n[1/6] Loading Dataset...")
    data = load_breast_cancer()
    print(f"    ✓ Loaded {len(data.data)} samples with {data.data.shape[1]} features")
    print(f"    ✓ Classes: {data.target_names}")
    
    print("\n[2/6] Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    print(f"    ✓ Training set: {len(X_train)} samples")
    print(f"    ✓ Test set: {len(X_test)} samples")
    
    print("\n[3/6] Scaling Features...")
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"    ✓ Features scaled (mean=0, std=1)")
    print(f"    ✓ Scaled data range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    print("\n[4/6] Training Model...")
    # Model hyperparameters
    model_params = {
        'max_iter': 2000,
        'solver': 'liblinear',
        'random_state': 42,
        'C': 1.0  # Regularization strength
    }
    
    model = LogisticRegression(**model_params)
    
    # Set experiment
    mlflow.set_experiment("Breast Cancer Classification")
    
    with mlflow.start_run():
        # Train model
        model.fit(X_train_scaled, y_train)
        print(f"    ✓ Model training completed")
        
        print("\n[5/6] Evaluating Model...")
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics for both train and test sets
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        
        # Print metrics
        print(f"\n    Training Metrics:")
        print(f"      Accuracy:  {train_accuracy:.4f}")
        
        print(f"\n    Test Metrics:")
        print(f"      Accuracy:  {test_accuracy:.4f}")
        print(f"      Precision: {test_precision:.4f}")
        print(f"      Recall:    {test_recall:.4f}")
        print(f"      F1 Score:  {test_f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\n    Confusion Matrix:")
        print(f"      [[TN={cm[0,0]}, FP={cm[0,1]}],")
        print(f"       [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        # Classification Report
        print(f"\n    Classification Report:")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=data.target_names))
        
        print("\n[6/6] Logging to MLflow...")
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", model_params['solver'])
        mlflow.log_param("max_iter", model_params['max_iter'])
        mlflow.log_param("C", model_params['C'])
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        
        # Log confusion matrix values
        mlflow.log_metric("true_negatives", int(cm[0,0]))
        mlflow.log_metric("false_positives", int(cm[0,1]))
        mlflow.log_metric("false_negatives", int(cm[1,0]))
        mlflow.log_metric("true_positives", int(cm[1,1]))
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test_scaled[:5],
            registered_model_name="BreastCancerModel"
        )
        print(f"    ✓ Model logged to MLflow")
        print(f"    ✓ Parameters logged: {len(model_params) + 3} params")
        print(f"    ✓ Metrics logged: 9 metrics")
    
    # Save model and scaler locally
    print("\n[7/6] Saving Model Artifacts...")
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    print(f"    ✓ Model saved to {model_path}")
    
    # Save scaler
    scaler_path = "models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"    ✓ Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "LogisticRegression",
        "scaler_type": "StandardScaler",
        "hyperparameters": model_params,
        "metrics": {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1_score": float(test_f1)
        },
        "confusion_matrix": {
            "true_negatives": int(cm[0,0]),
            "false_positives": int(cm[0,1]),
            "false_negatives": int(cm[1,0]),
            "true_positives": int(cm[1,1])
        },
        "dataset": {
            "n_samples": len(data.data),
            "n_features": data.data.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "classes": data.target_names.tolist()
        }
    }
    
    import json
    metadata_path = "models/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    ✓ Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"   Test Accuracy:  {test_accuracy:.2%}")
    print(f"   Test Precision: {test_precision:.2%}")
    print(f"   Test Recall:    {test_recall:.2%}")
    print(f"   Test F1 Score:  {test_f1:.2%}")
    
    print(f"\n📁 Artifacts saved:")
    print(f"   • {model_path}")
    print(f"   • {scaler_path}")
    print(f"   • {metadata_path}")
    
    print(f"\n🔍 View MLflow UI:")
    print(f"   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"   Open: http://localhost:5000")
    print("=" * 60 + "\n")
    
    return model, scaler, test_accuracy


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    try:
        train()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)