"""
Test suite for Breast Cancer Classification Model
Validates model loading, predictions, and performance metrics
"""

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)


def test_model_loading():
    """Test 1: Verify model and scaler files exist and load correctly"""
    print("=" * 60)
    print("TEST 1: Model and Scaler Loading")
    print("=" * 60)
    
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        
        assert model is not None, "Model is None"
        assert scaler is not None, "Scaler is None"
        
        print("✓ Model loaded successfully")
        print("✓ Scaler loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Scaler type: {type(scaler).__name__}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"✗ FAILED: Model files not found")
        print(f"  Error: {e}")
        print(f"  Run 'python train.py' first to create model files")
        return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_data_scaling():
    """Test 2: Verify data is properly scaled"""
    print("\n" + "=" * 60)
    print("TEST 2: Data Scaling Verification")
    print("=" * 60)
    
    try:
        scaler = joblib.load("models/scaler.pkl")
        
        # Load data
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        # Check scaling properties
        mean_val = X_test_scaled.mean()
        std_val = X_test_scaled.std()
        max_val = X_test_scaled.max()
        min_val = X_test_scaled.min()
        
        print(f"Scaled test data statistics:")
        print(f"  Mean:  {mean_val:.4f} (should be ~0)")
        print(f"  Std:   {std_val:.4f} (should be ~1)")
        print(f"  Max:   {max_val:.4f}")
        print(f"  Min:   {min_val:.4f}")
        
        # Verify scaling is correct (mean should be close to 0)
        assert abs(mean_val) < 1.0, f"Mean {mean_val} is too far from 0"
        
        print("✓ Data scaling is correct")
        return True
        
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_model_accuracy_on_test_set():
    """Test 3: Evaluate model on TEST SET ONLY (not entire dataset)"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Performance on Test Set")
    print("=" * 60)
    
    try:
        # Load model and scaler
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        
        # Load data and split (SAME as training)
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        print(f"Dataset info:")
        print(f"  Total samples: {len(data.data)}")
        print(f"  Test samples: {len(X_test)} (20%)")
        
        # Scale ONLY test data
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        # Assertions
        assert accuracy > 0.85, f"Accuracy {accuracy:.4f} is below threshold 0.85"
        assert recall > 0.80, f"Recall {recall:.4f} is too low (important for cancer detection)"
        
        print(f"\n✓ Accuracy exceeds threshold (>0.85)")
        print(f"✓ Recall is acceptable (>0.80)")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False


def test_single_prediction():
    """Test 4: Verify single sample prediction works"""
    print("\n" + "=" * 60)
    print("TEST 4: Single Sample Prediction")
    print("=" * 60)
    
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        
        # Create a random sample (30 features)
        sample = np.random.rand(1, 30) * 100
        
        print(f"Input sample shape: {sample.shape}")
        print(f"Input range: [{sample.min():.2f}, {sample.max():.2f}]")
        
        # Scale the sample
        sample_scaled = scaler.transform(sample)
        
        print(f"Scaled sample range: [{sample_scaled.min():.2f}, {sample_scaled.max():.2f}]")
        
        # Make prediction
        prediction = model.predict(sample_scaled)
        probabilities = model.predict_proba(sample_scaled)
        
        print(f"\nPrediction:")
        print(f"  Class: {prediction[0]} ({'benign' if prediction[0] == 1 else 'malignant'})")
        print(f"  Probabilities: [malignant={probabilities[0][0]:.4f}, benign={probabilities[0][1]:.4f}]")
        print(f"  Confidence: {probabilities[0][prediction[0]]:.4f}")
        
        # Assertions
        assert prediction[0] in [0, 1], f"Invalid prediction: {prediction[0]}"
        assert probabilities.shape == (1, 2), f"Invalid probability shape: {probabilities.shape}"
        assert abs(probabilities.sum() - 1.0) < 0.01, "Probabilities don't sum to 1"
        
        print(f"\n✓ Prediction is valid")
        print(f"✓ Probabilities sum to 1.0")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False


def test_prediction_with_actual_sample():
    """Test 5: Test with actual data samples"""
    print("\n" + "=" * 60)
    print("TEST 5: Prediction with Real Samples")
    print("=" * 60)
    
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        
        # Load actual data
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        # Take first 3 test samples
        samples = X_test[:3]
        true_labels = y_test[:3]
        
        # Scale and predict
        samples_scaled = scaler.transform(samples)
        predictions = model.predict(samples_scaled)
        probabilities = model.predict_proba(samples_scaled)
        
        print("Testing 3 real samples from test set:\n")
        for i in range(3):
            true_label = "benign" if true_labels[i] == 1 else "malignant"
            pred_label = "benign" if predictions[i] == 1 else "malignant"
            confidence = probabilities[i][predictions[i]]
            correct = "✓" if predictions[i] == true_labels[i] else "✗"
            
            print(f"Sample {i+1}:")
            print(f"  True label:    {true_label}")
            print(f"  Predicted:     {pred_label}")
            print(f"  Confidence:    {confidence:.2%}")
            print(f"  Result:        {correct} {'Correct' if predictions[i] == true_labels[i] else 'Incorrect'}")
            print()
        
        print("✓ Real sample predictions completed")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 60)
    print("RUNNING COMPLETE TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_model_loading,
        test_data_scaling,
        test_model_accuracy_on_test_set,
        test_single_prediction,
        test_prediction_with_actual_sample
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_func.__name__}: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results), 1):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status} - Test {i}: {test_func.__name__}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_all_tests()
    sys.exit(exit_code)