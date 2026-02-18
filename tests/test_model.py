import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    data = load_breast_cancer()
    X = scaler.transform(data.data)
    y = data.target

    print("Max value:", X.max())
    print("Min value:", X.min())

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    assert acc > 0.85
