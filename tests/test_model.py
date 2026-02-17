import joblib
import numpy as np


def test_prediction_shape():
    model = joblib.load("models/model.pkl")
    sample = np.random.rand(1, 30)
    prediction = model.predict(sample)
    assert prediction.shape == (1,)
