import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def train():
    # Ensure MLflow uses a relative path for tracking URI (important for CI)
    # mlflow.set_tracking_uri("mlruns")
    
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)

    mlflow.set_experiment("Breast Cancer Classification")

    with mlflow.start_run():
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_test_scaled[:5]
        )

        print(f"Model trained with accuracy: {accuracy}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save BOTH model and scaler
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")


if __name__ == "__main__":
    train()
