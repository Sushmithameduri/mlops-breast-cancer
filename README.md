# 🧠 MLOps Breast Cancer Classification Pipeline

<p align="center">

<!-- Tech Stack Badges -->
<img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow&logoColor=white" />
<img src="https://img.shields.io/badge/SQLite-Database-lightgrey?logo=sqlite&logoColor=blue" />
<img src="https://img.shields.io/badge/Pytest-Testing-green?logo=pytest&logoColor=white" />
<img src="https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-black?logo=githubactions&logoColor=white" />
<img src="https://img.shields.io/badge/MLOps-Production%20Ready-red" />

</p>

An end-to-end **production-style MLOps project** that trains, validates, registers, and version-controls a machine learning model using MLflow and CI automation.

This project demonstrates real-world MLOps practices including:

- ✅ Reproducible training
- ✅ Feature scaling
- ✅ Model validation testing
- ✅ MLflow experiment tracking
- ✅ Model Registry with versioning
- ✅ SQLite backend (production-ready)
- ✅ CI pipeline validation with GitHub Actions
- ✅ Automated quality threshold enforcement

---

# 📌 Project Overview

We train a **Logistic Regression classifier** on the Breast Cancer dataset to predict malignant vs benign tumors.

Dataset source:
- `sklearn.datasets.load_breast_cancer`

Model:
- `LogisticRegression`
- `StandardScaler`
- Accuracy ≈ **97%**

---

# 🏗 Architecture

```
.
├── app/
│   └── train.py
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── tests/
│   └── test_model.py
├── mlflow.db
├── pytest.ini
└── README.md
```

---

# 🚀 How to Run

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Train the Model

```bash
python app/train.py
```

This will:

- Create `mlflow.db`
- Log experiment metrics
- Register model as `BreastCancerModel`
- Save artifacts in `/models`

Expected accuracy:

```
~0.97
```

---

## 3️⃣ Run Tests (Model Validation)

```bash
pytest -v
```

The test:

- Loads `model.pkl`
- Loads `scaler.pkl`
- Runs inference
- Computes accuracy
- Asserts accuracy > 0.85

This ensures:

✔ Model quality does not degrade  
✔ CI fails if performance drops  
✔ Artifacts are valid  

---

# 📊 MLflow Tracking

Tracking backend:

```
sqlite:///mlflow.db
```

Launch UI:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

You can view:

- Parameters
- Metrics
- Artifacts
- Model versions
- Registry lifecycle stages

---

# 🧪 CI Integration

GitHub Actions pipeline:

- Installs dependencies
- Trains model
- Runs pytest
- Fails if:
  - Accuracy < threshold
  - Artifacts missing
  - Code errors occur

Green check = production-safe model.

---

# 🏆 MLOps Capabilities Demonstrated

### ✔ Model Registry
Version-controlled ML lifecycle management.

### ✔ Reproducibility
Fixed seed, controlled solver, SQLite tracking backend.

### ✔ Performance Gating
Automated accuracy threshold enforcement in CI.

### ✔ Artifact Management
Model and scaler saved independently to ensure proper inference.

---

# 🛠 Tech Stack

- Python
- scikit-learn
- MLflow
- SQLite
- Pytest
- GitHub Actions
- Git
- MLOps Best Practices

---

# 🔮 Future Enhancements

- Docker containerization
- REST API deployment
- Auto-promotion to Production based on metrics
- Drift detection
- Data versioning
- Model comparison across versions

---

# 👩‍💻 Author

**Sushmitha**  
MS in Data Science  
Building reliable, production-grade ML systems.
