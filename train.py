import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split the dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target,
)

# Train the model and record the experiment
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

mlflow.set_experiment("Iris-Classification")

with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        artifact_path="iris_model",
        serialization_format="pickle",
    )

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"MLflow run ID: {run.info.run_id}")

# Save files for the prediction API
joblib.dump(model, "model.joblib")
joblib.dump(iris.target_names, "target_names.joblib")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(iris.target_names, "models/target_names.joblib")

print("Training and MLflow logging completed.")
