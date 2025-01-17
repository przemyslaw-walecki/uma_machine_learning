import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Stability Selection Implementation
def stability_selection(X, y, base_model, num_features, n_iter=50, sample_size=0.7):
    n_samples, n_features = X.shape
    feature_importance = np.zeros(n_features)

    for _ in range(n_iter):
        indices = np.random.choice(range(n_samples), size=int(sample_size * n_samples), replace=False)
        X_sample, y_sample = X[indices], y[indices]
        base_model.fit(X_sample, y_sample)

        if hasattr(base_model, "feature_importances_"):
            feature_importance += base_model.feature_importances_
        elif hasattr(base_model, "coef_"):
            feature_importance += np.abs(base_model.coef_).flatten()
        else:
            raise ValueError(f"Model {type(base_model).__name__} does not support feature importance computation.")

    feature_importance /= n_iter
    selected_features = np.argsort(feature_importance)[-num_features:]
    return selected_features, feature_importance

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Baseline model evaluation
start_time = time.time()
rf_model.fit(X_train, y_train)
y_pred_baseline = rf_model.predict(X_test)
time_baseline = time.time() - start_time

metrics_baseline = {
    "accuracy": accuracy_score(y_test, y_pred_baseline),
    "precision": precision_score(y_test, y_pred_baseline),
    "recall": recall_score(y_test, y_pred_baseline),
    "f1_score": f1_score(y_test, y_pred_baseline),
    "execution_time": time_baseline
}

# Stability Selection
num_features = 10
start_time = time.time()
selected_features_stability, stability_importance = stability_selection(X_train, y_train, rf_model, num_features)
time_stability = time.time() - start_time

X_train_stability = X_train[:, selected_features_stability]
X_test_stability = X_test[:, selected_features_stability]

# Train and evaluate model on selected features (Stability Selection)
rf_model.fit(X_train_stability, y_train)
y_pred_stability = rf_model.predict(X_test_stability)

metrics_stability = {
    "accuracy": accuracy_score(y_test, y_pred_stability),
    "precision": precision_score(y_test, y_pred_stability),
    "recall": recall_score(y_test, y_pred_stability),
    "f1_score": f1_score(y_test, y_pred_stability),
    "execution_time": time_stability
}

# Recursive Feature Elimination (RFE)
rfe = RFE(estimator=rf_model, n_features_to_select=num_features)
start_time = time.time()
rfe.fit(X_train, y_train)
time_rfe = time.time() - start_time

X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Train and evaluate model on selected features (RFE)
rf_model.fit(X_train_rfe, y_train)
y_pred_rfe = rf_model.predict(X_test_rfe)

metrics_rfe = {
    "accuracy": accuracy_score(y_test, y_pred_rfe),
    "precision": precision_score(y_test, y_pred_rfe),
    "recall": recall_score(y_test, y_pred_rfe),
    "f1_score": f1_score(y_test, y_pred_rfe),
    "execution_time": time_rfe
}

# Test with additional models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42)
}

additional_metrics = {}

for model_name, model in models.items():
    # Stability Selection
    try:
        start_time = time.time()
        selected_features_stability, _ = stability_selection(X_train, y_train, model, num_features)
        X_train_stability = X_train[:, selected_features_stability]
        X_test_stability = X_test[:, selected_features_stability]
        model.fit(X_train_stability, y_train)
        y_pred = model.predict(X_test_stability)
        time_model = time.time() - start_time

        additional_metrics[model_name + " Stability Selection"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "execution_time": time_model
        }
    except ValueError as e:
        additional_metrics[model_name + " Stability Selection"] = str(e)

    # RFE
    start_time = time.time()
    rfe = RFE(estimator=model, n_features_to_select=num_features)
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe, y_train)
    y_pred = model.predict(X_test_rfe)
    time_model = time.time() - start_time

    additional_metrics[model_name + " RFE"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "execution_time": time_model
    }

# Visualization
plt.figure(figsize=(14, 6))

# Feature importance from Stability Selection
plt.subplot(1, 2, 1)
plt.barh(range(len(stability_importance)), sorted(stability_importance))
plt.title("Feature Importance (Stability Selection)")
plt.xlabel("Importance")
plt.ylabel("Features")

# Feature ranking from RFE
plt.subplot(1, 2, 2)
plt.barh(range(len(rfe.ranking_)), sorted(rfe.ranking_))
plt.title("Feature Ranking (RFE)")
plt.xlabel("Rank")
plt.ylabel("Features")

plt.tight_layout()
plt.show()

# Print metrics
print("Metrics for Baseline Model:")
for key, value in metrics_baseline.items():
    print(f"{key.capitalize()}: {value:.4f}")

print("\nMetrics for Stability Selection:")
for key, value in metrics_stability.items():
    print(f"{key.capitalize()}: {value:.4f}")

print("\nMetrics for Recursive Feature Elimination (RFE):")
for key, value in metrics_rfe.items():
    print(f"{key.capitalize()}: {value:.4f}")

print("\nMetrics for Additional Models:")
for model_name, metrics in additional_metrics.items():
    print(f"\n{model_name}:")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")
    else:
        print(metrics)
