import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE as SklearnRFE

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def stability_selection(X, y, base_model, num_features, n_iter=10, sample_size=0.7):
    n_samples, n_features = X.shape
    feature_importance = np.zeros(n_features)

    for _ in range(n_iter):
        indices = np.random.choice(range(n_samples), size=int(sample_size * n_samples), replace=False)
        base_model.fit(X[indices], y[indices])
        if hasattr(base_model, "feature_importances_"):
            feature_importance += base_model.feature_importances_
        elif hasattr(base_model, "coef_"):
            feature_importance += np.abs(base_model.coef_).flatten()
        else:
            raise ValueError(f"{type(base_model).__name__} does not support feature importance computation.")

    selected_features = np.argsort(feature_importance)[-num_features:]
    return selected_features, feature_importance / n_iter

def rfe(X, y, model, n_features_to_select, feature_names):
    feature_indices = list(range(X.shape[1]))
    deleted_features = []

    while len(feature_indices) > n_features_to_select:
        model.fit(X[:, feature_indices], y)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError(f"Model {type(model).__name__} does not support feature importance computation.")

        least_important_index = np.argmin(importances)
        deleted_features.append({
            "feature_index": feature_indices[least_important_index],
            "feature_name": feature_names[feature_indices[least_important_index]]
        })
        feature_indices.pop(least_important_index)

    return feature_indices, deleted_features

def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    execution_time = time.time() - start_time

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "execution_time": execution_time,
    }

def print_metrics(title, metrics):
    print(f"\n{title}:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  Execution Time: {metrics['execution_time']:.4f} seconds")

def print_deleted_features(title, deleted_features):
    print(f"\n{title}:")
    for feature in deleted_features:
        print(f"  {feature['feature_name']}")

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    rf_model = RandomForestClassifier(random_state=42)
    baseline_metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
    print_metrics("Baseline Metrics", baseline_metrics)

    # Number of features to select
    num_features = 10

    # Custom Stability Selection
    start_time = time.time()
    selected_features_stability, feature_importance = stability_selection(X_train, y_train, rf_model, num_features)
    time_stability = time.time() - start_time
    X_train_selected_stability = X_train[:, selected_features_stability]
    X_test_selected_stability = X_test[:, selected_features_stability]
    stability_metrics = evaluate_model(
        rf_model,
        X_train_selected_stability,
        X_test_selected_stability,
        y_train,
        y_test,
    )
    print_metrics("Custom Stability Selection Metrics", stability_metrics)
    print(f"Custom Stability Selection Execution Time: {time_stability:.4f} seconds")

    # Deleted features for custom stability selection
    deleted_features_stability = [
        {"feature_index": idx, "feature_name": feature_names[idx], "importance": feature_importance[idx]}
        for idx in range(len(feature_importance)) if idx not in selected_features_stability
    ]
    print_deleted_features("Deleted Features (Custom Stability Selection)", deleted_features_stability)

    # Custom RFE
    start_time = time.time()
    selected_features_rfe, deleted_features_rfe = rfe(X_train, y_train, rf_model, num_features, feature_names)
    time_rfe = time.time() - start_time
    X_train_selected_rfe = X_train[:, selected_features_rfe]
    X_test_selected_rfe = X_test[:, selected_features_rfe]
    rfe_metrics = evaluate_model(
        rf_model,
        X_train_selected_rfe,
        X_test_selected_rfe,
        y_train,
        y_test,
    )
    print_metrics("Custom RFE Metrics", rfe_metrics)
    print(f"Custom RFE Execution Time: {time_rfe:.4f} seconds")
    print_deleted_features("Deleted Features (Custom RFE)", deleted_features_rfe)

    # Stability Selection (Library-based) using Permutation Importance
    start_time = time.time()
    perm_importance = permutation_importance(rf_model, X_train_selected_stability, y_train, n_repeats=10, random_state=42)
    stability_selected_features_lib = np.argsort(perm_importance.importances_mean)[-num_features:]
    X_train_selected_lib = X_train_selected_stability[:, stability_selected_features_lib]
    X_test_selected_lib = X_test_selected_stability[:, stability_selected_features_lib]
    stability_metrics_lib = evaluate_model(rf_model, X_train_selected_lib, X_test_selected_lib, y_train, y_test)
    time_stability = time.time() - start_time
    print_metrics("Stability Selection (Library) Metrics", stability_metrics_lib)
    print(f"Stability Selection (Library) Execution Time: {time_stability:.4f} seconds")

    # RFE (Library-based)
    start_time = time.time()
    rfe_selector = SklearnRFE(rf_model, n_features_to_select=num_features)
    rfe_selector.fit(X_train_selected_rfe, y_train)
    rfe_selected_features_lib = rfe_selector.support_
    X_train_selected_rfe_lib = X_train_selected_rfe[:, rfe_selected_features_lib]
    X_test_selected_rfe_lib = X_test_selected_rfe[:, rfe_selected_features_lib]
    rfe_metrics_lib = evaluate_model(rf_model, X_train_selected_rfe_lib, X_test_selected_rfe_lib, y_train, y_test)
    time_rfe_lib = time.time() - start_time
    print_metrics("RFE (Library) Metrics", rfe_metrics_lib)
    print(f"RFE (Library) Execution Time: {time_rfe_lib:.4f} seconds")

    # plt.figure(figsize=(10, 5))
    # plt.barh(range(len(feature_importance)), sorted(feature_importance), label="Stability Selection Importance")
    # plt.title("Feature Importance (Stability Selection)")
    # plt.xlabel("Importance")
    # plt.ylabel("Features")
    # plt.legend()
    # plt.savefig("stability_selection_feature_importance.png")


if __name__ == "__main__":
    main()
