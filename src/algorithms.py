import numpy as np
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
    return selected_features

def rfe(X, y, model, n_features_to_select):
    feature_indices = list(range(X.shape[1]))

    while len(feature_indices) > n_features_to_select:
        model.fit(X[:, feature_indices], y)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError(f"Model {type(model).__name__} does not support feature importance computation.")

        least_important_index = np.argmin(importances)
        feature_indices.pop(least_important_index)

    return feature_indices

def stability_selection_library(X_train, y_train, model, num_features):
    model.fit(X_train, y_train)
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
    selected_features = np.argsort(perm_importance.importances_mean)[-num_features:]
    return selected_features

def rfe_library(X_train, y_train, model, num_features):
    rfe_selector = SklearnRFE(model, n_features_to_select=num_features)
    rfe_selector.fit(X_train, y_train)
    selected_features = np.where(rfe_selector.support_)[0]
    return selected_features

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









