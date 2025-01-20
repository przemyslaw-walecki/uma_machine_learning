import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

def rfe(X, y, model, n_features_to_select, feature_names):
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

def rfe_library(X_train, y_train, model, num_features, feature_names):
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

def print_metrics(title, metrics):
    print(f"\n{title}:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  Execution Time: {metrics['execution_time']:.4f} seconds")


def analyze_sample_size_impact(X, y, rf_model, num_features, feature_names):
    sample_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]
    results = []

    for size in sample_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - size, random_state=42
        )
        print(f"\n### Analyzing with {size * 100:.0f}% of data ({len(X_train)} samples) ###")
        
        start_time = time.time()
        selected_features_stability = stability_selection(X_train, y_train, rf_model, num_features)
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
        print_metrics("Stability Selection Metrics", stability_metrics)
        print(f"Stability Selection Execution Time: {time_stability:.4f} seconds")
 
        start_time = time.time()
        selected_features_rfe = rfe(X_train, y_train, rf_model, num_features, feature_names)
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
        print_metrics("RFE Metrics", rfe_metrics)
        print(f"RFE Execution Time: {time_rfe:.4f} seconds")

        results.append({
            "sample_size": size,
            "stability_metrics": stability_metrics,
            "rfe_metrics": rfe_metrics,
            "stability_time": time_stability,
            "rfe_time": time_rfe,
        })

    return results


def analyze_features_impact_for_models(X, y, models, feature_names, num_features_list, sample_size=0.8):
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1 - sample_size, random_state=42
    )

    for model_name, model in models.items():
        model_results = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "execution_time": [],
            "stability_times": [],
            "rfe_times": []
        }

        for num_features in num_features_list:
            start_time = time.time()
            selected_features_stability = stability_selection(X_train, y_train, model, num_features)
            time_stability = time.time() - start_time
            stability_metrics = evaluate_model(
                model,
                X_train[:, selected_features_stability],
                X_test[:, selected_features_stability],
                y_train,
                y_test,
            )

            start_time = time.time()
            selected_features_rfe = rfe(X_train, y_train, model, num_features, feature_names)
            time_rfe = time.time() - start_time
            rfe_metrics = evaluate_model(
                model,
                X_train[:, selected_features_rfe],
                X_test[:, selected_features_rfe],
                y_train,
                y_test,
            )


            model_results["accuracy"].append(stability_metrics["accuracy"])
            model_results["precision"].append(stability_metrics["precision"])
            model_results["recall"].append(stability_metrics["recall"])
            model_results["f1_score"].append(stability_metrics["f1_score"])
            model_results["execution_time"].append(stability_metrics["execution_time"])
            model_results["stability_times"].append(time_stability)

            model_results["accuracy"].append(rfe_metrics["accuracy"])
            model_results["precision"].append(rfe_metrics["precision"])
            model_results["recall"].append(rfe_metrics["recall"])
            model_results["f1_score"].append(rfe_metrics["f1_score"])
            model_results["execution_time"].append(rfe_metrics["execution_time"])
            model_results["rfe_times"].append(time_rfe)

        results[model_name] = model_results

    return results

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        #"Logistic Regression": LogisticRegression(random_state=42, max_iter=5000, solver='saga'),
    }

    num_features_list = list(range(1, 29, 2))
    feature_impact_results = analyze_features_impact_for_models(X, y, models, feature_names, num_features_list,)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    for model_name, model_results in feature_impact_results.items():
        plt.plot(num_features_list, model_results["accuracy"][:14], label=f'{model_name} Stability', marker='o')
        plt.plot(num_features_list, model_results["accuracy"][14:], label=f'{model_name} RFE', marker='o', linestyle='--')
    plt.title('Accuracy Comparison Over Number of Features')
    plt.xlabel('Number of Features to Select')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)


    plt.subplot(2, 3, 2)
    for model_name, model_results in feature_impact_results.items():
        plt.plot(num_features_list, model_results["precision"][:14], label=f'{model_name} Stability', marker='o')
        plt.plot(num_features_list, model_results["precision"][14:], label=f'{model_name} RFE', marker='o', linestyle='--')
    plt.title('Precision Comparison Over Number of Features')
    plt.xlabel('Number of Features to Select')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    for model_name, model_results in feature_impact_results.items():
        plt.plot(num_features_list, model_results["recall"][:14], label=f'{model_name} Stability', marker='o')
        plt.plot(num_features_list, model_results["recall"][14:], label=f'{model_name} RFE', marker='o', linestyle='--')
    plt.title('Recall Comparison Over Number of Features')
    plt.xlabel('Number of Features to Select')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    for model_name, model_results in feature_impact_results.items():
        plt.plot(num_features_list, model_results["f1_score"][:14], label=f'{model_name} Stability', marker='o')
        plt.plot(num_features_list, model_results["f1_score"][14:], label=f'{model_name} RFE', marker='o', linestyle='--')
    plt.title('F1 Score Comparison Over Number of Features')
    plt.xlabel('Number of Features to Select')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    for model_name, model_results in feature_impact_results.items():
        plt.plot(num_features_list, model_results["stability_times"][:14], label=f'{model_name} Stability', marker='o', linestyle='-')
        plt.plot(num_features_list, model_results["rfe_times"][:14], label=f'{model_name} RFE', marker='o', linestyle='--')
    plt.title('Execution Time Over Number of Features')
    plt.xlabel('Number of Features to Select')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("features_impact_comparison.png")

    # X, y = data.data, data.target
    # feature_names = data.feature_names
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # rf_model = RandomForestClassifier(random_state=42)

    # num_features = 20

    # start_time = time.time()
    # perm_importance = permutation_importance(rf_model, X_train, y_train, n_repeats=10, random_state=42)
    # stability_selected_features_lib = np.argsort(perm_importance.importances_mean)[-num_features:]
    # X_train_selected_lib = X_train[:, stability_selected_features_lib]
    # X_test_selected_lib = X_test[:, stability_selected_features_lib]
    # stability_metrics_lib = evaluate_model(rf_model, X_train_selected_lib, X_test_selected_lib, y_train, y_test)
    # time_stability = time.time() - start_time
    # print_metrics("Stability Selection (Library) Metrics", stability_metrics_lib)
    # print(f"Stability Selection (Library) Execution Time: {time_stability:.4f} seconds")

    # # RFE (Library-based)
    # start_time = time.time()
    # rfe_selector = SklearnRFE(rf_model, n_features_to_select=num_features)
    # rfe_selector.fit(X_train, y_train)
    # rfe_selected_features_lib = rfe_selector.support_
    # X_train_selected_rfe_lib = X_train[:, rfe_selected_features_lib]
    # X_test_selected_rfe_lib = X_test[:, rfe_selected_features_lib] 
    # rfe_metrics_lib = evaluate_model(rf_model, X_train_selected_rfe_lib, X_test_selected_rfe_lib, y_train, y_test)
    # time_rfe_lib = time.time() - start_time
    # print_metrics("RFE (Library) Metrics", rfe_metrics_lib)
    # print(f"RFE (Library) Execution Time: {time_rfe_lib:.4f} seconds")


if __name__ == "__main__":
    main()
