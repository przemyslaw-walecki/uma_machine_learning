import time
from collections import defaultdict
from statistics import mean

from sklearn.model_selection import train_test_split

from algorithms import stability_selection, evaluate_model, rfe


def print_metrics(title, metrics):
    print(f"\n{title}:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  Execution Time: {metrics['execution_time']:.4f} seconds")


def analyze_sample_size_impact(X, y, rf_model, num_features, feature_names, num_samples=5):
    sample_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]
    results = []

    for size in sample_sizes:
        print(f"\n### Analyzing with {size * 100:.0f}% of data ###")
        differences = {"accuracy": [], "time": [], "selected_features_overlap": []}
        stability_metrics_list = []
        rfe_metrics_list = []
        stability_times = []
        rfe_times = []

        for _ in range(num_samples):
            # Sample data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1 - size, random_state=None
            )

            # Stability Selection
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
            stability_metrics_list.append(stability_metrics)
            stability_times.append(time_stability)

            # Recursive Feature Elimination (RFE)
            start_time = time.time()
            selected_features_rfe = rfe(X_train, y_train, rf_model, num_features)
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
            rfe_metrics_list.append(rfe_metrics)
            rfe_times.append(time_rfe)

            # Compare results between Stability Selection and RFE
            accuracy_diff = stability_metrics["accuracy"] - rfe_metrics["accuracy"]
            time_diff = time_stability - time_rfe
            feature_overlap = len(
                set(selected_features_stability).intersection(set(selected_features_rfe))) / num_features

            differences["accuracy"].append(accuracy_diff)
            differences["time"].append(time_diff)
            differences["selected_features_overlap"].append(feature_overlap)

        # Aggregate metrics and differences
        def aggregate_metrics(metrics_list):
            aggregated = {k: mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
            return aggregated

        aggregated_stability_metrics = aggregate_metrics(stability_metrics_list)
        aggregated_rfe_metrics = aggregate_metrics(rfe_metrics_list)
        aggregated_differences = {k: mean(v) for k, v in differences.items()}

        # Print results
        print_metrics("Aggregated Stability Selection Metrics", aggregated_stability_metrics)
        print(f"Average Stability Selection Execution Time: {mean(stability_times):.4f} seconds")

        print_metrics("Aggregated RFE Metrics", aggregated_rfe_metrics)
        print(f"Average RFE Execution Time: {mean(rfe_times):.4f} seconds")

        print("\n### Comparison Results ###")
        print(f"Average Accuracy Difference (Stability - RFE): {aggregated_differences['accuracy']:.4f}")
        print(f"Average Time Difference (Stability - RFE): {aggregated_differences['time']:.4f} seconds")
        print(f"Average Selected Features Overlap: {aggregated_differences['selected_features_overlap']:.4f}")

        results.append({
            "sample_size": size,
            "stability_metrics": aggregated_stability_metrics,
            "rfe_metrics": aggregated_rfe_metrics,
            "stability_time": mean(stability_times),
            "rfe_time": mean(rfe_times),
            "differences": aggregated_differences,
        })

    return results


def analyze_features_impact_for_models(X, y, models, num_features_list, sample_size=0.8):
    results = defaultdict(lambda: defaultdict(list))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1 - sample_size, random_state=42
    )

    for model_name, model in models.items():
        for num_features in num_features_list:
            # Stability Selection
            start_time = time.time()
            selected_features_stability = stability_selection(X_train, y_train, model, num_features)
            stability_time = time.time() - start_time

            stability_metrics = evaluate_model(
                model,
                X_train[:, selected_features_stability],
                X_test[:, selected_features_stability],
                y_train,
                y_test,
            )

            # RFE Selection
            start_time = time.time()
            selected_features_rfe = rfe(X_train, y_train, model, num_features)
            rfe_time = time.time() - start_time

            rfe_metrics = evaluate_model(
                model,
                X_train[:, selected_features_rfe],
                X_test[:, selected_features_rfe],
                y_train,
                y_test,
            )

            # Record metrics for Stability Selection
            for metric, value in stability_metrics.items():
                results[model_name][f"stability_{metric}"].append(value)
            results[model_name]["stability_time"].append(stability_time)

            # Record metrics for RFE
            for metric, value in rfe_metrics.items():
                results[model_name][f"rfe_{metric}"].append(value)
            results[model_name]["rfe_time"].append(rfe_time)

    return results