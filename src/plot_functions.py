from statistics import mean, median

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.algorithms import stability_selection, evaluate_model


def plot_feature_impact(feature_impact_results, num_features_list, output_file="features_impact_comparison2.png"):
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics, start=1):
        plt.subplot(2, 3, i)
        for model_name, model_results in feature_impact_results.items():
            plt.plot(num_features_list, model_results[f"stability_{metric}"], label=f'{model_name} Stability', marker='o')
            plt.plot(num_features_list, model_results[f"rfe_{metric}"], label=f'{model_name} RFE', marker='o', linestyle='--')
        plt.title(f'{metric.capitalize()} Comparison Over Number of Features')
        plt.xlabel('Number of Features to Select')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

    plt.subplot(2, 3, 5)
    for model_name, model_results in feature_impact_results.items():
        plt.plot(num_features_list, model_results["stability_time"], label=f'{model_name} Stability', marker='o', linestyle='-')
        plt.plot(num_features_list, model_results["rfe_time"], label=f'{model_name} RFE', marker='o', linestyle='--')
    plt.title('Execution Time Over Number of Features')
    plt.xlabel('Number of Features to Select')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file)


def plot_iterations(X, y, base_model, output_file="iterations_comparison.png", num_samples=10, stat="mean"):
    max_iterations = 7
    num_features = 1
    sample_sizes = [0.3, 0.5, 0.7, 0.9]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(16, 4), sharey=True)
    fig.suptitle("Stability Selection: Iteration Comparison Across Sample Sizes", fontsize=16)

    for idx, sample_size in enumerate(sample_sizes):
        stability_scores_list = []

        for n_iter in range(1, max_iterations + 1):
            sampled_accuracies = []

            for _ in range(num_samples):
                selected_features = stability_selection(
                    X, y, base_model, num_features, n_iter=n_iter, sample_size=sample_size
                )
                X_train_selected = X_train[:, selected_features]
                X_test_selected = X_test[:, selected_features]

                stability_metrics = evaluate_model(
                    base_model,
                    X_train_selected,
                    X_test_selected,
                    y_train,
                    y_test,
                )
                sampled_accuracies.append(stability_metrics["accuracy"])

            if stat == "mean":
                stability_scores_list.append(mean(sampled_accuracies))
            elif stat == "median":
                stability_scores_list.append(median(sampled_accuracies))
            else:
                raise ValueError("Invalid statistic: choose 'mean' or 'median'")

        ax = axes[idx]
        ax.plot(range(1, max_iterations + 1), stability_scores_list, marker="o", label=f"{stat.capitalize()}")
        ax.set_title(f"Sample Size: {sample_size}")
        ax.set_xlabel("Number of Iterations")
        if idx == 0:
            ax.set_ylabel(f"Accuracy ({stat.capitalize()})")
        ax.grid(True)
        ax.legend(title="Statistic")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file)
    plt.close()


def plot_analysis_results(results, output_file="analysis_results.png"):
    sample_sizes = [result["sample_size"] for result in results]

    stability_accuracies = [result["stability_metrics"]["accuracy"] for result in results]
    rfe_accuracies = [result["rfe_metrics"]["accuracy"] for result in results]
    stability_times = [result["stability_time"] for result in results]
    rfe_times = [result["rfe_time"] for result in results]
    accuracy_differences = [result["differences"]["accuracy"] for result in results]
    time_differences = [result["differences"]["time"] for result in results]
    feature_overlaps = [result["differences"]["selected_features_overlap"] for result in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(sample_sizes, stability_accuracies, marker="o", label="Stability Accuracy")
    axes[0, 0].plot(sample_sizes, rfe_accuracies, marker="o", label="RFE Accuracy")
    axes[0, 0].set_title("Accuracy Comparison")
    axes[0, 0].set_xlabel("Sample Size")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(sample_sizes, stability_times, marker="o", label="Stability Time")
    axes[0, 1].plot(sample_sizes, rfe_times, marker="o", label="RFE Time")
    axes[0, 1].set_title("Execution Time Comparison")
    axes[0, 1].set_xlabel("Sample Size")
    axes[0, 1].set_ylabel("Time (seconds)")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(sample_sizes, accuracy_differences, marker="o", color="purple")
    axes[1, 0].set_title("Accuracy Differences (Stability - RFE)")
    axes[1, 0].set_xlabel("Sample Size")
    axes[1, 0].set_ylabel("Accuracy Difference")
    axes[1, 0].grid(True)

    axes[1, 1].plot(sample_sizes, feature_overlaps, marker="o", color="green")
    axes[1, 1].set_title("Feature Overlap Between Algorithms")
    axes[1, 1].set_xlabel("Sample Size")
    axes[1, 1].set_ylabel("Overlap Ratio")
    axes[1, 1].grid(True)

    plt.tight_layout()

    plt.savefig(output_file)
    plt.show()