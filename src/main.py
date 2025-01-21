from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from plot_functions import plot_iterations, plot_analysis_results, plot_feature_impact
from utils import analyze_features_impact_for_models, analyze_sample_size_impact


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    num_features_list = list(range(1, 29, 2))

    feature_impact_results = analyze_features_impact_for_models(
        X, y, models, num_features_list
    )
    plot_iterations(X, y, models["Random Forest"])
    plot_feature_impact(feature_impact_results, num_features_list)
    results = analyze_sample_size_impact(
        X, y, models["Random Forest"], 10, feature_names, num_samples=10)
    plot_analysis_results(results)


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