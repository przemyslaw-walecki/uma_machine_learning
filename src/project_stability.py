import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def test_stability_selection_iterations(X_train, X_test, y_train, y_test, model, num_features, iterations_list):
    results = []
    for n_iter in iterations_list:
        selected_features = stability_selection(X_train, y_train, model, num_features, n_iter=n_iter)
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        
        accuracy = evaluate_model(model, X_train_selected, X_test_selected, y_train, y_test)
        results.append((n_iter, accuracy))
    
    return results

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)

    num_features = 10
    iterations_list = [5, 10, 20, 50, 100]

    results = test_stability_selection_iterations(X_train, X_test, y_train, y_test, model, num_features, iterations_list)

    iterations, accuracies = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, marker='o', linestyle='-', color='b', label='Accuracy vs Iterations')
    plt.title("Effect of Stability Selection Iterations on Accuracy")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("iteration_comparison.png")

if __name__ == "__main__":
    main()
