# Initial Project Documentation

## Heuristic Algorithms for Irrelevant Feature Elimination Based on Higher-Order Dependencies

### Project Objective
The objective of this project is to implement two heuristic feature elimination algorithms: **Stability Selection** and **Recursive Feature Elimination (RFE)**, considering higher-order dependencies. These algorithms will be tested on the Breast Cancer dataset from `scikit-learn` using the Random Forest machine learning model.

---

### Dataset
The **Breast Cancer** dataset from `scikit-learn` will be used to test the algorithms. This dataset contains information about breast tumor characteristics and their classification as malignant (1) or benign (0).

#### Dataset Characteristics:
- **30 features** describing various tumor properties such as radius, texture, perimeter, area, etc.
- **569 samples** (rows), each assigned to one of two classes:
  - 0 (benign)
  - 1 (malignant)

---

### Algorithms

#### 1. Stability Selection
Stability Selection is an algorithm that eliminates features based on their stability. The algorithm works as follows:
- Perform multiple random sampling of data subsets.
- Train the model (in this case, Random Forest) on these subsets.
- Features frequently selected as important across different samples are considered relevant.
- Features rarely selected are eliminated.

#### 2. Recursive Feature Elimination (RFE)
RFE is an algorithm that iteratively removes the least important features by evaluating their impact on classification performance. The algorithm works as follows:
- Train the model on the full feature set.
- Iteratively remove features with the least influence on the model's outcome.
- Repeat the process until the desired number of features is reached.

---

### Solution Outline

1. **Data Preparation**:
   - Split the dataset into training data (80%) and testing data (20%).

2. **Model**:
   - **Random Forest** will be used as the baseline model for both algorithms.

3. **Algorithm Execution**:
   - **Stability Selection**: Evaluate feature stability using multiple random samples, selecting the most frequently chosen features.
   - **RFE**: Iteratively remove the least important features until the optimal feature subset is obtained.

4. **Model Evaluation**:
   - Assess accuracy, precision, recall, and F1-score on the test set.
   - Compare model performance before and after feature elimination.

5. **Visualization**:
   - Feature elimination results will be visualized using `matplotlib` to show feature importance before and after elimination.

---

### Experiments

#### 1. Impact of Feature Count on Model Quality
- Execute Stability Selection and RFE with varying numbers of features.
- Evaluate accuracy, precision, recall, and F1-score for each configuration.
- Analyze the impact of feature elimination on training time and model performance.

#### 2. Stability Selection: Effect of Iteration Count and Sample Size
- Test Stability Selection with different iteration counts.
- Vary the size of random data samples and evaluate feature selection stability.

#### 3. Comparison of Stability Selection and RFE
- Execute both algorithms on the same dataset.
- Compare the selected features, model accuracy, and execution time.
- Analyze performance differences between the two algorithms.

#### 4. Experiments with Other Models
- Evaluate the algorithms with other pre-trained models to verify the generalizability of the selected features.
- Compare results with existing Stability Selection and RFE implementations.

---

### Libraries Used
- **scikit-learn**: Used for loading the dataset and training models (Random Forest).
- **matplotlib**: Used for visualizing results, including feature importance and comparison charts before and after elimination.
- **numpy**: Used for numerical data manipulation.
- **pandas**: May be used for data analysis if necessary, but the dataset is compatible with `scikit-learn` directly.

---

### Expected Outcomes
After completing the experiments, the Stability Selection and RFE algorithms should:
- Identify the most influential features for breast cancer classification.
- Improve model efficiency by eliminating irrelevant features.
- Present the results visually in a clear and analyzable format.
