--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Filter Based Method
*          What is VarianceThreshold?
VarianceThreshold is a simple, unsupervised feature selection method provided by scikit-learn. It removes features with low variance, helping to clean and simplify datasets by eliminating columns that are not informative. This makes it an essential preprocessing step for machine learning tasks, especially when dealing with high-dimensional data.

![image](https://github.com/user-attachments/assets/866ded32-2df8-4689-99d3-ce15a869b513)
How It Works
Variance measures the spread of data in a feature:

High variance: Indicates significant differences across samples.
Low variance: Indicates little to no difference across samples.
For example:

A column with all 1s has zero variance.
A column with [1, 2, 3] has variance.
VarianceThreshold removes columns whose variance is below the specified threshold, leaving only the most variable and potentially useful features.
![image](https://github.com/user-attachments/assets/fe66e9c6-6e86-4bff-97a0-1984d935435f)
![image](https://github.com/user-attachments/assets/88816454-671d-41da-bce2-e1c9ed84f520)
![image](https://github.com/user-attachments/assets/7bbbc44e-9006-4537-95be-6ab4b13a5951)
![image](https://github.com/user-attachments/assets/76a32b60-5627-4698-b126-42a5e05e82b3)
![image](https://github.com/user-attachments/assets/0336a1f3-c16d-4cc1-8e2a-b62b993d546c)
![image](https://github.com/user-attachments/assets/690b9a85-8e23-48f5-b714-680934820460)
![image](https://github.com/user-attachments/assets/af4d7884-7589-48e7-8ed2-49017e396a73)
`When to Use VarianceThreshold?`\
`Preprocessing for Model Training:`
To remove noisy or redundant features that add no value to predictions.\
`Clustering or PCA:`
As a cleaning step before applying dimensionality reduction methods like PCA or clustering.\
`Unsupervised Learning:`
When you don't have labels (y) and need to preprocess the feature set.\
`Feature Engineering:`
Simplify the dataset for manual feature engineering or visualization.\
`Use Cases`\
`Handling Datasets with Constant or Near-Constant Features:`

For example, a column with [1, 1, 1, 1] contributes no information to the model.
Improving Computational Efficiency:

Reducing the dimensionality speeds up algorithms that scale with the number of features.
![image](https://github.com/user-attachments/assets/d7832245-607b-4124-b844-9b6834eedd71)


*        SelectKBest
The SelectKBest feature selection method from scikit-learn selects features with the highest scores based on a scoring function you define. Here's a summary of its components:

Purpose
SelectKBest selects the top k features that have the highest scores according to a scoring function. This is commonly used in preprocessing to reduce the dimensionality of your dataset, helping focus only on the most relevant features.

Key Parameters
score_func (callable, default=f_classif):
A function that calculates scores for features. Examples:

chi2: Chi-squared stats (for non-negative data, often for classification tasks).
f_classif: ANOVA F-value between feature and target (classification).\
f_regression: F-value for regression.\
mutual_info_classif: Mutual information for classification.\
mutual_info_regression: Mutual information for regression.\
k (int or "all", default=10):
Number of features to select. Setting "all" will keep all features.

Attributes
scores_: Array of scores for each feature.
pvalues_: P-values for the scores (if provided by the scoring function).
n_features_in_: Number of input features during the fit.
feature_names_in_: Names of the input features (if available).

![image](https://github.com/user-attachments/assets/987a5559-f08a-4f36-b0c6-51eb9fdbd4df)
![image](https://github.com/user-attachments/assets/bbc26ec8-d09d-48fc-abeb-d54b576c86f1)
![image](https://github.com/user-attachments/assets/c823ade5-2405-44ed-bba9-05c22212ba10)
![image](https://github.com/user-attachments/assets/fddb4c4c-82b7-4fca-8846-5f334678959e)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




# Wrapper Methods
* 
![image](https://github.com/user-attachments/assets/234f865c-d838-483d-846a-2a8d0c21edc8)
ExhaustiveFeatureSelector

Exhaustive Feature Selection for Classification and Regression. (new in v0.4.3)

Parameters:
- estimator: scikit-learn classifier or regressor
- min_features: int (default=1) - Minimum number of features to select
- max_features: int (default=1) - Maximum number of features to select
- print_progress: bool (default=True) - Prints progress as the number of epochs to stderr
- scoring: str (default='accuracy') - Scoring metric for evaluation. Options for classifiers: {'accuracy', 'f1', 'precision', 'recall', 'roc_auc'}, for regressors: {'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2'}, or a callable scorer.
- cv: int (default=5) - Cross-validation generator or int for k-fold cross-validation.
- n_jobs: int (default=1) - Number of CPUs for parallel processing. -1 means 'use all CPUs'.
- pre_dispatch: int, str (default='2*n_jobs') - Controls the number of jobs dispatched during parallel execution. 
- clone_estimator: bool (default=True) - Clones estimator if True; works with the original estimator if False.
- fixed_features: tuple (default=None) - Indices of fixed features that are always included in the selection.
- feature_groups: list or None (default=None) - Feature groups that are always selected together.

Attributes:
- best_idx_: array-like, shape = [n_predictions] - Indices of the selected feature subsets.
- best_feature_names_: array-like, shape = [n_predictions] - Names of the selected feature subsets.
- best_score_: float - Cross-validation average score of the selected subset.
- subsets_: dict - Dictionary of selected feature subsets with details such as feature indices, feature names, individual CV scores, and average scores.

Notes:
- If `feature_groups` is not None, the number of features is equal to the number of feature groups. Example: feature_groups=[[0], [1], [2, 3], [4]] means max_features cannot exceed 4.
- The features within a group may not have the same impact on the outcome (e.g., linear regression coefficients for features 2 and 3 can differ even if grouped together).
- If both `fixed_features` and `feature_groups` are provided, make sure `fixed_features` is included in the feature groups.

Methods:
- finalize_fit(): Finalizes the fit after an interruption, e.g., KeyboardInterrupt.
- fit(X, y, groups=None, fit_params): Fits the model on training data and performs feature selection.
  - X: Training data
  - y: Target values
  - groups: Optional, group labels for cross-validation
  - fit_params: Optional, parameters for the estimator's fit method.
- fit_transform(X, y, groups=None, fit_params): Fits and returns the selected features.
- get_metric_dict(confidence_interval=0.95): Returns a dictionary with metrics such as individual CV scores, average score, standard deviation, standard error, and confidence interval bounds.
- get_params(deep=True): Retrieves the parameters for the estimator.
- set_params(params): Sets the parameters for the estimator.
- transform(X): Returns the selected features from the input data X.

Examples:
For usage examples, please see: https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/


* # `SequentialFeatureSelector`
  ![image](https://github.com/user-attachments/assets/b2373921-90bd-42eb-ac11-a8425a415ca5)
  ![image](https://github.com/user-attachments/assets/bba3ecc0-afe8-43bf-852b-a840829c01a2)
  ![image](https://github.com/user-attachments/assets/e96308d1-cffa-48a8-908c-bc34e9030d8e)


