# 2. Methodology

## 2.1 Study Design

This study uses an empirical benchmark design to compare TabPFN with three established gradient-boosted tree algorithms: XGBoost, LightGBM, and CatBoost. The central experimental variable is the amount of labelled training data available to each model.

The benchmark was conducted on three binary tabular classification datasets: Adult, Bank Marketing, and Credit-G. For each dataset, models were evaluated using multiple training-set sizes and five random seeds. This design makes it possible to examine both overall predictive performance and how model performance changes as the amount of available training data increases.

The final benchmark contains 1,040 model-level observations across the three datasets and four models.
## 2.2 Datasets

Three publicly available binary-classification datasets were used in the benchmark: Adult, Bank Marketing, and Credit-G. The datasets provide variation in dataset size, feature types, and classification characteristics while maintaining a common binary-classification setting.

### Adult

The Adult dataset is a binary-classification dataset containing demographic and employment-related attributes. The classification task is to predict whether an individual's annual income exceeds $50,000. The dataset contains both numerical and categorical variables, making it suitable for evaluating tabular models on heterogeneous data [8].

**Dataset source:** UCI Machine Learning Repository, Adult dataset (ID 2).
**Link:** https://archive.ics.uci.edu/dataset/2/adult

### Bank Marketing

The Bank Marketing dataset contains information related to direct marketing campaigns conducted by a Portuguese banking institution. The binary classification task is to predict whether a client subscribes to a term deposit. The dataset contains both categorical and numerical variables and therefore provides another heterogeneous tabular classification problem [9].

**Dataset source:** UCI Machine Learning Repository, Bank Marketing dataset (ID 222).
**Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

### Credit-G

Credit-G is a binary-classification dataset concerning credit risk. It contains heterogeneous tabular features and provides a smaller-scale financial classification problem than Adult and Bank Marketing. The dataset was accessed through OpenML using dataset ID 31 [10].

**Dataset source:** OpenML, Credit-G dataset (ID 31).
**Link:** https://www.openml.org/d/31

These datasets were selected to provide variation in dataset size, feature characteristics, and classification behaviour while maintaining a common binary-classification evaluation framework.


## 2.3 Train-Test Splitting

For each dataset, the available observations were divided into training and test sets using an 80:20 split.

The split was stratified according to the target variable so that the class distribution was preserved between the training and test partitions. The resulting test set was kept fixed for the benchmark, while different subsets of the training partition were used to evaluate model behaviour under different training-data regimes.

This design ensures that comparisons between training sizes are made against the same held-out evaluation data within each dataset.

## 2.4 Training-Data Regimes

To investigate how model performance changes with the amount of available labelled data, multiple training subset sizes were evaluated.

For Adult and Bank Marketing, the tested training sizes were:

10, 20, 35, 50, 75, 100, 150, 250, 500, 750, 1,000, 1,500, 2,000, 3,000, 4,000, 6,000, 8,000, 12,000, 16,000, 24,000, and 32,000 observations.

Credit-G has a smaller available training partition, and therefore the benchmark was evaluated at the subset sizes supported by that dataset, up to 750 observations.

Each training subset was generated using stratified sampling. Consequently, the class distribution of each subset was maintained as closely as possible to the corresponding training partition. This was particularly important for the smallest training sizes, where an unstratified random sample could contain only one target class and prevent binary classifiers from being trained.

Five random seeds were used:

42, 123, 2024, 777, and 999.

The use of repeated seeds reduces dependence on any single random subset and allows variability in model performance to be measured.

## 3.5 Compared Models

Four models were evaluated in the benchmark: XGBoost, LightGBM, CatBoost, and TabPFN. The first three are gradient-boosted decision-tree algorithms, while TabPFN is a pretrained transformer-based foundation model for tabular data. The models therefore represent two different approaches to tabular classification.

The following subsections describe the main learning principles of each model. The mathematical formulations describe the underlying algorithms and are not intended to imply that the experimental implementation manually optimized these equations.

### 3.5.1 XGBoost

XGBoost is a gradient-boosted decision-tree algorithm that constructs an ensemble of decision trees sequentially. At boosting iteration (t), the model is represented as

[
\hat{y}_i^{(t)}
===============

\hat{y}_i^{(t-1)}
+
f_t(x_i),
]

where (f_t) is the newly added decision tree.

The objective combines a loss function with a regularization term controlling the complexity of the newly added tree:

[
\mathcal{L}^{(t)}
=================

\sum_{i=1}^{n}
l\left(y_i,\hat{y}_i^{(t)}\right)
+
\Omega(f_t),
]

where (l) measures prediction error and (\Omega(f_t)) penalizes model complexity.

XGBoost uses a second-order Taylor approximation of the objective to efficiently determine the contribution of a new tree. Defining

[
g_i=
\frac{\partial l(y_i,\hat{y}_i^{(t-1)})}
{\partial \hat{y}_i^{(t-1)}}
]

and

[
h_i=
\frac{\partial^2 l(y_i,\hat{y}_i^{(t-1)})}
{\partial (\hat{y}_i^{(t-1)})^2},
]

the approximate objective for the new tree can be expressed using the first- and second-order gradients.

The resulting tree ensemble can therefore progressively reduce the prediction error by adding trees that focus on the residual structure of the current model. XGBoost also incorporates regularization and computational optimizations designed to improve scalability and generalization [4].

### 3.5.2 LightGBM

LightGBM is another gradient-boosted decision-tree algorithm, but it introduces specific mechanisms intended to improve training efficiency for large or high-dimensional datasets [5].

As with conventional gradient boosting, a sequence of trees is constructed so that the ensemble prediction can be written as

[
\hat{y}_i
=========

\sum_{t=1}^{T} f_t(x_i).
]

A key feature of LightGBM is its use of a histogram-based representation for feature values when determining candidate split points. Instead of considering every individual feature value as a separate split candidate, values are grouped into discrete bins.

LightGBM also introduced two important techniques: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).

GOSS retains observations associated with larger gradients and samples a smaller proportion of observations with smaller gradients. The motivation is that observations with larger gradients contain more information about the current prediction error and therefore contribute more strongly to estimating useful splits [5].

EFB reduces the effective feature dimension by combining mutually exclusive sparse features into feature bundles. This reduces the number of features that must be evaluated when constructing trees [5].

These mechanisms are intended primarily to improve computational efficiency while maintaining competitive predictive performance.

### 3.5.3 CatBoost

CatBoost is a gradient-boosting algorithm that places particular emphasis on categorical-feature processing and the reduction of prediction shift [6].

A central component of CatBoost is the use of ordered target statistics for categorical variables. Conceptually, for an observation (i), a categorical feature value can be represented using information from preceding observations in a randomly ordered sequence rather than using the target value of the observation itself.

A smoothed target statistic can be represented conceptually as

[
TS_i
====

\frac{
\sum_{j<i} \mathbf{1}(x_j=x_i)y_j
+
aP
}{
\sum_{j<i} \mathbf{1}(x_j=x_i)+a
},
]

where (P) is a prior estimate, (a) controls smoothing, and the ordering prevents the target of observation (i) from being directly used to construct its own categorical representation.

CatBoost also introduces ordered boosting, in which the sequence of observations is used to reduce the prediction shift that can occur when gradient estimates are calculated using information from the same observations used to fit the current model.

The combination of ordered target statistics and ordered boosting forms a central part of CatBoost's approach to reducing target leakage and prediction shift while providing an efficient treatment of categorical variables [6].

### 3.5.4 TabPFN

TabPFN follows a fundamentally different approach from the three tree-based models. Rather than constructing a task-specific ensemble of decision trees, TabPFN is a pretrained transformer-based foundation model designed specifically for tabular prediction [7].

The model is pretrained on large collections of synthetic tabular datasets. During application to a new task, labelled training observations are provided to the pretrained model as context together with the observations for which predictions are required.

The transformer mechanism uses attention to relate observations and features. A simplified scaled dot-product attention operation can be expressed as

[
\operatorname{Attention}(Q,K,V)
===============================

\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V,
]

where (Q), (K), and (V) represent the query, key, and value matrices and (d_k) is the dimensionality used for scaling.

Unlike conventional task-specific supervised learning, the model does not construct a new gradient-boosted tree ensemble for each dataset. Instead, the predictive procedure has been learned during pretraining and is applied to the new tabular task through in-context learning [7].

The original TabPFN work demonstrated strong performance on small- and medium-sized tabular classification tasks and compared the model against established tree-based methods including XGBoost, LightGBM, and CatBoost [7]. The present study uses pretrained TabPFN model weights and evaluates its performance against the same three classes of conventional tree-based baselines across different training-data regimes.

### 3.5.5 Model Comparison

The four models therefore differ in how they obtain predictive information from the available task data.

XGBoost, LightGBM, and CatBoost learn task-specific ensembles by sequentially constructing decision trees. Their differences arise primarily from their optimization, sampling, feature-processing, and categorical-data mechanisms.

TabPFN instead applies a pretrained transformer whose predictive capabilities were learned during prior pretraining. This difference in learning paradigm provides the motivation for the present comparison, particularly when the amount of task-specific labelled data is varied.

The benchmark does not assume that one learning paradigm is universally superior. Instead, it evaluates their predictive and computational behaviour under the same datasets, training-data regimes, and repeated random-seed configurations.


## 2.6 Data Preprocessing

Preprocessing was performed separately for the tree-based models and TabPFN.

For the tree-based models, numerical features were processed using median imputation. Categorical features were processed using most-frequent-value imputation followed by one-hot encoding. Unknown categorical values encountered during transformation were handled by the encoder without producing an error.

For TabPFN, numerical features were processed using median imputation. Categorical features were processed using most-frequent-value imputation followed by ordinal encoding. Unknown categorical values were assigned a dedicated encoded value.

The preprocessing transformations were fitted using the training data and subsequently applied to the corresponding test data, preventing information from the held-out test set from influencing preprocessing.

## 2.7 Evaluation Metrics

Five predictive performance metrics were recorded.

### Accuracy

Accuracy measures the proportion of test observations classified correctly:

[
Accuracy =
\frac{TP + TN}
{TP + TN + FP + FN}.
]

### Precision

Precision measures the proportion of positive predictions that are correct:

[
Precision =
\frac{TP}
{TP + FP}.
]

### Recall

Recall measures the proportion of actual positive observations that are correctly identified:

[
Recall =
\frac{TP}
{TP + FN}.
]

### F1-score

The F1-score is the harmonic mean of precision and recall:

[
F1 =
2\frac{Precision \times Recall}
{Precision + Recall}.
]

### ROC-AUC

ROC-AUC measures the model's ability to discriminate between the two target classes across classification thresholds.

In addition to predictive metrics, training time and prediction time were recorded to assess computational behaviour.

## 2.8 Experimental Procedure

For each dataset and each random seed, a stratified training subset was generated for every supported training size.

For each subset, the four models were fitted using the same training observations and evaluated against the corresponding fixed test set.

The procedure can be summarized as:

1. Load the dataset.
2. Separate predictors and target.
3. Create a stratified 80:20 train-test split.
4. Generate stratified training subsets for each specified sample size.
5. Repeat the subset generation using each of the five random seeds.
6. Preprocess the training and test data using the model-specific preprocessing pipeline.
7. Fit XGBoost, LightGBM, CatBoost, and TabPFN.
8. Measure training time.
9. Generate predictions and class probabilities on the held-out test set.
10. Calculate Accuracy, Precision, Recall, F1-score, ROC-AUC, and prediction time.
11. Store the results for subsequent statistical analysis.

No model was selected based on the test-set results. The test set was used only for final evaluation of each experimental configuration.

## 2.9 Statistical Analysis

The repeated benchmark observations were subsequently analysed using paired comparisons.

TabPFN was compared separately against XGBoost, LightGBM, and CatBoost for each dataset and evaluation metric. Comparisons were matched by dataset, training size, and random seed, allowing differences between models to be evaluated on the same experimental configurations.

The Wilcoxon signed-rank test was used to assess whether paired performance differences were systematically different from zero. Statistical significance was evaluated at the 0.05 level. Because the analysis comprises 45 paired comparisons, Holm-adjusted p-values were also calculated across the full comparison family; both unadjusted and Holm-corrected significance are reported.

In addition to p-values, mean paired differences, confidence intervals, and rank-biserial effect sizes were calculated to describe the magnitude and direction of the observed differences.

This analysis produced 45 TabPFN-versus-tree comparisons across the three datasets and five evaluation metrics. The statistical analysis therefore complements the raw mean performance comparisons by accounting for repeated matched experimental configurations.

## 2.10 Reproducibility

All benchmark results were stored as CSV files containing the model, dataset, training size, random seed, predictive metrics, and computational measurements.

The subsequent analysis was performed independently from model training using the saved benchmark results. This separation allows the statistical analysis, tables, and figures to be regenerated without repeating the computationally expensive model benchmark.

The final experimental outputs include validated raw result files, aggregate performance tables, learning-curve summaries, computational-performance summaries, statistical comparisons, and publication-oriented tables and figures.
