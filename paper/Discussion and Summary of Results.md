# 5. Discussion and Summary of Results

## 5.1 Overview of the Findings

The objective of this study was to examine how TabPFN compares with established gradient-boosted tree algorithms across different training-data regimes. The benchmark evaluated XGBoost, LightGBM, CatBoost, and TabPFN on the Adult, Bank Marketing, and Credit-G datasets using multiple training sizes and five random seeds. The results show that no single model consistently dominates across all datasets, evaluation metrics, and training-data regimes.

The clearest overall advantage for TabPFN was observed on Bank Marketing. TabPFN achieved the highest mean Accuracy, Recall, F1-score, and ROC-AUC, while CatBoost achieved the highest Precision. In contrast, CatBoost achieved the strongest aggregate results across most metrics on Adult and Credit-G. These findings indicate that the relative performance of TabPFN is strongly dependent on the dataset and metric rather than being universally superior to conventional tree-based models.

The learning-regime analysis provides further evidence that training-data size alone does not determine which model performs best. On Adult, TabPFN was competitive in the smaller and medium regimes, while CatBoost became stronger at larger training sizes. On Bank Marketing, TabPFN maintained an advantage across the small, medium, and large regimes. On Credit-G, TabPFN remained competitive with the tree-based models but CatBoost remained particularly strong.

The statistical analysis supports the presence of systematic differences between models. Of the 45 TabPFN-versus-tree comparisons, 34 were statistically significant at the unadjusted 0.05 level, while 31 remained significant after Holm correction. However, statistically significant differences occurred in both directions. Therefore, significance alone does not imply that TabPFN was superior; the direction and magnitude of the observed differences also depended on the dataset and baseline model.

The computational results reveal an additional trade-off. TabPFN achieved strong predictive performance in several experimental settings, but it had substantially higher measured training and prediction times than the conventional tree-based models in this implementation. The prediction-time difference was particularly pronounced on Adult and Bank Marketing.

## 5.2 Research Question 1: Does TabPFN Provide an Advantage When Training Data Are Limited?

The results provide partial support for this question.

TabPFN performs strongly in the lower-data regimes, particularly in comparison with XGBoost and LightGBM. On Adult, TabPFN achieves a mean small-regime ROC-AUC of 0.7993, compared with 0.7304 for XGBoost and 0.6419 for LightGBM. However, CatBoost remains slightly stronger at 0.8015.

The advantage is more pronounced on Bank Marketing, where TabPFN achieves a small-regime ROC-AUC of 0.7520, compared with 0.6437 for XGBoost and 0.6195 for LightGBM. Credit-G also shows strong TabPFN performance relative to XGBoost and LightGBM, although CatBoost remains competitive.

These findings support the view that TabPFN can be particularly competitive when labelled training data are limited. However, they do not demonstrate that TabPFN is universally the strongest model in low-data settings because CatBoost remains highly competitive on Adult and Credit-G.

## 5.3 Research Question 2: How Does the Relative Performance of TabPFN Change as Training Size Increases?

The results do not support a single universal pattern.

On Adult, TabPFN performs strongly in smaller and medium training regimes, but CatBoost becomes stronger at larger training sizes. In the large regime, CatBoost achieves a mean ROC-AUC of 0.9182 compared with 0.9116 for TabPFN.

Bank Marketing provides an important contrast. TabPFN maintains an advantage across the small, medium, and large regimes. Its mean ROC-AUC increases from 0.7520 in the small regime to 0.8907 in the medium regime and 0.9339 in the large regime. In the large regime, this remains higher than the corresponding values for XGBoost, LightGBM, and CatBoost.

Credit-G provides an intermediate pattern. TabPFN achieves a mean ROC-AUC of 0.6767 in the small regime and 0.7714 in the medium regime. CatBoost remains very competitive, achieving 0.6922 and 0.7651 respectively.

The results therefore do not support a simple assumption that TabPFN's advantage necessarily disappears as more training data become available. Instead, the effect of training-data size is dataset-dependent.

## 5.4 Research Question 3: Is the TabPFN Advantage Consistent Across Datasets and Evaluation Metrics?

The answer is no.

The aggregate results differ substantially across the three datasets. On Adult, CatBoost achieves the strongest mean Accuracy, Precision, F1-score, and ROC-AUC, while XGBoost achieves the highest Recall. On Bank Marketing, TabPFN achieves the strongest mean Accuracy, Recall, F1-score, and ROC-AUC, while CatBoost achieves the highest Precision. On Credit-G, CatBoost achieves the strongest mean Accuracy, Recall, F1-score, and ROC-AUC, while XGBoost achieves the highest Precision.

The statistical comparisons reinforce this dataset dependence. On Adult, TabPFN significantly outperforms XGBoost and LightGBM on several metrics but significantly underperforms CatBoost on several metrics. On Bank Marketing, TabPFN significantly outperforms the tree-based models on most of the evaluated metrics, including all five comparisons with XGBoost and LightGBM. On Credit-G, TabPFN significantly outperforms XGBoost and LightGBM on several metrics, whereas none of the five TabPFN-CatBoost comparisons is statistically significant after the paired analysis.

These findings demonstrate that TabPFN's relative performance depends on both the dataset and the evaluation metric. Reporting only an overall average would therefore conceal important differences in model behaviour.

## 5.5 Research Question 4: What Computational Trade-offs Exist?

The computational results demonstrate a clear trade-off between predictive performance and execution time.

TabPFN has the highest mean training time across all three datasets in the benchmark. Mean training time is 0.679 seconds on Adult, 0.685 seconds on Bank Marketing, and 0.659 seconds on Credit-G.

The difference is considerably larger for prediction time. On Adult, TabPFN requires approximately 9.036 seconds for prediction compared with less than 0.1 seconds for each of the three tree-based models. On Bank Marketing, TabPFN requires approximately 8.946 seconds compared with approximately 0.04–0.09 seconds for the tree models. On Credit-G, TabPFN requires approximately 0.667 seconds compared with approximately 0.01 seconds for the tree-based models.

Therefore, the predictive advantages observed for TabPFN in several settings are accompanied by substantially higher measured prediction times in this experimental implementation. Applications in which predictive performance is the primary objective may accept this additional cost, whereas applications requiring low inference latency may favour conventional tree-based approaches.

## 5.6 Interpretation of the Model Differences

The observed results indicate that TabPFN and gradient-boosted tree models represent different approaches to tabular prediction.

XGBoost, LightGBM, and CatBoost construct task-specific ensembles using the available labelled observations. Their performance can therefore benefit directly from additional task-specific training data. TabPFN follows a different paradigm by applying a pretrained model to a new tabular task.

The strong performance of TabPFN in several lower- and medium-data configurations is consistent with the potential value of information acquired during pretraining when relatively little task-specific data are available. However, the results also demonstrate that pretraining does not eliminate dataset dependence. CatBoost remains highly competitive on Adult and Credit-G, while TabPFN performs particularly strongly on Bank Marketing.

The differences between datasets suggest that the usefulness of a pretrained tabular model depends on more than sample size alone. Dataset characteristics and the relationship between predictors and target may influence the relative suitability of the approaches. However, the present benchmark does not isolate these factors experimentally, so causal explanations for the observed dataset differences cannot be established from the current results.

## 5.7 Implications for Model Selection

The findings have several implications for practical model selection.

First, TabPFN should not automatically replace established gradient-boosted tree models. CatBoost achieves the strongest aggregate performance on two of the three datasets, demonstrating that conventional approaches remain highly competitive.

Second, TabPFN should not be dismissed simply because gradient-boosted trees are strong baselines. Bank Marketing demonstrates that TabPFN can provide meaningful improvements across several predictive metrics and training-data regimes.

Third, model selection should consider computational requirements together with predictive performance. TabPFN may be attractive when its predictive advantages justify the additional computational cost, while conventional tree-based models may be preferable when low inference latency is an important requirement.

Finally, evaluating models at multiple training sizes provides information that cannot be obtained from a single benchmark configuration. The learning curves demonstrate that model rankings can change as additional training data become available.

## 5.8 Evaluation of the Hypotheses

### H1: TabPFN will be competitive or superior when training data are limited.

**Partially supported.**

TabPFN performs strongly in the smaller training regimes and outperforms XGBoost and LightGBM on several low-data comparisons. However, CatBoost remains competitive or superior on some datasets, particularly Adult and Credit-G. The findings therefore support the competitiveness of TabPFN in low-data settings but do not establish universal superiority.

### H2: TabPFN's relative advantage will change as training size increases.

**Partially supported.**

The Adult results show a change in relative performance, with CatBoost becoming stronger at larger training sizes. However, Bank Marketing does not follow this pattern because TabPFN maintains its advantage across the evaluated training regimes. The appropriate conclusion is that training-data size influences relative model performance, but the direction and magnitude of the effect depend on the dataset.

### H3: TabPFN's performance relative to tree models will vary across datasets and metrics.

**Supported.**

Adult, Bank Marketing, and Credit-G produce substantially different performance patterns, and the strongest model varies across evaluation metrics. The statistical comparisons further demonstrate that the direction of model differences changes according to the dataset and baseline model.

### H4: Predictive advantages will involve computational trade-offs.

**Supported.**

TabPFN provides strong predictive performance in several settings but has substantially higher measured prediction times than the tree-based models in this benchmark. The results therefore demonstrate a clear predictive-performance versus computational-cost trade-off.

## 5.9 Overall Interpretation

The central finding of this study is that there is no universal winner between TabPFN and gradient-boosted tree algorithms.

TabPFN demonstrates strong predictive capability and provides its clearest advantage on Bank Marketing, where it performs strongly across multiple metrics and training-data regimes. At the same time, CatBoost remains highly competitive and achieves the strongest aggregate results on Adult and Credit-G.

The results therefore support a complementary view of tabular foundation models. TabPFN represents a valuable addition to the set of available tabular-learning methods, particularly when its predictive advantages justify its computational requirements. However, the continued strength of gradient-boosted tree algorithms means that they remain important baselines and practical alternatives.

More broadly, the study demonstrates the importance of evaluating tabular models across multiple training-data regimes rather than relying on a single benchmark configuration. Model rankings can change with training size, and the effect is not necessarily consistent across datasets.

Overall, the evidence supports a dataset- and application-dependent approach to model selection. Rather than asking whether TabPFN is universally better than gradient-boosted trees, a more useful practical question is under which conditions the additional predictive capabilities of a tabular foundation model provide sufficient benefit to justify its computational cost.