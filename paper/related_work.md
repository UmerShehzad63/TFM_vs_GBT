# 3. Related Work

## 3.1 Gradient-Boosted Decision Trees for Tabular Learning

Gradient boosting is an ensemble learning approach in which predictive models are constructed sequentially, with later models focusing on errors made by previous models. Gradient-boosted decision trees (GBDTs) became an important approach for structured and tabular prediction because decision trees can naturally represent nonlinear relationships and interactions between heterogeneous features.

XGBoost is one of the most widely adopted implementations of gradient-boosted decision trees. Chen and Guestrin introduced XGBoost as a scalable tree-boosting system with algorithmic and systems-level optimizations intended to improve computational efficiency and scalability. The method includes a sparsity-aware learning procedure and approximate tree-learning techniques, together with implementation-level optimizations for memory and computation.

LightGBM was subsequently proposed as a highly efficient gradient-boosting decision-tree implementation. Ke et al. introduced Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to reduce the computational cost associated with evaluating tree splits. These techniques were designed to improve training efficiency while maintaining competitive predictive performance.

CatBoost represents another major development in gradient boosting, with particular attention to categorical variables. Prokhorenkova et al. introduced ordered boosting and a specialized approach to categorical-feature processing. These techniques were designed to reduce prediction shift associated with target leakage in conventional boosting approaches.

Together, XGBoost, LightGBM, and CatBoost provide strong and widely established baselines for tabular classification. Their continued relevance is particularly important when evaluating newer tabular-learning approaches because improvements over these methods represent a substantially stronger comparison than improvements over weaker conventional baselines.

## 3.2 Deep Learning for Tabular Data

Despite the success of deep learning in domains such as natural language processing and computer vision, tabular data have historically presented a more difficult setting for neural-network approaches. Tabular datasets frequently combine numerical and categorical variables, contain heterogeneous feature semantics, and often provide relatively limited numbers of labelled observations.

As a result, gradient-boosted decision trees have remained particularly strong competitors for tabular prediction. The emergence of transformer-based approaches has nevertheless motivated renewed research into whether models pretrained across many tabular tasks can provide more effective general-purpose learning capabilities.

This line of research differs from conventional task-specific supervised learning. Rather than learning an independent predictive model from scratch for every dataset, a pretrained model can transfer information acquired during a prior training phase to a previously unseen tabular task.

## 3.3 Tabular Foundation Models

Foundation-model approaches for tabular data extend the idea of pretraining and transfer to structured datasets. The goal is to learn general predictive capabilities that can subsequently be applied to new datasets with limited task-specific adaptation.

TabPFN is a prominent example of this approach. Hollmann et al. introduced TabPFN as a transformer-based tabular foundation model trained on large collections of synthetic tabular prediction tasks. The model uses in-context learning: labelled training observations from a new dataset are provided as context, allowing the pretrained model to produce predictions for unseen observations.

The approach differs fundamentally from conventional gradient-boosted tree learning. XGBoost, LightGBM, and CatBoost construct a task-specific ensemble from the available training observations. TabPFN instead relies on a model whose predictive algorithm has been learned during pretraining and applies that learned capability to a new dataset.

The original TabPFN study focused particularly on small- and medium-sized tabular datasets. Its benchmark evaluated datasets with up to 10,000 samples and 500 features and included established tree-based baselines such as XGBoost, CatBoost, and LightGBM. The study reported strong performance for TabPFN in this setting.

This result provides the primary motivation for investigating the behaviour of TabPFN under different amounts of available training data.

## 3.4 Existing Empirical Comparisons

The original TabPFN evaluation provides extensive evidence that a tabular foundation model can be highly competitive with established machine-learning methods. Its benchmark used multiple datasets and repeated experimental runs and compared TabPFN with a range of conventional baselines, including gradient-boosted tree models.

However, the existence of strong aggregate benchmark performance does not by itself establish that TabPFN is universally preferable to gradient-boosted trees. The relative performance of a model can depend on the amount of labelled data available, the characteristics of the dataset, and the evaluation metric.

This distinction is particularly important when considering practical model selection. A method that performs well when only a small number of labelled observations are available may have a different relative advantage when substantially more training data are accessible. Conversely, a conventional tree-based method may remain competitive or preferable for some datasets even when a foundation model performs strongly on average.

More recent work has continued to expand the scope of tabular foundation models. For example, subsequent TabPFN research has investigated continued pretraining and larger-scale variants, demonstrating that the field is evolving beyond the original small-data setting. These developments reinforce the importance of specifying the exact model version and experimental setting when comparing tabular foundation models with conventional algorithms.

## 3.5 Research Gap

The existing literature establishes two important observations. First, gradient-boosted tree methods remain strong general-purpose approaches for tabular prediction. Second, TabPFN demonstrates that a pretrained foundation-model approach can achieve highly competitive performance on small and medium-sized tabular datasets.

The remaining practical question addressed by this study is not simply whether TabPFN can outperform tree-based models on a benchmark. Instead, this study examines how the relative performance of the approaches changes as the amount of labelled training data is systematically varied.

To investigate this question, the present study evaluates TabPFN alongside XGBoost, LightGBM, and CatBoost using multiple training-data regimes, repeated random seeds, and a common held-out test set for each dataset. The analysis considers several complementary predictive metrics rather than relying on a single measure of accuracy.

The study therefore provides a training-data-regime perspective on the comparison between tabular foundation models and established gradient-boosted tree algorithms. Rather than assuming that either approach is universally superior, the objective is to identify the conditions under which their relative performance differs and to characterize the resulting computational trade-offs.
