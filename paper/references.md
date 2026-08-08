# References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature, 637*, 319–326. https://doi.org/10.1038/s41586-024-08328-6

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*, 3146–3154.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems, 31*.

Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

OpenML. (n.d.). Credit-g [Dataset]. OpenML dataset ID 31. https://www.openml.org/d/31

Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin, 1*(6), 80–83.

---

## Software and Data Sources

The experiments used publicly available datasets accessed through OpenML and the corresponding Python machine-learning ecosystem.

The Adult dataset is maintained by the UCI Machine Learning Repository and represents a binary classification task predicting whether an individual's annual income exceeds $50,000.

The Bank Marketing dataset is maintained by the UCI Machine Learning Repository and represents a binary classification task predicting whether a client subscribes to a term deposit.

The Credit-G dataset was accessed through OpenML using dataset ID 31.

The benchmark models were implemented using established machine-learning libraries for XGBoost, LightGBM, CatBoost, and TabPFN. Exact software versions and experimental configuration should be reported in the methodology or reproducibility section where available.