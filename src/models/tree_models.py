"""
Tree models.
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def get_tree_models(seed):

    return {

        "XGBoost": XGBClassifier(

            random_state=seed,

            n_estimators=200,

            eval_metric="logloss"

        ),

        "LightGBM": LGBMClassifier(

            random_state=seed,

            n_estimators=200,

            verbose=-1

        ),

        "CatBoost": CatBoostClassifier(

            random_seed=seed,

            iterations=200,

            verbose=False

        )

    }