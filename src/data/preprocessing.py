"""
Data preprocessing.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_tree_preprocessor(X):

    numeric = X.select_dtypes(include=["number"]).columns

    categorical = X.select_dtypes(
        exclude=["number"]
    ).columns

    return ColumnTransformer(

        [

            (

                "num",

                SimpleImputer(strategy="median"),

                numeric

            ),

            (

                "cat",

                Pipeline([

                    (

                        "imputer",

                        SimpleImputer(

                            strategy="most_frequent"

                        )

                    ),

                    (

                        "encoder",

                        OneHotEncoder(

                            handle_unknown="ignore"

                        )

                    )

                ]),

                categorical

            )

        ],

        force_int_remainder_cols=False

    )


def build_tabpfn_preprocessor(X):

    numeric = X.select_dtypes(include=["number"]).columns

    categorical = X.select_dtypes(
        exclude=["number"]
    ).columns

    return ColumnTransformer(

        [

            (

                "num",

                SimpleImputer(strategy="median"),

                numeric

            ),

            (

                "cat",

                Pipeline([

                    (

                        "imputer",

                        SimpleImputer(

                            strategy="most_frequent"

                        )

                    ),

                    (

                        "encoder",

                        OrdinalEncoder(

                            handle_unknown="use_encoded_value",

                            unknown_value=-1

                        )

                    )

                ]),

                categorical

            )

        ],

        force_int_remainder_cols=False

    )