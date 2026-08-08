"""
TabPFN model.
"""

import os

from tabpfn import TabPFNClassifier


def get_tabpfn():

    model_path = os.getenv("TABPFN_MODEL_PATH", "auto")

    return TabPFNClassifier(
        device="cuda",
        model_path=model_path,
    )