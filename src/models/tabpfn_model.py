"""
TabPFN model.
"""

import os

from tabpfn import TabPFNClassifier

# Paste your API key here
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_AQ4o05f1WuW7inHI4-lrWHy_em3RwIl00DYoqvC038c"

# Allow CPU execution
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"


def get_tabpfn():
    return TabPFNClassifier()