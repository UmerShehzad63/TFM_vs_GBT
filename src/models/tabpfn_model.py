
"""
TabPFN model.
"""

import os

from dotenv import load_dotenv
from tabpfn import TabPFNClassifier

load_dotenv()

if not os.getenv("TABPFN_TOKEN"):
    raise RuntimeError(
        "TABPFN_TOKEN is not set. Add it to your .env file or environment."
    )

# Safe to leave even when using GPU
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"


def get_tabpfn():

    return TabPFNClassifier(
        device="cuda"
    )
