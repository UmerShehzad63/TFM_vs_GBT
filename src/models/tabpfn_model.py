"""
TabPFN model.
"""

import os
from dotenv import load_dotenv

# Load .env FIRST
load_dotenv()

if not os.getenv("TABPFN_TOKEN"):
    raise RuntimeError(
        "TABPFN_TOKEN is not set. Add it to your .env file."
    )

# Set before importing TabPFN
os.environ["TABPFN_TOKEN"] = os.getenv("TABPFN_TOKEN")
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

from tabpfn import TabPFNClassifier


def get_tabpfn():
    return TabPFNClassifier(
        device="cuda"
    )