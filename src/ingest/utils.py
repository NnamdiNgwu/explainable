
from __future__ import annotations
import argparse
import logging
import os
import sys
import time
import pathlib
import re
import math
from typing import Tuple, Dict
# import dask.dataframe as dd
import pandas as pd
import numpy as np



"""
Utility functions for data ingestion and preprocessing.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text or len(text) == 0:
        return 0.0
    
    # Get probability of each character
    text_len = len(text)
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    for count in char_counts.values():
        probability = count / text_len
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def load_csv(path, parse_dates=("date",), **kwargs):
    """Load CSV with optimized settings for CERT data."""
    logger.info(f"Loading {path}")
    
    default_kwargs = {
        'parse_dates': list(parse_dates),
        'low_memory': False,
        'dtype': {'user': 'str', 'pc': 'str'}  # Keep these as strings
    }
    default_kwargs.update(kwargs)
    
    return pd.read_csv(path, **default_kwargs)


def validate_cert_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required CERT schema columns."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for consistency."""
    # Convert to lowercase and replace spaces/special chars
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
    return df


def memory_usage_report(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print memory usage report for DataFrame."""
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"{name} memory usage: {memory_mb:.2f} MB ({len(df)} rows, {len(df.columns)} columns)")