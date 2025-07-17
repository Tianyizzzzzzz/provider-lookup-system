"""
Data loading utility module
Provides common data reading and processing functionality
"""

import pandas as pd
import json
import os
import warnings
from datetime import datetime

# Suppress pandas DtypeWarning warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


def read_csv_chunked(filepath, chunk_size=10000, dtype=None, usecols=None):
    """
    Read CSV file in chunks

    Args:
        filepath: CSV file path
        chunk_size: Size of each chunk
        dtype: Data type dictionary
        usecols: Columns to read

    Returns:
        Generator of DataFrames
    """
    return pd.read_csv(
        filepath,
        dtype=dtype or {},
        chunksize=chunk_size,
        usecols=usecols,
        low_memory=False
    )


def read_csv_full(filepath, dtype=None, usecols=None):
    """
    Read complete CSV file

    Args:
        filepath: CSV file path
        dtype: Data type dictionary
        usecols: Columns to read

    Returns:
        DataFrame
    """
    return pd.read_csv(
        filepath,
        dtype=dtype or {},
        usecols=usecols,
        low_memory=False
    )


def check_file_size(filepath):
    """
    Check file size (MB)

    Args:
        filepath: File path

    Returns:
        float: File size (MB)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    return os.path.getsize(filepath) / 1024 / 1024


def get_available_columns(filepath, required_columns):
    """
    Check available columns in CSV file

    Args:
        filepath: CSV file path
        required_columns: Required columns dictionary {old_column_name: new_column_name}

    Returns:
        dict: Available columns dictionary
    """
    # Read file header to determine column names
    header_df = pd.read_csv(filepath, nrows=0)
    print(f"File contains {len(header_df.columns)} columns")

    available_columns = {}
    missing_columns = []

    for old_col, new_col in required_columns.items():
        if old_col in header_df.columns:
            available_columns[old_col] = new_col
        else:
            missing_columns.append(old_col)

    if missing_columns:
        print(f"Warning: The following columns do not exist in the file: {missing_columns}")

    print(f"Will read the following columns: {list(available_columns.keys())}")
    return available_columns


def log_progress(message, with_timestamp=True):
    """
    Print log message with timestamp

    Args:
        message: Log message
        with_timestamp: Whether to include timestamp
    """
    if with_timestamp:
        print(f"{message} - {datetime.now()}")
    else:
        print(message)


def ensure_directory(filepath):
    """
    Ensure directory exists

    Args:
        filepath: File path
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)