"""
数据加载工具模块
提供通用的数据读取和处理功能
"""

import pandas as pd
import json
import os
import warnings
from datetime import datetime

# 抑制pandas的DtypeWarning警告
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


def read_csv_chunked(filepath, chunk_size=10000, dtype=None, usecols=None):
    """
    分块读取CSV文件

    Args:
        filepath: CSV文件路径
        chunk_size: 每块大小
        dtype: 数据类型字典
        usecols: 要读取的列

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
    完整读取CSV文件

    Args:
        filepath: CSV文件路径
        dtype: 数据类型字典
        usecols: 要读取的列

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
    检查文件大小（MB）

    Args:
        filepath: 文件路径

    Returns:
        float: 文件大小（MB）
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    return os.path.getsize(filepath) / 1024 / 1024


def get_available_columns(filepath, required_columns):
    """
    检查CSV文件中可用的列

    Args:
        filepath: CSV文件路径
        required_columns: 需要的列字典 {原列名: 新列名}

    Returns:
        dict: 可用的列字典
    """
    # 读取文件头部来确定列名
    header_df = pd.read_csv(filepath, nrows=0)
    print(f"文件包含 {len(header_df.columns)} 列")

    available_columns = {}
    missing_columns = []

    for old_col, new_col in required_columns.items():
        if old_col in header_df.columns:
            available_columns[old_col] = new_col
        else:
            missing_columns.append(old_col)

    if missing_columns:
        print(f"警告: 以下列不存在于文件中: {missing_columns}")

    print(f"将读取以下列: {list(available_columns.keys())}")
    return available_columns


def log_progress(message, with_timestamp=True):
    """
    打印带时间戳的日志消息

    Args:
        message: 日志消息
        with_timestamp: 是否包含时间戳
    """
    if with_timestamp:
        print(f"{message} - {datetime.now()}")
    else:
        print(message)


def ensure_directory(filepath):
    """
    确保目录存在

    Args:
        filepath: 文件路径
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)