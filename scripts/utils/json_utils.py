"""
JSON处理工具模块
提供JSON文件的读取、写入和批处理功能
"""

import json
import os
from .data_loader import ensure_directory, log_progress


def load_json(filepath):
    """
    加载JSON文件

    Args:
        filepath: JSON文件路径

    Returns:
        list/dict: JSON数据
    """
    log_progress(f"正在读取JSON文件: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON文件不存在: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            log_progress(f"JSON文件加载完成，共 {len(data):,} 条记录", False)
        else:
            log_progress("JSON文件加载完成", False)

        return data

    except Exception as e:
        raise Exception(f"读取JSON文件时出错: {e}")


def save_json(data, filepath, indent=2):
    """
    保存数据为JSON文件

    Args:
        data: 要保存的数据
        filepath: 输出文件路径
        indent: JSON缩进
    """
    log_progress(f"正在保存到 {filepath}")

    try:
        ensure_directory(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        # 获取文件大小
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024

        log_progress(f"数据保存完成！", False)
        log_progress(f"输出文件: {filepath}", False)
        log_progress(f"文件大小: {file_size_mb:.2f} MB", False)

        if isinstance(data, list):
            log_progress(f"总记录数: {len(data):,}", False)

    except Exception as e:
        raise Exception(f"保存JSON文件时出错: {e}")


def save_json_streaming(data, filepath, batch_size=50000):
    """
    流式保存大型JSON数组
    适用于大数据集，分批写入以节省内存

    Args:
        data: 数据迭代器或列表
        filepath: 输出文件路径
        batch_size: 批处理大小
    """
    log_progress(f"正在流式保存到 {filepath}")

    try:
        ensure_directory(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('[\n')  # 开始JSON数组

            total_records = 0
            is_first = True

            # 如果是列表，按批次处理
            if isinstance(data, list):
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]

                    for j, record in enumerate(batch):
                        if not is_first:
                            f.write(',\n')

                        json_str = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
                        f.write('  ' + json_str)
                        is_first = False
                        total_records += 1

                    if (total_records) % 100000 == 0:
                        log_progress(f"已写入 {total_records:,} 条记录", False)

            f.write('\n]')  # 结束JSON数组

        # 获取文件大小
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024

        log_progress(f"流式保存完成！", False)
        log_progress(f"输出文件: {filepath}", False)
        log_progress(f"文件大小: {file_size_mb:.2f} MB", False)
        log_progress(f"总记录数: {total_records:,}", False)

    except Exception as e:
        raise Exception(f"流式保存JSON文件时出错: {e}")


def get_sample_record(data, index=0):
    """
    获取示例记录

    Args:
        data: JSON数据
        index: 记录索引

    Returns:
        dict: 示例记录
    """
    if isinstance(data, list) and len(data) > index:
        return data[index]
    return None


def print_sample_record(data, keys=None, index=0, max_length=50):
    """
    打印示例记录

    Args:
        data: JSON数据
        keys: 要显示的键列表
        index: 记录索引
        max_length: 值的最大显示长度
    """
    sample = get_sample_record(data, index)
    if not sample:
        log_progress("没有可显示的示例记录", False)
        return

    log_progress("示例记录:", False)

    display_keys = keys if keys else list(sample.keys())[:10]  # 默认显示前10个键

    for key in display_keys:
        if key in sample:
            value = sample[key]
            if len(str(value)) > max_length:
                value = str(value)[:max_length] + "..."
            log_progress(f"  {key}: {value}", False)