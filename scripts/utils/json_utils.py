"""
JSON processing utility module
Provides JSON file reading, writing and batch processing functionality
"""

import json
import os
from .data_loader import ensure_directory, log_progress


def load_json(filepath):
    """
    Load JSON file

    Args:
        filepath: JSON file path

    Returns:
        list/dict: JSON data
    """
    log_progress(f"Reading JSON file: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file does not exist: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            log_progress(f"JSON file loaded successfully, total {len(data):,} records", False)
        else:
            log_progress("JSON file loaded successfully", False)

        return data

    except Exception as e:
        raise Exception(f"Error reading JSON file: {e}")


def save_json(data, filepath, indent=2):
    """
    Save data as JSON file

    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    log_progress(f"Saving to {filepath}")

    try:
        ensure_directory(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        # Get file size
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024

        log_progress(f"Data saved successfully!", False)
        log_progress(f"Output file: {filepath}", False)
        log_progress(f"File size: {file_size_mb:.2f} MB", False)

        if isinstance(data, list):
            log_progress(f"Total records: {len(data):,}", False)

    except Exception as e:
        raise Exception(f"Error saving JSON file: {e}")


def save_json_streaming(data, filepath, batch_size=50000):
    """
    Stream save large JSON arrays
    Suitable for large datasets, write in batches to save memory

    Args:
        data: Data iterator or list
        filepath: Output file path
        batch_size: Batch processing size
    """
    log_progress(f"Streaming save to {filepath}")

    try:
        ensure_directory(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('[\n')  # Start JSON array

            total_records = 0
            is_first = True

            # If it's a list, process in batches
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
                        log_progress(f"Written {total_records:,} records", False)

            f.write('\n]')  # End JSON array

        # Get file size
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024

        log_progress(f"Streaming save completed!", False)
        log_progress(f"Output file: {filepath}", False)
        log_progress(f"File size: {file_size_mb:.2f} MB", False)
        log_progress(f"Total records: {total_records:,}", False)

    except Exception as e:
        raise Exception(f"Error streaming save JSON file: {e}")


def get_sample_record(data, index=0):
    """
    Get sample record

    Args:
        data: JSON data
        index: Record index

    Returns:
        dict: Sample record
    """
    if isinstance(data, list) and len(data) > index:
        return data[index]
    return None


def print_sample_record(data, keys=None, index=0, max_length=50):
    """
    Print sample record

    Args:
        data: JSON data
        keys: List of keys to display
        index: Record index
        max_length: Maximum display length for values
    """
    sample = get_sample_record(data, index)
    if not sample:
        log_progress("No sample record to display", False)
        return

    log_progress("Sample record:", False)

    display_keys = keys if keys else list(sample.keys())[:10]  # Default to first 10 keys

    for key in display_keys:
        if key in sample:
            value = sample[key]
            if len(str(value)) > max_length:
                value = str(value)[:max_length] + "..."
            log_progress(f"  {key}: {value}", False)