"""
Taxonomy enrichment script
Add classification and specialization information to provider data based on taxonomy codes
"""

import pandas as pd
import os
import sys

# Add utils module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils import (
    read_csv_full, log_progress, load_json, save_json_streaming,
    print_sample_record, ProgressTracker
)


def enrich_taxonomy():
    """
    Enrich provider information using taxonomy data
    """

    # File path configuration
    taxonomy_file = r"D:\EMRTS\PROVIDER_LOOKUP\data\taxonomy\taxonomy.csv"
    input_file = r"D:\EMRTS\PROVIDER_LOOKUP\output\provider_with_endpoints.json"
    output_file = r"D:\EMRTS\PROVIDER_LOOKUP\output\provider_with_taxonomy.json"

    log_progress("Starting taxonomy classification enrichment")

    # Check input files
    if not os.path.exists(input_file):
        log_progress(f"Error: Input data file does not exist: {input_file}")
        log_progress("Please run 3_merge_endpoints.py first")
        return False

    if not os.path.exists(taxonomy_file):
        log_progress(f"Error: Taxonomy file does not exist: {taxonomy_file}")
        return False

    try:
        # 1. Read taxonomy data
        log_progress("Reading taxonomy data")
        taxonomy_df = read_csv_full(taxonomy_file, dtype={'Code': str})
        log_progress(f"Taxonomy data loaded successfully, total {len(taxonomy_df):,} records")

        # Display taxonomy file column names
        log_progress(f"Taxonomy file column names: {list(taxonomy_df.columns)}")

        # Check if key columns exist
        if 'Code' not in taxonomy_df.columns:
            log_progress("Error: Taxonomy file missing 'Code' column")
            return False

        # Intelligently detect classification and specialization columns
        classification_col = None
        specialization_col = None

        for col in taxonomy_df.columns:
            col_lower = col.lower()
            if 'classification' in col_lower and not classification_col:
                classification_col = col
            elif ('specialization' in col_lower or 'specialty' in col_lower) and not specialization_col:
                specialization_col = col

        # Use default assumptions if not found
        if not classification_col and len(taxonomy_df.columns) >= 2:
            classification_col = taxonomy_df.columns[1]
            log_progress(f"Using '{classification_col}' as classification column")

        if not specialization_col and len(taxonomy_df.columns) >= 3:
            specialization_col = taxonomy_df.columns[2]
            log_progress(f"Using '{specialization_col}' as specialization column")

        # Create taxonomy dictionary for fast lookup
        log_progress("Creating taxonomy lookup dictionary")
        taxonomy_dict = {}

        for _, row in taxonomy_df.iterrows():
            code = str(row['Code']).strip() if pd.notna(row['Code']) else ''
            if code:
                taxonomy_dict[code] = {
                    'classification': str(row[classification_col]).strip() if classification_col and pd.notna(
                        row[classification_col]) else '',
                    'specialization': str(row[specialization_col]).strip() if specialization_col and pd.notna(
                        row[specialization_col]) else ''
                }

        log_progress(f"Taxonomy dictionary created successfully, total {len(taxonomy_dict):,} valid codes")

        # 2. Read provider data
        provider_data = load_json(input_file)

        # 3. Merge taxonomy information
        log_progress("Merging taxonomy information")

        tracker = ProgressTracker(len(provider_data), 100000, "Classification enrichment")

        matched_count = 0
        unmatched_count = 0

        for record in provider_data:
            # Get primary_taxonomy_code, safely handle various data types
            taxonomy_code_raw = record.get('primary_taxonomy_code')

            if taxonomy_code_raw is None or taxonomy_code_raw == '' or (
                    isinstance(taxonomy_code_raw, float) and pd.isna(taxonomy_code_raw)):
                taxonomy_code = ''
            else:
                taxonomy_code = str(taxonomy_code_raw).strip()

            if taxonomy_code and taxonomy_code in taxonomy_dict:
                # Found matching taxonomy information
                taxonomy_info = taxonomy_dict[taxonomy_code]
                record['classification'] = taxonomy_info['classification']
                record['specialization'] = taxonomy_info['specialization']
                matched_count += 1
            else:
                # No matching taxonomy information found
                record['classification'] = ''
                record['specialization'] = ''
                unmatched_count += 1

            tracker.update()

        tracker.finish()

        log_progress(f"Classification enrichment completed:")
        log_progress(f"  Successfully matched: {matched_count:,} records")
        log_progress(f"  Unmatched: {unmatched_count:,} records")
        log_progress(f"  Match rate: {(matched_count / len(provider_data) * 100):.2f}%")

        # 4. Save final data
        save_json_streaming(provider_data, output_file)

        # Display sample record
        sample_keys = ['npi', 'provider_name', 'primary_taxonomy_code', 'classification', 'specialization']
        print_sample_record(provider_data, sample_keys)

        log_progress("Taxonomy classification enrichment completed")
        return True

    except Exception as e:
        log_progress(f"Error during processing: {e}")
        return False


if __name__ == "__main__":
    success = enrich_taxonomy()
    if success:
        print("\n✅ Taxonomy classification enrichment completed successfully!")
        print("Output file: D:\\EMRTS\\PROVIDER_LOOKUP\\output\\provider_with_taxonomy.json")
        print("🎉 All data processing steps completed!")
    else:
        print("\n❌ Taxonomy classification enrichment failed")
        sys.exit(1)