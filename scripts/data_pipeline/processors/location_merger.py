"""
Additional address merging script
Merge additional address information from pl_pfile.csv into base data
"""

import pandas as pd
import os
import sys

# Add utils module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils import (
    read_csv_chunked, read_csv_full, check_file_size, log_progress,
    load_json, save_json_streaming, print_sample_record,
    track_chunk_processing, ProgressTracker
)


def process_locations_chunk(chunk_df, additional_locations):
    """Process additional address data chunk"""
    for npi, group in chunk_df.groupby('NPI'):
        if npi not in additional_locations:
            additional_locations[npi] = []

        for _, row in group.iterrows():
            location = {}
            # Extract address information
            address_fields = {
                'Provider Secondary Practice Location Address- Address Line 1': 'address_line1',
                'Provider Secondary Practice Location Address- Address Line 2': 'address_line2',
                'Provider Secondary Practice Location Address - City Name': 'city',
                'Provider Secondary Practice Location Address - State Name': 'state',
                'Provider Secondary Practice Location Address - Postal Code': 'postal_code',
                'Provider Secondary Practice Location Address - Country Code (If outside US)': 'country',
                'Provider Secondary Practice Location Address - Telephone Number': 'phone'
            }

            for old_field, new_field in address_fields.items():
                if old_field in row and pd.notna(row[old_field]):
                    location[new_field] = str(row[old_field]).strip()

            if location:  # Only add non-empty addresses
                additional_locations[npi].append(location)


def merge_locations():
    """
    Merge additional address data into base provider data
    """

    # File path configuration
    data_dir = r"D:\EMRTS\PROVIDER_LOOKUP\data\nppes\NPPES_Data_Dissemination_June_2025_V2"
    output_dir = r"D:\EMRTS\PROVIDER_LOOKUP\output"

    base_file = os.path.join(output_dir, "base_provider_data.json")
    pl_file = os.path.join(data_dir, "pl_pfile_20050523-20250608.csv")
    output_file = os.path.join(output_dir, "provider_with_locations.json")

    log_progress("Starting additional address data merging")

    # Check input files
    if not os.path.exists(base_file):
        log_progress(f"Error: Base data file does not exist: {base_file}")
        log_progress("Please run 1_parse_nppes.py first")
        return False

    if not os.path.exists(pl_file):
        log_progress(f"Warning: Additional address file does not exist: {pl_file}")
        log_progress("Will skip additional address merging and copy base data directly")

        # Copy base data directly
        try:
            provider_data = load_json(base_file)
            # Add empty additional_locations field to each record
            for record in provider_data:
                record['additional_locations'] = []

            save_json_streaming(provider_data, output_file)
            log_progress("Base data copied successfully (without additional addresses)")
            return True
        except Exception as e:
            log_progress(f"Error copying base data: {e}")
            return False

    try:
        # 1. Read base provider data
        provider_data = load_json(base_file)

        # 2. Process additional address file
        log_progress("Processing additional address file")

        # Check file size to decide processing method
        file_size_mb = check_file_size(pl_file)
        log_progress(f"Additional address file size: {file_size_mb:.2f} MB")

        additional_locations = {}
        chunk_size = 10000

        if file_size_mb > 500:  # Use chunk processing for large files
            log_progress("Using chunk processing for additional address file")
            chunks = read_csv_chunked(
                pl_file,
                chunk_size=chunk_size,
                dtype={'NPI': str}
            )

            for chunk_num, chunk_df in track_chunk_processing(chunks, chunk_size, "Processing additional addresses"):
                process_locations_chunk(chunk_df, additional_locations)
        else:
            log_progress("Reading additional address file directly")
            pl_df = read_csv_full(pl_file, dtype={'NPI': str})
            log_progress(f"Additional address file records: {len(pl_df):,}")
            process_locations_chunk(pl_df, additional_locations)

        log_progress(f"Additional address processing completed, involving {len(additional_locations):,} NPIs")

        # 3. Merge data
        log_progress("Merging additional addresses into provider data")

        tracker = ProgressTracker(len(provider_data), 100000, "Merging additional addresses")

        merged_count = 0
        for record in provider_data:
            npi = record.get('npi')
            if npi and npi in additional_locations:
                record['additional_locations'] = additional_locations[npi]
                merged_count += 1
            else:
                record['additional_locations'] = []

            tracker.update()

        tracker.finish()

        log_progress(f"Merging completed:")
        log_progress(f"  Records with additional addresses: {merged_count:,}")
        log_progress(f"  Total records: {len(provider_data):,}")
        log_progress(f"  Additional address coverage rate: {(merged_count / len(provider_data) * 100):.2f}%")

        # 4. Save merged data
        save_json_streaming(provider_data, output_file)

        # Display sample record
        sample_keys = ['npi', 'provider_name', 'primary_taxonomy_code', 'additional_locations']
        print_sample_record(provider_data, sample_keys)

        log_progress("Additional address merging completed")
        return True

    except Exception as e:
        log_progress(f"Error during processing: {e}")
        return False


if __name__ == "__main__":
    success = merge_locations()
    if success:
        print("\n✅ Additional address merging completed successfully!")
        print("Output file: D:\\EMRTS\\PROVIDER_LOOKUP\\output\\provider_with_locations.json")
        print("Next step: Run 3_merge_endpoints.py")
    else:
        print("\n❌ Additional address merging failed")
        sys.exit(1)