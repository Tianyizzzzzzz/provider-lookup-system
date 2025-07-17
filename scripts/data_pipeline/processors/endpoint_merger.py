"""
Endpoint merging script
Merge endpoint information from endpoint_pfile.csv into provider data
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


def process_endpoints_chunk(chunk_df, endpoints_data):
    """Process endpoint data chunk"""
    for npi, group in chunk_df.groupby('NPI'):
        if npi not in endpoints_data:
            endpoints_data[npi] = []

        for _, row in group.iterrows():
            endpoint = {}
            # Extract endpoint information
            endpoint_fields = {
                'Endpoint': 'endpoint_url',
                'Endpoint Type': 'endpoint_type',
                'Endpoint Type Description': 'endpoint_type_description',
                'Endpoint Description': 'endpoint_description',
                'Affiliation': 'affiliation'
            }

            for old_field, new_field in endpoint_fields.items():
                if old_field in row and pd.notna(row[old_field]):
                    endpoint[new_field] = str(row[old_field]).strip()

            if endpoint:  # Only add non-empty endpoints
                endpoints_data[npi].append(endpoint)


def merge_endpoints():
    """
    Merge endpoint data into provider data
    """

    # File path configuration
    data_dir = r"D:\EMRTS\PROVIDER_LOOKUP\data\nppes\NPPES_Data_Dissemination_June_2025_V2"
    output_dir = r"D:\EMRTS\PROVIDER_LOOKUP\output"

    input_file = os.path.join(output_dir, "provider_with_locations.json")
    endpoint_file = os.path.join(data_dir, "endpoint_pfile_20050523-20250608.csv")
    output_file = os.path.join(output_dir, "provider_with_endpoints.json")

    log_progress("Starting endpoint data merging")

    # Check input files
    if not os.path.exists(input_file):
        log_progress(f"Error: Input data file does not exist: {input_file}")
        log_progress("Please run 2_merge_locations.py first")
        return False

    if not os.path.exists(endpoint_file):
        log_progress(f"Warning: Endpoint file does not exist: {endpoint_file}")
        log_progress("Will skip endpoint merging and copy existing data directly")

        # Copy existing data directly
        try:
            provider_data = load_json(input_file)
            # Add empty endpoints field to each record
            for record in provider_data:
                record['endpoints'] = []

            save_json_streaming(provider_data, output_file)
            log_progress("Existing data copied successfully (without endpoints)")
            return True
        except Exception as e:
            log_progress(f"Error copying existing data: {e}")
            return False

    try:
        # 1. Read existing provider data
        provider_data = load_json(input_file)

        # 2. Process endpoint file
        log_progress("Processing endpoint file")

        # Check file size to decide processing method
        file_size_mb = check_file_size(endpoint_file)
        log_progress(f"Endpoint file size: {file_size_mb:.2f} MB")

        endpoints_data = {}
        chunk_size = 10000

        if file_size_mb > 500:  # Use chunk processing for large files
            log_progress("Using chunk processing for endpoint file")
            chunks = read_csv_chunked(
                endpoint_file,
                chunk_size=chunk_size,
                dtype={'NPI': str}
            )

            for chunk_num, chunk_df in track_chunk_processing(chunks, chunk_size, "Processing endpoints"):
                process_endpoints_chunk(chunk_df, endpoints_data)
        else:
            log_progress("Reading endpoint file directly")
            endpoint_df = read_csv_full(endpoint_file, dtype={'NPI': str})
            log_progress(f"Endpoint file records: {len(endpoint_df):,}")
            process_endpoints_chunk(endpoint_df, endpoints_data)

        log_progress(f"Endpoint processing completed, involving {len(endpoints_data):,} NPIs")

        # 3. Merge data
        log_progress("Merging endpoints into provider data")

        tracker = ProgressTracker(len(provider_data), 100000, "Merging endpoints")

        merged_count = 0
        for record in provider_data:
            npi = record.get('npi')
            if npi and npi in endpoints_data:
                record['endpoints'] = endpoints_data[npi]
                merged_count += 1
            else:
                record['endpoints'] = []

            tracker.update()

        tracker.finish()

        log_progress(f"Merging completed:")
        log_progress(f"  Records with endpoints: {merged_count:,}")
        log_progress(f"  Total records: {len(provider_data):,}")
        log_progress(f"  Endpoint coverage rate: {(merged_count / len(provider_data) * 100):.2f}%")

        # 4. Save merged data
        save_json_streaming(provider_data, output_file)

        # Display sample record
        sample_keys = ['npi', 'provider_name', 'primary_taxonomy_code', 'additional_locations', 'endpoints']
        print_sample_record(provider_data, sample_keys)

        log_progress("Endpoint merging completed")
        return True

    except Exception as e:
        log_progress(f"Error during processing: {e}")
        return False


if __name__ == "__main__":
    success = merge_endpoints()
    if success:
        print("\n✅ Endpoint merging completed successfully!")
        print("Output file: D:\\EMRTS\\PROVIDER_LOOKUP\\output\\provider_with_endpoints.json")
        print("Next step: Run 4_enrich_taxonomy.py")
    else:
        print("\n❌ Endpoint merging failed")
        sys.exit(1)