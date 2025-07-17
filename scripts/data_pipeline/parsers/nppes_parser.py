"""
NPPES main data parsing script
Extract basic provider information from npidata_pfile.csv
"""

import pandas as pd
import os
import sys

# Add utils module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils import (
    read_csv_chunked, get_available_columns, log_progress,
    save_json_streaming, print_sample_record, track_chunk_processing
)


def parse_nppes_main():
    """
    Parse NPPES main data file
    """

    # File path configuration
    data_dir = r"D:\EMRTS\PROVIDER_LOOKUP\data\nppes\NPPES_Data_Dissemination_June_2025_V2"
    output_dir = r"D:\EMRTS\PROVIDER_LOOKUP\output"

    main_file = os.path.join(data_dir, "npidata_pfile_20050523-20250608.csv")
    output_file = os.path.join(output_dir, "base_provider_data.json")

    log_progress("Starting NPPES main data parsing")

    # Define columns to extract
    main_columns = {
        'NPI': 'npi',
        'Entity Type Code': 'entity_type_code',
        'Provider Last Name (Legal Name)': 'provider_last_name',
        'Provider First Name': 'provider_first_name',
        'Provider Middle Name': 'provider_middle_name',
        'Provider Organization Name (Legal Business Name)': 'organization_name',
        'Provider First Line Business Mailing Address': 'business_address_line1',
        'Provider Second Line Business Mailing Address': 'business_address_line2',
        'Provider Business Mailing Address City Name': 'business_city',
        'Provider Business Mailing Address State Name': 'business_state',
        'Provider Business Mailing Address Postal Code': 'business_postal_code',
        'Provider Business Mailing Address Country Code (If outside U.S.)': 'business_country',
        'Healthcare Provider Taxonomy Code_1': 'primary_taxonomy_code'
    }

    try:
        # Check available columns
        available_columns = get_available_columns(main_file, main_columns)

        if not available_columns:
            log_progress("Error: No required columns found")
            return False

        # Read main file in chunks
        chunk_size = 10000
        main_data_list = []

        log_progress("Starting chunk reading of main file")
        chunks = read_csv_chunked(
            main_file,
            chunk_size=chunk_size,
            dtype={'NPI': str},
            usecols=list(available_columns.keys())
        )

        for chunk_num, chunk_df in track_chunk_processing(chunks, chunk_size, "Processing main data"):
            # Rename columns
            chunk_df = chunk_df.rename(columns=available_columns)

            # Combine provider names
            if 'provider_first_name' in chunk_df.columns and 'provider_last_name' in chunk_df.columns:
                chunk_df['provider_name'] = chunk_df.apply(
                    lambda row: ' '.join(filter(None, [
                        str(row.get('provider_first_name', '')).strip() if pd.notna(
                            row.get('provider_first_name')) else '',
                        str(row.get('provider_middle_name', '')).strip() if pd.notna(
                            row.get('provider_middle_name')) else '',
                        str(row.get('provider_last_name', '')).strip() if pd.notna(
                            row.get('provider_last_name')) else ''
                    ])), axis=1
                )

            # Use organization_name as fallback name
            if 'organization_name' in chunk_df.columns:
                chunk_df['provider_name'] = chunk_df.apply(
                    lambda row: row.get('provider_name', '') if row.get('provider_name', '').strip()
                    else str(row.get('organization_name', '')).strip() if pd.notna(
                        row.get('organization_name')) else '',
                    axis=1
                )

            main_data_list.append(chunk_df)

        # Merge all chunks
        log_progress("Merging all data chunks")
        main_data = pd.concat(main_data_list, ignore_index=True)
        log_progress(f"Main data processing completed, total records: {len(main_data):,}")

        # Convert to JSON format
        log_progress("Converting to JSON format")
        result_data = []

        for _, row in main_data.iterrows():
            record = {
                'npi': row['npi'],
                'entity_type_code': row.get('entity_type_code'),
                'provider_name': row.get('provider_name', ''),
                'primary_address': {
                    'line1': row.get('business_address_line1'),
                    'line2': row.get('business_address_line2'),
                    'city': row.get('business_city'),
                    'state': row.get('business_state'),
                    'postal_code': row.get('business_postal_code'),
                    'country': row.get('business_country')
                },
                'primary_taxonomy_code': row.get('primary_taxonomy_code')
            }

            # Clean empty values
            record = {k: v for k, v in record.items() if v is not None and v != ''}
            if 'primary_address' in record:
                record['primary_address'] = {k: v for k, v in record['primary_address'].items()
                                             if v is not None and v != ''}

            result_data.append(record)

        # Save results
        save_json_streaming(result_data, output_file)

        # Display sample record
        sample_keys = ['npi', 'entity_type_code', 'provider_name', 'primary_taxonomy_code']
        print_sample_record(result_data, sample_keys)

        log_progress("Main data parsing completed")
        return True

    except Exception as e:
        log_progress(f"Error during processing: {e}")
        return False


if __name__ == "__main__":
    success = parse_nppes_main()
    if success:
        print("\n✅ Main data parsing completed successfully!")
        print("Output file: D:\\EMRTS\\PROVIDER_LOOKUP\\output\\base_provider_data.json")
        print("Next step: Run 2_merge_locations.py")
    else:
        print("\n❌ Main data parsing failed")
        sys.exit(1)