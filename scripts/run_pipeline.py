"""
NPPES data processing pipeline control script
Execute all data processing steps in sequence
"""

import os
import sys
import subprocess
from datetime import datetime


def run_script(script_name, script_description):
    """
    Run specified script

    Args:
        script_name: Script file name
        script_description: Script description

    Returns:
        bool: Whether successful
    """
    print(f"\n{'=' * 60}")
    print(f"Step: {script_description}")
    print(f"Script: {script_name}")
    print(f"Start time: {datetime.now()}")
    print(f"{'=' * 60}")

    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if not os.path.exists(script_path):
        print(f"❌ Error: Script file does not exist: {script_path}")
        return False

    try:
        # Run script
        result = subprocess.run([sys.executable, script_path],
                                capture_output=True,
                                text=True,
                                encoding='utf-8')

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("Error messages:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {script_description} completed successfully")
            return True
        else:
            print(f"❌ {script_description} execution failed (return code: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False


def check_prerequisites():
    """Check necessary files and directories"""
    print("Checking runtime environment...")

    # Check data directory
    data_dir = r"D:\EMRTS\PROVIDER_LOOKUP\data\nppes\NPPES_Data_Dissemination_June_2025_V2"
    if not os.path.exists(data_dir):
        print(f"❌ Error: NPPES data directory does not exist: {data_dir}")
        return False

    # Check main data files
    required_files = [
        "npidata_pfile_20050523-20250608.csv"
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Error: Missing required data files: {missing_files}")
        return False

    # Check optional files
    optional_files = [
        "pl_pfile_20050523-20250608.csv",
        "endpoint_pfile_20050523-20250608.csv"
    ]

    for file in optional_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"✅ Found optional file: {file}")
        else:
            print(f"⚠️  Optional file does not exist: {file} (related processing will be skipped)")

    # Check taxonomy file
    taxonomy_file = r"D:\EMRTS\PROVIDER_LOOKUP\data\taxonomy\taxonomy.csv"
    if os.path.exists(taxonomy_file):
        print(f"✅ Found taxonomy file: taxonomy.csv")
    else:
        print(f"⚠️  Taxonomy file does not exist: {taxonomy_file} (classification enrichment will be skipped)")

    print("✅ Environment check completed")
    return True


def run_pipeline(start_from=1):
    """
    Run complete data processing pipeline

    Args:
        start_from: Which step to start from (1-4)
    """

    # Define processing steps
    steps = [
        ("1_parse_nppes.py", "Parse NPPES main data"),
        ("2_merge_locations.py", "Merge additional addresses"),
        ("3_merge_endpoints.py", "Merge endpoint information"),
        ("4_enrich_taxonomy.py", "Taxonomy classification enrichment")
    ]

    print("🚀 NPPES data processing pipeline started")
    print(f"Total steps: {len(steps)}")
    print(f"Start time: {datetime.now()}")

    if start_from > 1:
        print(f"⏭️  Starting from step {start_from}")

    # Check environment
    if not check_prerequisites():
        print("\n❌ Environment check failed, cannot continue")
        return False

    pipeline_start_time = datetime.now()

    # Execute steps
    for i, (script_name, description) in enumerate(steps[start_from - 1:], start_from):
        success = run_script(script_name, f"Step {i}: {description}")

        if not success:
            print(f"\n❌ Pipeline failed at step {i}, stopping execution")
            print("You can fix the issue and restart from the failed step using:")
            print(f"python run_pipeline.py --start-from {i}")
            return False

    # Pipeline completed
    total_time = datetime.now() - pipeline_start_time

    print(f"\n🎉 NPPES data processing pipeline completed successfully!")
    print(f"Total time: {total_time}")
    print(f"Completion time: {datetime.now()}")
    print("\n📋 Processing results:")

    # Display output file information
    output_files = [
        ("base_provider_data.json", "Base provider data"),
        ("provider_with_locations.json", "Data with additional addresses"),
        ("provider_with_endpoints.json", "Data with endpoints"),
        ("provider_with_taxonomy.json", "Final complete data")
    ]

    output_dir = r"D:\EMRTS\PROVIDER_LOOKUP\output"

    for filename, description in output_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024 / 1024
            print(f"  ✅ {description}: {filename} ({file_size:.2f} MB)")
        else:
            print(f"  ❌ {description}: {filename} (not generated)")

    return True


def main():
    """Main function"""

    # Parse command line arguments
    start_from = 1
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("NPPES Data Processing Pipeline")
            print("Usage:")
            print("  python run_pipeline.py              # Run complete pipeline")
            print("  python run_pipeline.py --start-from N  # Start from step N")
            print("  python run_pipeline.py --help          # Show help")
            print("\nStep descriptions:")
            print("  1. Parse NPPES main data")
            print("  2. Merge additional addresses")
            print("  3. Merge endpoint information")
            print("  4. Taxonomy classification enrichment")
            return

        elif sys.argv[1] == "--start-from" and len(sys.argv) > 2:
            try:
                start_from = int(sys.argv[2])
                if start_from < 1 or start_from > 4:
                    print("❌ Error: start-from parameter must be between 1-4")
                    return
            except ValueError:
                print("❌ Error: start-from parameter must be a number")
                return

    # Run pipeline
    success = run_pipeline(start_from)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()