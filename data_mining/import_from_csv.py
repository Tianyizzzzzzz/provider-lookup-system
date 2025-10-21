"""
Import directly from NPPES CSV file (most reliable method)
Uses chunked reading for memory efficiency
"""

import os
import sys
import django
import pandas as pd
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, 'provider_lookup_web')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')
django.setup()

from django.db import transaction
from apps.providers.models import Provider

print("="*70)
print("IMPORT FROM NPPES CSV - RELIABLE METHOD")
print("="*70)

# NPPES CSV column mapping
COLUMN_MAPPING = {
    'NPI': 'npi',
    'Entity Type Code': 'entity_type',
    'Provider First Name': 'first_name',
    'Provider Last Name (Legal Name)': 'last_name',
    'Provider Middle Name': 'middle_name',
    'Provider Organization Name (Legal Business Name)': 'organization_name',
    'Provider Name Prefix Text': 'name_prefix',
    'Provider Name Suffix Text': 'name_suffix',
    'Provider Credential Text': 'credential',
    'Provider Business Mailing Address Telephone Number': 'phone',
    'Provider Business Mailing Address Fax Number': 'fax',
    'Provider Enumeration Date': 'enumeration_date',
    'Last Update Date': 'last_update_date',
    'NPI Deactivation Date': 'deactivation_date',
    'NPI Reactivation Date': 'reactivation_date'
}

def clean_value(value):
    """Clean value: handle NaN, empty strings, placeholders"""
    if pd.isna(value):
        return None
    
    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.upper() in ['UNKNOWN', 'N/A', 'NA', 'MISSING', 'NONE']:
            return None
    
    return value

def parse_date(date_value):
    """Parse date from various formats"""
    if pd.isna(date_value) or date_value == '':
        return None
    
    try:
        # NPPES uses MM/DD/YYYY format
        if isinstance(date_value, str):
            return pd.to_datetime(date_value, format='%m/%d/%Y').date()
        else:
            return pd.to_datetime(date_value).date()
    except:
        return None

def import_from_csv(csv_file, limit=None, chunk_size=10000):
    """
    Import from NPPES CSV using chunked reading
    
    Args:
        csv_file: Path to CSV file
        limit: Optional limit for testing
        chunk_size: Number of rows per chunk
    """
    print(f"\n📥 Importing from CSV: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return False
    
    file_size_gb = os.path.getsize(csv_file) / 1e9
    print(f"   File size: {file_size_gb:.1f} GB")
    
    if limit:
        print(f"   TEST MODE: {limit:,} records only")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        return False
    
    # Count total lines for progress bar
    print("\n📊 Counting records...")
    total_lines = 0
    with open(csv_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    total_records = total_lines - 1  # Minus header
    
    if limit:
        total_records = min(total_records, limit)
    
    print(f"   Total records to process: {total_records:,}")
    
    # Read CSV in chunks
    print(f"\n🔄 Processing in chunks of {chunk_size:,}...")
    
    updated_count = 0
    created_count = 0
    error_count = 0
    processed = 0
    
    # Only read columns we need
    columns_to_read = list(COLUMN_MAPPING.keys())
    
    try:
        # Read CSV in chunks
        chunk_iterator = pd.read_csv(
            csv_file,
            usecols=columns_to_read,
            chunksize=chunk_size,
            encoding='utf-8',
            low_memory=False,
            dtype=str  # Read everything as string first
        )
        
        with tqdm(total=total_records, desc="Importing") as pbar:
            for chunk_num, chunk in enumerate(chunk_iterator):
                if limit and processed >= limit:
                    break
                
                # Process chunk
                providers_batch = []
                
                for idx, row in chunk.iterrows():
                    if limit and processed >= limit:
                        break
                    
                    try:
                        npi = clean_value(row.get('NPI'))
                        if not npi:
                            continue
                        
                        # Map entity type code
                        entity_code = clean_value(row.get('Entity Type Code'))
                        entity_type = 'Individual' if entity_code == '1' else 'Organization'
                        
                        # Handle NULL values with empty strings for NOT NULL fields
                        provider_data = {
                            'entity_type': entity_type,
                            'first_name': clean_value(row.get('Provider First Name')) or '',
                            'last_name': clean_value(row.get('Provider Last Name (Legal Name)')) or '',
                            'middle_name': clean_value(row.get('Provider Middle Name')) or '',
                            'organization_name': clean_value(row.get('Provider Organization Name (Legal Business Name)')) or '',
                            'name_prefix': clean_value(row.get('Provider Name Prefix Text')) or '',
                            'name_suffix': clean_value(row.get('Provider Name Suffix Text')) or '',
                            'credential': clean_value(row.get('Provider Credential Text')) or '',
                            'phone': clean_value(row.get('Provider Business Mailing Address Telephone Number')) or '',
                            'fax': clean_value(row.get('Provider Business Mailing Address Fax Number')) or '',
                            'enumeration_date': parse_date(row.get('Provider Enumeration Date')),
                            'last_update_date': parse_date(row.get('Last Update Date')),
                            'deactivation_date': parse_date(row.get('NPI Deactivation Date')),
                            'reactivation_date': parse_date(row.get('NPI Reactivation Date'))
                        }
                        
                        # Use update_or_create
                        provider, created = Provider.objects.update_or_create(
                            npi=npi,
                            defaults=provider_data
                        )
                        
                        if created:
                            created_count += 1
                        else:
                            updated_count += 1
                        
                        processed += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:
                            print(f"\n⚠️  Error processing row: {str(e)[:100]}")
                        continue
                
                # Commit after each chunk
                transaction.commit()
    
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✅ Import complete!")
    print(f"   Updated: {updated_count:,} records")
    print(f"   Created: {created_count:,} records")
    print(f"   Errors: {error_count:,}")
    print(f"   Total processed: {processed:,}")
    
    return True

def verify_import():
    """Verify import quality"""
    print("\n🔍 VERIFICATION")
    print("="*70)
    
    total = Provider.objects.count()
    print(f"Total providers: {total:,}")
    
    # Field completeness
    fields = {
        'first_name': Provider.objects.exclude(first_name__isnull=True).exclude(first_name='').count(),
        'middle_name': Provider.objects.exclude(middle_name__isnull=True).exclude(middle_name='').count(),
        'credential': Provider.objects.exclude(credential__isnull=True).exclude(credential='').count(),
        'phone': Provider.objects.exclude(phone__isnull=True).exclude(phone='').count(),
        'enumeration_date': Provider.objects.exclude(enumeration_date__isnull=True).count(),
    }
    
    print("\nField Completeness:")
    for field, count in fields.items():
        pct = (count / total * 100) if total > 0 else 0
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {field:<20}: {count:>10,} ({pct:>5.1f}%)")
    
    # Check variance
    print("\n📊 Checking data variance (sample 1000)...")
    sample = list(Provider.objects.all()[:1000].values(
        'middle_name', 'credential', 'enumeration_date'
    ))
    
    if sample:
        df = pd.DataFrame(sample)
        
        middle_unique = df['middle_name'].notna().sum()
        cred_unique = df['credential'].notna().sum()
        date_unique = df['enumeration_date'].nunique()
        
        print(f"  Middle names with data: {middle_unique}/1000")
        print(f"  Credentials with data: {cred_unique}/1000")
        print(f"  Unique enumeration dates: {date_unique}")
        
        if date_unique > 100:
            print("  ✅ Good date variance!")
        else:
            print("  ⚠️  Low date variance")
    
    # Entity distribution
    print("\n📊 Entity Type Distribution:")
    from django.db.models import Count
    entity_dist = Provider.objects.values('entity_type').annotate(count=Count('entity_type'))
    for item in entity_dist:
        pct = (item['count'] / total * 100) if total > 0 else 0
        print(f"  {item['entity_type']:<20}: {item['count']:>10,} ({pct:>5.1f}%)")
    
    print("\n" + "="*70)
    
    # Success check
    success = (
        fields['enumeration_date'] > total * 0.9 and  # 90%+ have dates
        fields['middle_name'] > 0 and  # Some have middle names
        fields['credential'] > 0  # Some have credentials
    )
    
    if success:
        print("✅ IMPORT SUCCESSFUL!")
        print("🚀 Data quality is good - proceed with analysis!")
    else:
        print("⚠️  Data quality concerns detected")
    
    return success

def main():
    """Main workflow"""
    
    csv_file = 'data/raw/nppes/NPPES_Data_Dissemination_June_2025_V2/npidata_pfile_20050523-20250608.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("\nPlease verify the path to your NPPES CSV file.")
        return
    
    print(f"\n📁 CSV file found: {csv_file}")
    print(f"   Size: {os.path.getsize(csv_file) / 1e9:.1f} GB")
    
    print("\n💡 Choose import mode:")
    print("  1. Test (50,000 records) - ~2 minutes")
    print("  2. Analysis sample (200,000 records) - ~8 minutes ← RECOMMENDED")
    print("  3. Large sample (500,000 records) - ~20 minutes")
    print("  4. Full import (~9M records) - ~60 minutes")
    
    choice = input("\nSelect (1-4): ")
    
    if choice == '1':
        limit = 50000
        chunk_size = 5000
    elif choice == '2':
        limit = 200000
        chunk_size = 10000
    elif choice == '3':
        limit = 500000
        chunk_size = 10000
    elif choice == '4':
        limit = None
        chunk_size = 10000
    else:
        print("Invalid choice")
        return
    
    # Import
    success = import_from_csv(csv_file, limit=limit, chunk_size=chunk_size)
    
    if success:
        verify_import()

if __name__ == "__main__":
    main()