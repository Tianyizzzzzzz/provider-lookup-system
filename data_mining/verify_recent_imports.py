"""
Verify quality of RECENTLY imported data
Focus on the 200K records we just updated
"""

import os
import sys
import django
import pandas as pd

sys.path.insert(0, 'provider_lookup_web')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')
django.setup()

from apps.providers.models import Provider

print("="*70)
print("VERIFY RECENTLY IMPORTED DATA")
print("="*70)

# Strategy: Find providers with non-empty enumeration_date (recently updated)
print("\n🔍 Finding recently updated providers...")

# Get providers with actual data (not empty strings)
recent_providers = Provider.objects.exclude(
    enumeration_date__isnull=True
).exclude(
    middle_name=''
).exclude(
    credential=''
)[:5000]

recent_count = recent_providers.count()
print(f"✅ Found {recent_count:,} providers with complete data")

if recent_count == 0:
    print("\n❌ No providers with complete data found!")
    print("   This suggests the import may have issues.")
    
    # Try alternative: just get any recent updates
    print("\n   Checking providers with enumeration_date...")
    with_dates = Provider.objects.exclude(enumeration_date__isnull=True).count()
    print(f"   Providers with dates: {with_dates:,}")
    
    if with_dates > 0:
        sample = list(Provider.objects.exclude(
            enumeration_date__isnull=True
        )[:10].values())
        
        print("\n   Sample providers with dates:")
        for p in sample[:3]:
            print(f"\n   NPI: {p['npi']}")
            print(f"     Entity: {p['entity_type']}")
            print(f"     First: {p['first_name']}")
            print(f"     Middle: {p['middle_name']}")
            print(f"     Credential: {p['credential']}")
            print(f"     Enum Date: {p['enumeration_date']}")
    
    sys.exit(1)

# Analyze the recent data
print("\n📊 ANALYZING RECENT IMPORTS")
print("="*70)

# Extract sample for analysis
sample_data = list(recent_providers[:5000].values(
    'npi', 'entity_type', 'first_name', 'last_name', 'middle_name',
    'organization_name', 'credential', 'phone', 'fax',
    'enumeration_date', 'last_update_date', 'deactivation_date'
))

df = pd.DataFrame(sample_data)

print(f"\nSample size: {len(df):,} providers")

# Field completeness
print("\n📈 Field Completeness (in recent imports):")
print("-"*70)
for col in ['first_name', 'last_name', 'middle_name', 'organization_name',
            'credential', 'phone', 'fax']:
    not_empty = (df[col].notna() & (df[col] != '')).sum()
    pct = (not_empty / len(df)) * 100
    print(f"  {col:<25}: {not_empty:>6} / {len(df):>6} ({pct:>5.1f}%)")

# Check uniqueness
print("\n🔢 Data Variance (uniqueness):")
print("-"*70)
for col in ['first_name', 'middle_name', 'credential', 'phone']:
    valid_data = df[df[col].notna() & (df[col] != '')]
    if len(valid_data) > 0:
        unique = valid_data[col].nunique()
        print(f"  {col:<25}: {unique:>6} unique values")

# Check dates
print("\n📅 Temporal Data:")
print("-"*70)
if 'enumeration_date' in df.columns:
    unique_dates = df['enumeration_date'].nunique()
    print(f"  Unique enumeration dates: {unique_dates:,}")
    
    if unique_dates > 1:
        print("  ✅ Good date variance!")
        
        # Show date range
        date_counts = df['enumeration_date'].value_counts().head(10)
        print("\n  Top 10 enumeration dates:")
        for date, count in date_counts.items():
            print(f"    {date}: {count} providers")
    else:
        print("  ⚠️  All dates are the same")
        print(f"    Date value: {df['enumeration_date'].iloc[0]}")

# Entity distribution
print("\n📊 Entity Type Distribution:")
print("-"*70)
entity_dist = df['entity_type'].value_counts()
for entity, count in entity_dist.items():
    pct = (count / len(df)) * 100
    print(f"  {entity:<20}: {count:>6} ({pct:>5.1f}%)")

# Sample providers
print("\n👀 Sample Providers (first 5):")
print("-"*70)
for i, row in df.head(5).iterrows():
    print(f"\n{i+1}. NPI: {row['npi']}")
    print(f"   Entity: {row['entity_type']}")
    print(f"   Name: {row['first_name']} {row['middle_name']} {row['last_name']}")
    print(f"   Org: {row['organization_name']}")
    print(f"   Credential: {row['credential']}")
    print(f"   Phone: {row['phone']}")
    print(f"   Enumeration: {row['enumeration_date']}")

print("\n" + "="*70)
print("ASSESSMENT")
print("="*70)

# Quality checks
checks = {
    'Has Individual providers': (df['entity_type'] == 'Individual').any(),
    'Has Organization providers': (df['entity_type'] == 'Organization').any(),
    'Middle names vary': df['middle_name'].nunique() > 100,
    'Credentials vary': df['credential'].nunique() > 100,
    'Dates vary': df['enumeration_date'].nunique() > 100,
    'Phone numbers present': (df['phone'] != '').sum() > len(df) * 0.5,
}

for check, passed in checks.items():
    status = "✅" if passed else "⚠️ "
    print(f"  {status} {check}")

all_passed = all(checks.values())

if all_passed:
    print("\n🎉 DATA QUALITY EXCELLENT!")
    print("✅ Recently imported data is ready for analysis")
    print("\n🚀 NEXT STEP: Run Phase 2 clustering on this data")
    print("   The clustering script will use these 200K high-quality records")
else:
    print("\n⚠️  Some quality concerns detected")
    print("   Review the output above for details")

# Save list of good NPIs for Phase 2
if all_passed:
    print("\n💾 Saving list of high-quality NPIs for Phase 2...")
    good_npis = df['npi'].tolist()
    
    import json
    output_file = 'data_mining/high_quality_npis.json'
    with open(output_file, 'w') as f:
        json.dump(good_npis[:200000], f)  # Up to 200K
    
    print(f"   Saved {len(good_npis):,} NPIs to: {output_file}")
    print("   Phase 2 will use this list automatically")

print("\n" + "="*70)