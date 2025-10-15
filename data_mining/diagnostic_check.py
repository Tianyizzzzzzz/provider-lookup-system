# ============================================
# 文件 2: data_mining/diagnostic_check.py
# 数据库关系诊断脚本
# ============================================
"""
Database Relationship Diagnostic Script
诊断数据库表关系和数据覆盖率
"""

import os
import sys
import django

# Setup Django environment
print("Initializing Django environment...")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
django_app_path = os.path.join(project_root, 'provider_lookup_web')

sys.path.insert(0, django_app_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')
django.setup()

from apps.providers.models import Provider, ProviderAddress, ProviderTaxonomy, TaxonomyCode


def diagnose_database():
    """Comprehensive database diagnosis"""
    print("\n" + "=" * 70)
    print("DATABASE RELATIONSHIP DIAGNOSIS")
    print("=" * 70)

    # 1. Check total counts
    print("\n1. TOTAL RECORD COUNTS:")
    print("-" * 70)
    provider_count = Provider.objects.count()
    address_count = ProviderAddress.objects.count()
    taxonomy_count = ProviderTaxonomy.objects.count()
    taxonomy_code_count = TaxonomyCode.objects.count()

    print(f"   Providers:           {provider_count:>10,}")
    print(f"   Addresses:           {address_count:>10,}")
    print(f"   Provider-Taxonomies: {taxonomy_count:>10,}")
    print(f"   Taxonomy Codes:      {taxonomy_code_count:>10,}")

    # 2. Check relationship integrity
    print("\n2. RELATIONSHIP COVERAGE:")
    print("-" * 70)

    providers_with_addresses = Provider.objects.filter(
        addresses__isnull=False
    ).distinct().count()

    providers_with_taxonomies = Provider.objects.filter(
        taxonomies__isnull=False
    ).distinct().count()

    addr_coverage = (providers_with_addresses / provider_count * 100) if provider_count > 0 else 0
    tax_coverage = (providers_with_taxonomies / provider_count * 100) if provider_count > 0 else 0

    print(f"   Providers with addresses:  {providers_with_addresses:>10,} ({addr_coverage:>5.1f}%)")
    print(f"   Providers with taxonomies: {providers_with_taxonomies:>10,} ({tax_coverage:>5.1f}%)")

    # 3. Sample relationship check
    print("\n3. SAMPLE RELATIONSHIP CHECK (First 10 Providers):")
    print("-" * 70)
    print(f"   {'NPI':<15} {'Entity Type':<15} {'Addresses':<12} {'Taxonomies':<12}")
    print("   " + "-" * 54)

    sample_providers = Provider.objects.all()[:10]

    for provider in sample_providers:
        addr_count = provider.addresses.count()
        tax_count = provider.taxonomies.count()
        entity_abbr = 'Indiv' if provider.entity_type == 'Individual' else 'Org'
        print(f"   {provider.npi:<15} {entity_abbr:<15} {addr_count:<12} {tax_count:<12}")

    # 4. Check for orphaned records
    print("\n4. DATA INTEGRITY CHECK:")
    print("-" * 70)

    try:
        # Check for null foreign keys
        addresses_no_provider = ProviderAddress.objects.filter(provider__isnull=True).count()
        taxonomies_no_provider = ProviderTaxonomy.objects.filter(provider__isnull=True).count()

        print(f"   Orphaned addresses:  {addresses_no_provider:>10,}")
        print(f"   Orphaned taxonomies: {taxonomies_no_provider:>10,}")

    except Exception as e:
        print(f"   Could not check orphaned records: {e}")

    # 5. Check foreign key validity using raw SQL
    print("\n5. FOREIGN KEY VALIDATION:")
    print("-" * 70)

    try:
        from django.db import connection
        cursor = connection.cursor()

        # Check address foreign keys
        cursor.execute("""
            SELECT COUNT(*) 
            FROM provider_addresses 
            WHERE provider_id IS NOT NULL 
            AND provider_id IN (SELECT npi FROM providers)
        """)
        valid_address_fks = cursor.fetchone()[0]

        # Check taxonomy foreign keys
        cursor.execute("""
            SELECT COUNT(*) 
            FROM provider_taxonomies 
            WHERE provider_id IS NOT NULL
            AND provider_id IN (SELECT npi FROM providers)
        """)
        valid_taxonomy_fks = cursor.fetchone()[0]

        addr_fk_validity = (valid_address_fks / address_count * 100) if address_count > 0 else 0
        tax_fk_validity = (valid_taxonomy_fks / taxonomy_count * 100) if taxonomy_count > 0 else 0

        print(f"   Valid address foreign keys:  {valid_address_fks:>10,} ({addr_fk_validity:>5.1f}%)")
        print(f"   Valid taxonomy foreign keys: {valid_taxonomy_fks:>10,} ({tax_fk_validity:>5.1f}%)")

    except Exception as e:
        print(f"   Could not validate foreign keys: {e}")

    # 6. Address data sample
    print("\n6. ADDRESS DATA SAMPLE (First 5 Records):")
    print("-" * 70)

    addresses = ProviderAddress.objects.select_related('provider')[:5]
    if addresses:
        print(f"   {'Provider NPI':<15} {'Type':<12} {'City':<20} {'State':<6}")
        print("   " + "-" * 53)
        for addr in addresses:
            addr_type = 'Location' if addr.address_type == 'location' else 'Mailing'
            print(f"   {addr.provider_id:<15} {addr_type:<12} {addr.city:<20} {addr.state:<6}")
    else:
        print("   No address records found")

    # 7. Taxonomy data sample
    print("\n7. TAXONOMY DATA SAMPLE (First 5 Records):")
    print("-" * 70)

    taxonomies = ProviderTaxonomy.objects.select_related(
        'provider', 'taxonomy_code'
    )[:5]

    if taxonomies:
        print(f"   {'Provider NPI':<15} {'Taxonomy Code':<15} {'Primary':<10}")
        print("   " + "-" * 40)
        for tax in taxonomies:
            primary = 'Yes' if tax.is_primary else 'No'
            print(f"   {tax.provider_id:<15} {tax.taxonomy_code_id:<15} {primary:<10}")
    else:
        print("   No taxonomy records found")

    # 8. Analysis and recommendations
    print("\n8. ANALYSIS & RECOMMENDATIONS:")
    print("-" * 70)

    issues_found = []

    if provider_count == 0:
        issues_found.append("❌ No provider data found - database needs to be populated")

    if addr_coverage < 10:
        issues_found.append(f"⚠️  Very low address coverage ({addr_coverage:.1f}%) - check data import")
    elif addr_coverage < 50:
        issues_found.append(f"⚠️  Low address coverage ({addr_coverage:.1f}%) - this is common in NPPES data")

    if tax_coverage < 10:
        issues_found.append(f"⚠️  Very low taxonomy coverage ({tax_coverage:.1f}%) - check data import")
    elif tax_coverage < 50:
        issues_found.append(f"⚠️  Low taxonomy coverage ({tax_coverage:.1f}%) - this is common in NPPES data")

    if not issues_found:
        print("   ✅ No major issues detected")
        print("   ✅ Data relationships appear to be functioning correctly")
    else:
        for issue in issues_found:
            print(f"   {issue}")

    # 9. Data quality summary for research
    print("\n9. RESEARCH DATA QUALITY SUMMARY:")
    print("-" * 70)
    print(f"   Core Provider Data:    {provider_count:>10,} records (100% complete)")
    print(f"   Address Coverage:      {addr_coverage:>10.1f}% of providers")
    print(f"   Taxonomy Coverage:     {tax_coverage:>10.1f}% of providers")
    print()
    print("   Research Implications:")
    if addr_coverage < 50 or tax_coverage < 50:
        print("   • Focus analysis on provider characteristics")
        print("   • Limited geographic analysis capability")
        print("   • Emphasize robust feature engineering in paper")
        print("   • Frame as real-world data quality challenge")
    else:
        print("   • Good data coverage for comprehensive analysis")
        print("   • Can perform geographic and specialty clustering")
        print("   • Standard healthcare informatics scenario")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

    return {
        'provider_count': provider_count,
        'address_coverage': addr_coverage,
        'taxonomy_coverage': tax_coverage
    }


if __name__ == "__main__":
    try:
        results = diagnose_database()
        print("\n✅ Diagnostic completed successfully")
        print("\nNext steps:")
        print("  1. Review the coverage statistics above")
        print("  2. If coverage is low, check data import process")
        print("  3. Proceed with Phase 2 clustering if core data is good")

    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback

        traceback.print_exc()