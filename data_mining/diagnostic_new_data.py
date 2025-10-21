"""
Diagnostic check specifically for NEWLY IMPORTED data
Only analyzes the 200K records with complete data
"""

import os
import sys
import django
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import time

sys.path.insert(0, 'provider_lookup_web')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')
django.setup()

from apps.providers.models import Provider

print("="*70)
print("DIAGNOSTIC - NEW IMPORTED DATA ONLY")
print("="*70)

# Extract NEW data only (with non-empty credentials)
print("\n1. Extracting NEW imported data (10,000 sample)...")
new_providers = list(Provider.objects.exclude(
    credential=''
).exclude(
    middle_name=''
).order_by('?')[:10000].values())

df = pd.DataFrame(new_providers)
df.set_index('npi', inplace=True)

print(f"✅ Extracted {len(df):,} NEW provider records")

# Engineer features
print("\n2. Engineering features...")

df['is_individual'] = (df['entity_type'] == 'Individual').astype(int)
df['is_organization'] = (df['entity_type'] == 'Organization').astype(int)

# Fix: check for non-empty strings
df['has_first_name'] = (~df['first_name'].isna() & (df['first_name'] != '')).astype(int)
df['has_last_name'] = (~df['last_name'].isna() & (df['last_name'] != '')).astype(int)
df['has_middle_name'] = (~df['middle_name'].isna() & (df['middle_name'] != '')).astype(int)
df['has_organization_name'] = (~df['organization_name'].isna() & (df['organization_name'] != '')).astype(int)
df['has_name_prefix'] = (~df['name_prefix'].isna() & (df['name_prefix'] != '')).astype(int)
df['has_name_suffix'] = (~df['name_suffix'].isna() & (df['name_suffix'] != '')).astype(int)
df['has_credential'] = (~df['credential'].isna() & (df['credential'] != '')).astype(int)
df['has_phone'] = (~df['phone'].isna() & (df['phone'] != '')).astype(int)
df['has_fax'] = (~df['fax'].isna() & (df['fax'] != '')).astype(int)

# Temporal features
current_date = pd.Timestamp.now()
df['enumeration_date'] = pd.to_datetime(df['enumeration_date'], errors='coerce')
df['last_update_date'] = pd.to_datetime(df['last_update_date'], errors='coerce')

df['years_since_enumeration'] = (current_date - df['enumeration_date']).dt.days / 365.25
df['years_since_update'] = (current_date - df['last_update_date']).dt.days / 365.25

enum_median = df['years_since_enumeration'].median()
update_median = df['years_since_update'].median()

df['years_since_enumeration'].fillna(enum_median, inplace=True)
df['years_since_update'].fillna(update_median, inplace=True)

df['is_active'] = df['deactivation_date'].isna().astype(int)

# Composite scores
name_fields = ['has_first_name', 'has_last_name', 'has_middle_name', 
              'has_organization_name', 'has_name_prefix', 'has_name_suffix']
df['name_completeness_score'] = df[name_fields].sum(axis=1) / len(name_fields)

contact_fields = ['has_phone', 'has_fax', 'has_credential']
df['contact_completeness_score'] = df[contact_fields].sum(axis=1) / len(contact_fields)

df['overall_quality_score'] = (df['name_completeness_score'] + 
                               df['contact_completeness_score']) / 2

clustering_features = [
    'is_individual', 'is_organization', 'has_first_name', 'has_last_name',
    'has_middle_name', 'has_organization_name', 'has_credential',
    'has_phone', 'has_fax', 'years_since_enumeration', 'years_since_update',
    'is_active', 'name_completeness_score', 'contact_completeness_score',
    'overall_quality_score'
]

features_df = df[clustering_features].copy()

# Fill any remaining NaN
for col in features_df.columns:
    features_df[col].fillna(0, inplace=True)

print(f"✅ Created {len(clustering_features)} features")

# DIAGNOSTIC RESULTS
print("\n" + "="*70)
print("DIAGNOSTIC RESULTS")
print("="*70)

print("\n📊 FEATURE STATISTICS:")
print("-"*70)
print(features_df.describe())

print("\n📈 FEATURE VARIANCE:")
print("-"*70)
variances = features_df.var()
print(variances)

zero_var = variances[variances == 0]
print(f"\nFeatures with zero variance: {len(zero_var)}")
if len(zero_var) > 0:
    print("⚠️  WARNING: Some features have zero variance!")
    print(zero_var)
else:
    print("✅ All features have variance!")

print("\n🔢 UNIQUE VALUES PER FEATURE:")
print("-"*70)
for col in features_df.columns:
    unique_count = features_df[col].nunique()
    print(f"{col}: {unique_count} unique values")
    if unique_count <= 5:
        value_counts = features_df[col].value_counts()
        print(f"  Distribution: {dict(value_counts)}")

# Scale features
print("\n🎯 SCALED FEATURES CHECK:")
print("-"*70)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

print(f"Scaled shape: {scaled_features.shape}")
print(f"Contains NaN: {np.isnan(scaled_features).any()}")
print(f"Contains Inf: {np.isinf(scaled_features).any()}")
print(f"Min value: {np.min(scaled_features):.3f}")
print(f"Max value: {np.max(scaled_features):.3f}")

# Quick clustering test
print("\n⚡ QUICK CLUSTERING TEST:")
print("-"*70)

start = time.time()
minibatch = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=1000, n_init=3)
labels = minibatch.fit_predict(scaled_features)
elapsed = time.time() - start

silhouette = silhouette_score(scaled_features, labels)

print(f"  Clusters: 4")
print(f"  Time: {elapsed:.2f}s")
print(f"  Silhouette: {silhouette:.4f}")
print(f"  Cluster sizes: {np.bincount(labels)}")

# Assessment
print("\n" + "="*70)
print("ASSESSMENT")
print("="*70)

checks = {
    'No zero-variance features': len(zero_var) == 0,
    'Reasonable silhouette (0.2-0.7)': 0.2 < silhouette < 0.7,
    'No NaN in scaled features': not np.isnan(scaled_features).any(),
    'Multiple unique values': all(features_df.nunique() > 1),
}

for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {check}")

all_passed = all(checks.values())

if all_passed:
    print("\n🎉 DATA QUALITY EXCELLENT!")
    print("✅ Ready for Phase 2 clustering analysis")
    print("\n🚀 Run: python data_mining/02_clustering_analysis_optimized.py")
else:
    print("\n⚠️  Data quality issues detected")
    
    if len(zero_var) > 0:
        print(f"\n💡 SOLUTION: Remove {len(zero_var)} zero-variance features:")
        print("   These features have no information value for clustering")
        for feat in zero_var.index:
            print(f"   • {feat}")
    
    if not (0.2 < silhouette < 0.7):
        print(f"\n💡 Silhouette score: {silhouette:.4f}")
        if silhouette > 0.9:
            print("   Too high! Suggests only 2 distinct groups (Individual/Org)")
            print("   This is OK - clustering will focus on entity type")
        elif silhouette < 0.2:
            print("   Low clustering quality - data may be too homogeneous")

print("\n" + "="*70)