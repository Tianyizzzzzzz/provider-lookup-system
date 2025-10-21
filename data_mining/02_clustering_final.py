"""
Phase 2: Provider Clustering Analysis - ADAPTIVE VERSION
Automatically removes zero-variance features
Uses only NEW imported data (200K records)
"""

import os
import sys
import django
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'provider_lookup_web')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')
django.setup()

from apps.providers.models import Provider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*70)
print("PHASE 2: ADAPTIVE PROVIDER CLUSTERING ANALYSIS")
print("="*70)

class AdaptiveClusteringAnalysis:
    """Adaptive clustering that handles zero-variance features"""
    
    def __init__(self, sample_size=200000, random_state=42):
        self.sample_size = sample_size
        self.random_state = random_state
        self.output_dir = 'data_mining/clustering_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.raw_data = None
        self.features_df = None
        self.scaled_features = None
        self.cluster_labels = None
        self.results = {}
        self.removed_features = []
        
        logger.info(f"Adaptive clustering initialized (sample: {sample_size:,})")
    
    def extract_sample_data(self):
        """Extract NEW imported data only (with non-empty credentials)"""
        logger.info("Step 1/7: Extracting NEW imported data...")
        
        # Get only new data (non-empty credentials and middle names)
        providers_query = Provider.objects.exclude(
            credential=''
        ).exclude(
            middle_name=''
        ).order_by('?')[:self.sample_size]
        
        providers_data = list(providers_query.values())
        self.raw_data = pd.DataFrame(providers_data)
        
        logger.info(f"✅ Extracted {len(self.raw_data):,} NEW provider records")
        logger.info(f"   (Filtered for non-empty credentials)")
        return True
    
    def engineer_features(self):
        """Engineer features with empty string handling"""
        logger.info("Step 2/7: Engineering features...")
        
        df = self.raw_data.copy()
        df.set_index('npi', inplace=True)
        
        # Binary features - CHECK FOR EMPTY STRINGS
        df['is_individual'] = (df['entity_type'] == 'Individual').astype(int)
        df['is_organization'] = (df['entity_type'] == 'Organization').astype(int)
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
        
        # Fill NaN with median
        df['years_since_enumeration'] = df['years_since_enumeration'].fillna(
            df['years_since_enumeration'].median()
        )
        df['years_since_update'] = df['years_since_update'].fillna(
            df['years_since_update'].median()
        )
        
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
        
        self.features_df = df[clustering_features].copy()
        
        # Fill any remaining NaN
        for col in self.features_df.columns:
            self.features_df[col] = self.features_df[col].fillna(0)
        
        logger.info(f"✅ Engineered {len(clustering_features)} initial features")
        
        # CHECK FOR ZERO VARIANCE
        variances = self.features_df.var()
        zero_var_features = variances[variances < 0.001].index.tolist()
        
        if zero_var_features:
            logger.info(f"⚠️  Removing {len(zero_var_features)} zero-variance features:")
            for feat in zero_var_features:
                logger.info(f"     • {feat}")
            
            self.removed_features = zero_var_features
            self.features_df = self.features_df.drop(columns=zero_var_features)
        
        logger.info(f"✅ Final feature count: {len(self.features_df.columns)}")
        logger.info(f"   Active features: {list(self.features_df.columns)}")
        
        return True
    
    def scale_features(self):
        """Standardize features"""
        logger.info("Step 3/7: Scaling features...")
        
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(self.features_df)
        
        logger.info(f"✅ Features scaled: shape {self.scaled_features.shape}")
        return True
    
    def determine_optimal_clusters(self):
        """Find optimal k using FAST method"""
        logger.info("Step 4/7: Determining optimal clusters (FAST mode)...")
        
        max_k = 10
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        k_range = range(2, max_k + 1)
        
        # Use SAMPLE for silhouette (much faster)
        sample_size = min(10000, len(self.scaled_features))
        sample_indices = np.random.choice(len(self.scaled_features), sample_size, replace=False)
        sample_features = self.scaled_features[sample_indices]
        
        logger.info(f"  Using {sample_size:,} sample for k optimization (faster)")
        
        total_start = time.time()
        
        for k in k_range:
            iter_start = time.time()
            logger.info(f"  Testing k={k}...")
            
            # Cluster on full data
            minibatch = MiniBatchKMeans(
                n_clusters=k, random_state=self.random_state,
                batch_size=2000, n_init=3, max_iter=100
            )
            labels_full = minibatch.fit_predict(self.scaled_features)
            
            # But evaluate on SAMPLE (much faster)
            labels_sample = minibatch.predict(sample_features)
            
            inertias.append(minibatch.inertia_)
            silhouette_scores.append(silhouette_score(sample_features, labels_sample))
            davies_bouldin_scores.append(davies_bouldin_score(sample_features, labels_sample))
            
            iter_time = time.time() - iter_start
            logger.info(f"    {iter_time:.1f}s (Silhouette: {silhouette_scores[-1]:.3f})")
        
        total_time = time.time() - total_start
        
        # Find optimal k
        inertia_deltas_pct = np.abs(np.diff(inertias)) / inertias[:-1] * 100
        elbow_idx = np.where(inertia_deltas_pct < 5)[0]
        optimal_k_elbow = k_range[elbow_idx[0] + 1] if len(elbow_idx) > 0 else 4
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        recommended_k = int((optimal_k_elbow + optimal_k_silhouette) / 2)
        
        logger.info(f"✅ Optimal k: {recommended_k} (took {total_time:.1f}s total)")
        
        self.results['elbow_analysis'] = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'recommended_k': recommended_k,
            'total_time': total_time
        }
        
        return recommended_k
    
    def perform_clustering(self, n_clusters=4):
        """Perform clustering"""
        logger.info(f"Step 5/7: Final clustering (k={n_clusters})...")
        
        start = time.time()
        minibatch = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=self.random_state,
            batch_size=2000, n_init=10, max_iter=300
        )
        minibatch_labels = minibatch.fit_predict(self.scaled_features)
        minibatch_time = time.time() - start
        minibatch_silhouette = silhouette_score(self.scaled_features, minibatch_labels)
        
        logger.info(f"  MiniBatch: {minibatch_time:.1f}s, Silhouette: {minibatch_silhouette:.3f}")
        
        self.cluster_labels = minibatch_labels
        self.features_df['cluster'] = minibatch_labels
        
        self.results['clustering'] = {
            'n_clusters': n_clusters,
            'minibatch_silhouette': minibatch_silhouette,
            'minibatch_time': minibatch_time,
            'cluster_centers': minibatch.cluster_centers_,
            'cluster_sizes': pd.Series(minibatch_labels).value_counts().to_dict()
        }
        
        logger.info(f"✅ Clustering complete")
        return True
    
    def analyze_clusters(self):
        """Analyze clusters"""
        logger.info("Step 6/7: Analyzing clusters...")
        
        analysis_df = self.features_df.copy()
        analysis_df['entity_type'] = self.raw_data['entity_type'].values
        
        cluster_profiles = {}
        
        for cluster_id in range(self.results['clustering']['n_clusters']):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(analysis_df) * 100,
                'avg_years_active': cluster_data.get('years_since_enumeration', pd.Series([0])).mean(),
                'active_rate': cluster_data.get('is_active', pd.Series([1])).mean() * 100,
            }
            
            # Add available scores
            for score_col in ['name_completeness_score', 'contact_completeness_score', 'overall_quality_score']:
                if score_col in cluster_data.columns:
                    profile[score_col] = cluster_data[score_col].mean() * 100
            
            cluster_profiles[f'Cluster {cluster_id}'] = profile
        
        self.results['cluster_profiles'] = cluster_profiles
        
        logger.info("✅ Cluster analysis complete")
        for name, profile in cluster_profiles.items():
            logger.info(f"  {name}: {profile['size']:,} ({profile['percentage']:.1f}%)")
        
        return True
    
    def create_visualizations(self):
        """Generate visualizations"""
        logger.info("Step 7/7: Creating visualizations...")
        
        self._create_elbow_plot()
        self._create_cluster_pca()
        self._create_cluster_sizes()
        
        logger.info("✅ Visualizations complete")
        return True
    
    def _create_elbow_plot(self):
        """Elbow method plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        k_range = self.results['elbow_analysis']['k_range']
        inertias = self.results['elbow_analysis']['inertias']
        silhouette_scores = self.results['elbow_analysis']['silhouette_scores']
        recommended_k = self.results['elbow_analysis']['recommended_k']
        
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=recommended_k, color='red', linestyle='--', linewidth=2,
                   label=f'Recommended k={recommended_k}')
        ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax1.set_ylabel('Inertia', fontweight='bold')
        ax1.set_title('Elbow Method', fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=recommended_k, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontweight='bold')
        ax2.set_title('Silhouette Analysis', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Optimal Cluster Determination (n={len(self.features_df):,})',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/Fig1_Optimal_Clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cluster_pca(self):
        """PCA visualization"""
        pca = PCA(n_components=2, random_state=self.random_state)
        pca_features = pca.fit_transform(self.scaled_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1],
                           c=self.cluster_labels, cmap='Set2',
                           alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
        ax.set_title('Provider Clusters (PCA Projection)', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/Fig2_Cluster_PCA.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cluster_sizes(self):
        """Cluster size distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cluster_profiles = self.results['cluster_profiles']
        names = list(cluster_profiles.keys())
        sizes = [cluster_profiles[c]['size'] for c in names]
        percentages = [cluster_profiles[c]['percentage'] for c in names]
        
        bars = ax.bar(names, sizes, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_ylabel('Number of Providers', fontweight='bold')
        ax.set_title('Cluster Size Distribution', fontweight='bold', pad=15)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/Fig3_Cluster_Sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate report"""
        logger.info("Generating report...")
        
        report = f"""
{'='*80}
PHASE 2: ADAPTIVE PROVIDER CLUSTERING ANALYSIS
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Sample: {len(self.features_df):,} providers (NEW imports only)

FEATURE ENGINEERING:
{'-'*80}
Initial features: 15
Zero-variance features removed: {len(self.removed_features)}
  {', '.join(self.removed_features) if self.removed_features else 'None'}

Final features: {len(self.features_df.columns)}
  {', '.join(self.features_df.columns)}

CLUSTERING RESULTS:
{'-'*80}
Optimal k: {self.results['clustering']['n_clusters']}
Silhouette Score: {self.results['clustering']['minibatch_silhouette']:.3f}
Clustering Time: {self.results['clustering']['minibatch_time']:.1f}s

CLUSTER PROFILES:
{'-'*80}
"""
        
        for name, profile in self.results['cluster_profiles'].items():
            report += f"""
{name}:
  Size: {profile['size']:,} ({profile['percentage']:.1f}%)
"""
            for key, value in profile.items():
                if key not in ['size', 'percentage']:
                    report += f"  {key}: {value:.1f}\n"
        
        report += f"""
READY FOR PHASE 3: CLASSIFICATION
{'='*80}
"""
        
        path = f'{self.output_dir}/Clustering_Report.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Report: {path}")
        return report
    
    def run_complete_analysis(self):
        """Execute pipeline"""
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("STARTING ADAPTIVE CLUSTERING ANALYSIS")
        print("="*70 + "\n")
        
        try:
            self.extract_sample_data()
            self.engineer_features()
            self.scale_features()
            optimal_k = self.determine_optimal_clusters()
            self.perform_clustering(n_clusters=optimal_k)
            self.analyze_clusters()
            self.create_visualizations()
            self.generate_report()
            
            elapsed = datetime.now() - start_time
            
            print("\n" + "="*70)
            print("✅ PHASE 2 COMPLETE!")
            print("="*70)
            print(f"Time: {elapsed}")
            print(f"\nResults: {self.output_dir}/")
            print(f"Clusters: {self.results['clustering']['n_clusters']}")
            print(f"Silhouette: {self.results['clustering']['minibatch_silhouette']:.3f}")
            print("\n🚀 READY FOR PHASE 3: CLASSIFICATION")
            print("="*70)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    analyzer = AdaptiveClusteringAnalysis(sample_size=200000, random_state=42)
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✅ Phase 2 complete!")
    else:
        print("\n❌ Phase 2 failed.")