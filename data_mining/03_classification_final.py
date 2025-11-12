"""
================================================================================
PHASE 3: PROVIDER CLUSTER CLASSIFICATION (UNIFIED DATA VERSION)
================================================================================
Purpose: Train supervised learning models to predict cluster membership
Input: Same data filtering as Phase 2 (non-empty credential and middle_name)
Output: Classification models, performance metrics, visualizations
Data Consistency: Uses EXACT same data filtering as Phase 2 for consistency
Expected Runtime: 45-60 minutes
================================================================================
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup Django
sys.path.insert(0, 'provider_lookup_web')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')
import django
django.setup()

from apps.providers.models import Provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProviderClassification:
    """
    Multi-class classification to predict provider cluster membership
    """
    
    def __init__(self, sample_size=150000, test_size=0.2, random_state=42):
        """
        Initialize classification analysis
        
        Args:
            sample_size: Number of providers to analyze
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
        """
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Output directory
        self.output_dir = Path('data_mining/classification_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'data_info': {},
            'models': {},
            'best_model': None,
            'feature_importance': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'sample_size': sample_size,
                'test_size': test_size,
                'random_state': random_state
            }
        }
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        logger.info("="*80)
        logger.info("PHASE 3: PROVIDER CLUSTER CLASSIFICATION")
        logger.info("="*80)
    
    def load_and_prepare_data(self):
        """
        Load providers and generate cluster labels (reproducing Phase 2)
        Uses EXACT same filtering as Phase 2: non-empty credentials and middle names
        """
        logger.info("\n📊 Step 1: Loading and Preparing Data...")
        
        try:
            # Load providers (EXACT same criteria as Phase 2)
            logger.info("   Loading providers with same filters as Phase 2...")
            logger.info("   Filters: non-empty credential AND non-empty middle_name")
            
            providers = Provider.objects.exclude(
                credential=''
            ).exclude(
                middle_name=''
            ).values(
                'npi', 'entity_type',
                'first_name', 'last_name', 
                'middle_name', 'organization_name',
                'credential',
                'phone', 'fax',
                'enumeration_date', 'last_update_date',
                'deactivation_date'
            )
            
            # Convert to list and limit to sample_size (like Phase 2)
            providers_list = list(providers[:self.sample_size])
            df = pd.DataFrame(providers_list)
            
            logger.info(f"   ✅ Loaded {len(df):,} providers from database")
            logger.info(f"      (using same filtering as Phase 2)")
            
            if len(df) == 0:
                raise ValueError("No providers found with non-empty credentials and middle names!")
            
            # No additional sampling needed - already limited by query
            
            # Feature engineering (same as Phase 2)
            logger.info("\n   🔧 Engineering features...")
            
            # Basic features
            df['is_individual'] = (df['entity_type'] == 'Individual').astype(int)
            df['is_organization'] = (df['entity_type'] == 'Organization').astype(int)
            
            # Name features (checking both NULL and empty string)
            df['has_first_name'] = (
                ~df['first_name'].isna() & (df['first_name'] != '')
            ).astype(int)
            df['has_last_name'] = (
                ~df['last_name'].isna() & (df['last_name'] != '')
            ).astype(int)
            df['has_middle_name'] = (
                ~df['middle_name'].isna() & (df['middle_name'] != '')
            ).astype(int)
            df['has_organization_name'] = (
                ~df['organization_name'].isna() & (df['organization_name'] != '')
            ).astype(int)
            df['has_credential'] = (
                ~df['credential'].isna() & (df['credential'] != '')
            ).astype(int)
            
            # Contact features
            df['has_phone'] = (
                ~df['phone'].isna() & (df['phone'] != '')
            ).astype(int)
            df['has_fax'] = (
                ~df['fax'].isna() & (df['fax'] != '')
            ).astype(int)
            
            # Temporal features
            today = pd.Timestamp.now()
            df['years_since_enumeration'] = (today - pd.to_datetime(df['enumeration_date'])).dt.days / 365.25
            df['years_since_update'] = (today - pd.to_datetime(df['last_update_date'])).dt.days / 365.25
            df['is_active'] = df['deactivation_date'].isna().astype(int)
            
            # Composite quality scores
            df['name_completeness_score'] = (
                df['has_first_name'] * 25 +
                df['has_last_name'] * 25 +
                df['has_middle_name'] * 15 +
                df['has_organization_name'] * 25 +
                df['has_credential'] * 10
            )
            
            df['contact_completeness_score'] = (
                df['has_phone'] * 50 +
                df['has_fax'] * 50
            )
            
            df['overall_quality_score'] = (
                df['name_completeness_score'] * 0.5 +
                df['contact_completeness_score'] * 0.3 +
                df['is_active'] * 20
            )
            
            # Define ALL potential feature columns
            all_feature_cols = [
                'is_individual', 'is_organization',
                'has_first_name', 'has_last_name', 'has_middle_name',
                'has_organization_name', 'has_credential',
                'has_phone', 'has_fax',
                'years_since_enumeration', 'years_since_update',
                'is_active',
                'name_completeness_score', 'contact_completeness_score',
                'overall_quality_score'
            ]
            
            # Remove zero-variance features (like Phase 2)
            variance = df[all_feature_cols].var()
            feature_cols = variance[variance > 0].index.tolist()
            
            if len(feature_cols) < len(all_feature_cols):
                removed = set(all_feature_cols) - set(feature_cols)
                logger.info(f"   ⚠️  Removed {len(removed)} zero-variance features: {removed}")
            
            logger.info(f"   ✅ Using {len(feature_cols)} features with variance > 0")
            
            # Prepare feature matrix
            X = df[feature_cols].copy()
            self.feature_names = feature_cols
            
            logger.info(f"\n   ✅ Features prepared:")
            logger.info(f"      Features: {len(feature_cols)}")
            logger.info(f"      Samples: {len(X):,}")
            
            # Generate cluster labels (reproduce Phase 2)
            logger.info("\n   🔄 Generating cluster labels (reproducing Phase 2)...")
            
            # Use same parameters as Phase 2
            kmeans = MiniBatchKMeans(
                n_clusters=3,  # From Phase 2 optimal k
                random_state=self.random_state,
                batch_size=1024,
                n_init=10,
                max_iter=300
            )
            
            # Fit and predict
            y = kmeans.fit_predict(X)
            y = pd.Series(y, index=X.index)
            
            logger.info(f"\n   ✅ Clustering complete:")
            logger.info(f"      Classes: {sorted(y.unique())}")
            logger.info(f"      Class distribution:")
            for cluster_id, count in y.value_counts().sort_index().items():
                pct = count / len(y) * 100
                logger.info(f"         Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
            
            # Verify sample size matches Phase 2 expectations
            expected_min = 100000  # Phase 2 had ~120K
            if len(X) < expected_min:
                logger.warning(f"\n   ⚠️  Sample size ({len(X):,}) is less than Phase 2 (~120K)")
                logger.warning(f"      This may be due to database having fewer matching records")
            else:
                logger.info(f"\n   ✅ Sample size ({len(X):,}) matches Phase 2 expectations")
            
            # Store data info
            self.results['data_info'] = {
                'n_samples': len(X),
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'n_classes': len(y.unique()),
                'class_distribution': y.value_counts().to_dict()
            }
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise
    
    def split_and_scale_data(self, X, y):
        """
        Split data into train/test and scale features
        """
        logger.info("\n🔀 Step 2: Splitting and Scaling Data...")
        
        # Stratified split to maintain class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"   Train set: {len(self.X_train):,} samples")
        logger.info(f"   Test set: {len(self.X_test):,} samples")
        logger.info(f"   Features scaled: StandardScaler")
        
        return True
    
    def train_models(self):
        """
        Train multiple classification models
        """
        logger.info("\n🤖 Step 3: Training Classification Models...")
        logger.info("   (This may take 30-45 minutes)")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"\n   Training: {name}...")
            start_time = time.time()
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Predict
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, n_jobs=-1)
                
                training_time = time.time() - start_time
                
                # Store results
                self.results['models'][name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"      ✅ Accuracy: {accuracy:.3f}")
                logger.info(f"      CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                logger.info(f"      Time: {training_time:.1f}s")
                
            except Exception as e:
                logger.error(f"      ❌ Failed: {e}")
                continue
        
        # Select best model
        best_model_name = max(
            self.results['models'].keys(),
            key=lambda x: self.results['models'][x]['accuracy']
        )
        self.results['best_model'] = best_model_name
        
        logger.info(f"\n   🏆 Best Model: {best_model_name}")
        logger.info(f"      Accuracy: {self.results['models'][best_model_name]['accuracy']:.3f}")
        
        return True
    
    def extract_feature_importance(self):
        """
        Extract feature importance from tree-based models
        """
        logger.info("\n📊 Step 4: Analyzing Feature Importance...")
        
        for name, model_data in self.results['models'].items():
            model = model_data['model']
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                self.results['feature_importance'][name] = {
                    feat: imp for feat, imp in zip(self.feature_names, importances)
                }
                
                logger.info(f"\n   {name} - Top 3 Features:")
                top_features = sorted(
                    zip(self.feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                for feat, imp in top_features:
                    logger.info(f"      {feat}: {imp:.3f}")
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_).mean(axis=0)
                self.results['feature_importance'][name] = {
                    feat: coef for feat, coef in zip(self.feature_names, coefficients)
                }
        
        return True
    
    def create_visualizations(self):
        """
        Create comprehensive visualization suite
        """
        logger.info("\n📈 Step 5: Creating Visualizations...")
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
        # Figure 1: Model Performance Comparison
        self._plot_model_comparison()
        
        # Figure 2: Confusion Matrix (Best Model)
        self._plot_confusion_matrix()
        
        # Figure 3: Feature Importance (Tree models)
        self._plot_feature_importance()
        
        # Figure 4: ROC Curves (if applicable)
        self._plot_roc_curves()
        
        logger.info("   ✅ All visualizations created")
        
        return True
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Classification Model Performance Comparison', 
                     fontsize=16, fontweight='bold')
        
        models = list(self.results['models'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            values = [self.results['models'][m][metric] for m in models]
            colors = ['#2ecc71' if m == self.results['best_model'] else '#3498db' 
                     for m in models]
            
            bars = ax.barh(models, values, color=colors, alpha=0.8)
            ax.set_xlabel(metric_name, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / 'Fig1_Model_Performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   📊 Saved: {output_path.name}")
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix for best model"""
        best_model_name = self.results['best_model']
        y_pred = self.results['models'][best_model_name]['predictions']
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Cluster {i}' for i in sorted(self.y_test.unique())],
                   yticklabels=[f'Cluster {i}' for i in sorted(self.y_test.unique())],
                   cbar_kws={'label': 'Count'})
        
        ax.set_title(f'Confusion Matrix: {best_model_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        
        # Add accuracy annotation
        accuracy = self.results['models'][best_model_name]['accuracy']
        ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.3f}', 
               transform=ax.transAxes, ha='center', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        output_path = self.output_dir / 'Fig2_Confusion_Matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   📊 Saved: {output_path.name}")
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        tree_models = {name: imp for name, imp in self.results['feature_importance'].items() 
                      if 'Forest' in name or 'Boosting' in name}
        
        if not tree_models:
            logger.info("   ⚠️  No tree-based models for feature importance")
            return
        
        fig, axes = plt.subplots(1, len(tree_models), figsize=(7*len(tree_models), 6))
        if len(tree_models) == 1:
            axes = [axes]
        
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        for ax, (model_name, importances) in zip(axes, tree_models.items()):
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            features, values = zip(*sorted_features)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
            bars = ax.barh(features, values, color=colors, alpha=0.8)
            
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(model_name, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        output_path = self.output_dir / 'Fig3_Feature_Importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   📊 Saved: {output_path.name}")
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for (name, model_data), color in zip(self.results['models'].items(), colors):
            if model_data['probabilities'] is not None:
                # For multi-class, use one-vs-rest approach
                y_test_bin = pd.get_dummies(self.y_test)
                y_pred_proba = model_data['probabilities']
                
                # Calculate ROC AUC for each class and average
                fpr_list, tpr_list, roc_auc_list = [], [], []
                for i in range(y_test_bin.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_bin.iloc[:, i], y_pred_proba[:, i])
                    roc_auc = roc_auc_score(y_test_bin.iloc[:, i], y_pred_proba[:, i])
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
                    roc_auc_list.append(roc_auc)
                
                # Plot macro-average
                avg_roc_auc = np.mean(roc_auc_list)
                ax.plot([], [], color=color, lw=2, 
                       label=f'{name} (AUC = {avg_roc_auc:.3f})')
                
                # Plot each class with lighter color
                for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
                    ax.plot(fpr, tpr, color=color, lw=1, alpha=0.3)
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'Fig4_ROC_Curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   📊 Saved: {output_path.name}")
    
    def generate_report(self):
        """
        Generate comprehensive text report
        """
        logger.info("\n📄 Step 6: Generating Report...")
        
        report_lines = [
            "="*80,
            "PHASE 3: PROVIDER CLUSTER CLASSIFICATION REPORT",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sample Size: {self.results['data_info']['n_samples']:,} providers",
            "",
            "DATA SUMMARY:",
            "-"*80,
            f"Features: {self.results['data_info']['n_features']}",
            f"Feature Names: {', '.join(self.results['data_info']['feature_names'])}",
            f"Classes: {self.results['data_info']['n_classes']}",
            "",
            "Class Distribution:",
        ]
        
        for cluster_id, count in sorted(self.results['data_info']['class_distribution'].items()):
            pct = count / self.results['data_info']['n_samples'] * 100
            report_lines.append(f"  Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
        
        report_lines.extend([
            "",
            "MODEL PERFORMANCE:",
            "-"*80,
        ])
        
        for name, model_data in self.results['models'].items():
            is_best = (name == self.results['best_model'])
            marker = " 🏆 BEST" if is_best else ""
            
            report_lines.extend([
                f"\n{name}{marker}:",
                f"  Accuracy:  {model_data['accuracy']:.4f}",
                f"  Precision: {model_data['precision']:.4f}",
                f"  Recall:    {model_data['recall']:.4f}",
                f"  F1 Score:  {model_data['f1_score']:.4f}",
                f"  CV Score:  {model_data['cv_mean']:.4f} ± {model_data['cv_std']:.4f}",
                f"  Training Time: {model_data['training_time']:.1f}s",
            ])
        
        # Feature importance
        if self.results['feature_importance']:
            report_lines.extend([
                "",
                "FEATURE IMPORTANCE:",
                "-"*80,
            ])
            
            for model_name, importances in self.results['feature_importance'].items():
                report_lines.append(f"\n{model_name}:")
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_features:
                    report_lines.append(f"  {feat}: {imp:.4f}")
        
        # Best model details
        best_name = self.results['best_model']
        best_model = self.results['models'][best_name]
        y_pred = best_model['predictions']
        
        report_lines.extend([
            "",
            "BEST MODEL CLASSIFICATION REPORT:",
            "-"*80,
            classification_report(self.y_test, y_pred, 
                                target_names=[f'Cluster {i}' for i in sorted(self.y_test.unique())]),
            "",
            "="*80,
            "PHASE 3 COMPLETE ✅",
            "="*80,
        ])
        
        # Save report with UTF-8 encoding (fix Windows encoding issue)
        report_text = '\n'.join(report_lines)
        report_path = self.output_dir / 'Classification_Report.txt'
        report_path.write_text(report_text, encoding='utf-8')
        
        logger.info(f"   📄 Saved: {report_path.name}")
        
        # Also save JSON results
        json_results = {
            'metadata': self.results['metadata'],
            'data_info': self.results['data_info'],
            'best_model': self.results['best_model'],
            'models': {
                name: {
                    'accuracy': data['accuracy'],
                    'precision': data['precision'],
                    'recall': data['recall'],
                    'f1_score': data['f1_score'],
                    'cv_mean': data['cv_mean'],
                    'cv_std': data['cv_std'],
                    'training_time': data['training_time']
                }
                for name, data in self.results['models'].items()
            },
            'feature_importance': self.results['feature_importance']
        }
        
        json_path = self.output_dir / 'classification_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"   📄 Saved: {json_path.name}")
        
        return True
    
    def run_complete_analysis(self):
        """
        Execute complete classification pipeline
        """
        start_time = time.time()
        
        try:
            # Step 1: Load data
            X, y = self.load_and_prepare_data()
            
            # Step 2: Split and scale
            self.split_and_scale_data(X, y)
            
            # Step 3: Train models
            self.train_models()
            
            # Step 4: Feature importance
            self.extract_feature_importance()
            
            # Step 5: Visualizations
            self.create_visualizations()
            
            # Step 6: Generate report
            self.generate_report()
            
            # Summary
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
            
            logger.info("\n" + "="*80)
            logger.info("✅ PHASE 3 CLASSIFICATION COMPLETE!")
            logger.info("="*80)
            logger.info(f"Total Runtime: {elapsed_str}")
            logger.info(f"Best Model: {self.results['best_model']}")
            logger.info(f"Best Accuracy: {self.results['models'][self.results['best_model']]['accuracy']:.3f}")
            logger.info(f"\nOutput Directory: {self.output_dir}/")
            logger.info("  - Fig1_Model_Performance.png")
            logger.info("  - Fig2_Confusion_Matrix.png")
            logger.info("  - Fig3_Feature_Importance.png")
            logger.info("  - Fig4_ROC_Curves.png")
            logger.info("  - Classification_Report.txt")
            logger.info("  - classification_results.json")
            logger.info("\n🎓 READY FOR THESIS WRITING!")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Classification failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Run classification analysis
    classifier = ProviderClassification(
        sample_size=150000,  # Use more data than Phase 2
        test_size=0.2,
        random_state=42
    )
    
    success = classifier.run_complete_analysis()
    
    if success:
        print("\n✅ Phase 3 Complete! All deliverables ready for thesis.")
    else:
        print("\n❌ Phase 3 encountered errors. Check logs above.")