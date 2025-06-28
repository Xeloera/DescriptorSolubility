from modules.config import Config
from modules.dataset import Fingerprint
from modules.model_picker import ModelSuite
from modules.models.mixtureofexpertsregressor import MixtureOfExpertsRegressor
from modules.functions.metrics import (rmse, mae, mape, r2, explained_var)
from modules.dataset import DripFeedingCV

from datetime import datetime
import json 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import cupy as cp
    from cuml.decomposition import PCA 
    from cuml.preprocessing import StandardScaler 
    from cuml.manifold import TSNE 
    from cuml.manifold import UMAP 
    from cuml.cluster import KMeans
    from cuml.linear_model import LinearRegression,LogisticRegression
    from cuml.ensemble import RandomForestRegressor, RandomForestClassifier
    USE_CUML = True
except ImportError:
    USE_CUML = False
    cp = None
    from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
try:
    import umap.umap_ as UMAP
    USE_UMAP = True
except ImportError:
    USE_UMAP = False
    
import pandas as pd

class Results:
    def __init__(self, config: Config):
        self.config = config
        
        
    def analyze_and_report_results(self, all_results, fold_results):
        """Comprehensive result analysis and reporting"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                **{k: v for k, v in result.items() if k not in ['predictions', 'true_values', 'test_indices', 'best_estimator']},
                **{f"metric_{k}": v for k, v in result['metrics'].items()}
            }
            for result in all_results
        ])
        
        # Overall statistics by model
        model_stats = results_df.groupby('model').agg({
            'metric_rmse': ['mean', 'std', 'min'],
            'metric_r2': ['mean', 'std', 'max'],
            'metric_mae': ['mean', 'std', 'min'],
            'metric_mape': ['mean', 'std', 'min']
        }).round(4)
        
        print("\nModel Performance Summary:")
        print(model_stats)
        
        # PCA impact analysis
        pca_impact = results_df.groupby(['model', 'pca_config']).agg({
            'metric_rmse': 'mean',
            'metric_r2': 'mean'
        }).round(4)
        
        print(f"\nPCA Impact Analysis:")
        print(pca_impact)
        
        # Solvent-specific analysis
        solvent_analysis = self.analyze_by_solvent(all_results)
        
        # Feature importance analysis
        self.analyze_feature_importance(fold_results)
        
        # Create visualizations
        if self.config.create_visualizations:
            self.create_comprehensive_visualizations(all_results, fold_results)
        
        # Save detailed results
        results_df.to_csv(self.output_dir / "detailed_results.csv", index=False)
        
        # Save summary statistics
        summary_stats = {
            'overall_best_rmse': float(results_df['metric_rmse'].min()),
            'overall_best_r2': float(results_df['metric_r2'].max()),
            'model_comparison': model_stats.to_dict(),
            'pca_impact': pca_impact.to_dict(),
            'total_configurations_tested': len(results_df),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Generate comprehensive report
        if self.config.create_report:
            self.generate_comprehensive_report(summary_stats, solvent_analysis)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
    
    def analyze_and_report_results_enhanced(self, all_results, output_dir, df, feature_matrices, X):
        """Enhanced result analysis with OpenCOSMO and drip-feeding comparison"""
        # Store parameters as instance variables for use in other methods
        self.output_dir = output_dir
        self.df = df
        self.feature_matrices = feature_matrices
        self.y = df[self.config.target_col]
        
        # Set up feature matrix and names from feature_matrices
        # Note: feature_matrices contains tuples of (X_matrix, feature_names)
        if feature_matrices and 'with_cosmo' in feature_matrices:
            self.X, self.feature_names = feature_matrices['with_cosmo']
        elif feature_matrices and len(feature_matrices) > 0:
            # Use the first available feature matrix
            first_key = list(feature_matrices.keys())[0]
            self.X, self.feature_names = feature_matrices[first_key]
        else:
            # Fallback - use the passed X parameter
            self.X = X
            self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ENHANCED RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                **{k: v for k, v in result.items() if k not in ['predictions', 'true_values', 'test_indices', 'best_estimator', 'feature_names']},
                **{f"metric_{k}": v for k, v in result['metrics'].items()}
            }
            for result in all_results
        ])
        
        # MoE Analysis - Show detailed info for best MoE models
        moe_results = [r for r in all_results if 'moe' in r['model']]
        if moe_results:
            print("\n" + "="*60)
            print("MIXTURE OF EXPERTS ANALYSIS")
            print("="*60)
            
            for result in moe_results:
                if 'best_estimator' in result:
                    try:
                        estimator = result['best_estimator']
                        if hasattr(estimator, 'named_steps') and 'model' in estimator.named_steps:
                            moe_model = estimator.named_steps['model']
                            if isinstance(moe_model, MixtureOfExpertsRegressor):
                                print(f"\nMoE Model: {result['model']} (Fold {result['fold']})")
                                print(f"  Matrix: {result['matrix_type']}, CV: {result['cv_strategy']}")
                                print(f"  Performance: RMSE={result['metrics']['rmse']:.4f}, R²={result['metrics']['r2']:.4f}")
                                if hasattr(moe_model, 'threshold') and moe_model.threshold is not None:
                                    if moe_model.threshold_method == "solubility_cutoff":
                                        print(f"  Solubility Threshold: logS = {moe_model.threshold:.1f}")
                                        # Force a quick summary fit to show threshold info
                                        if len(result['true_values']) > 0:
                                            gate_labels = (result['true_values'] > moe_model.threshold).astype(int)
                                            insoluble_count = (gate_labels == 0).sum()
                                            soluble_count = (gate_labels == 1).sum()
                                            total = len(gate_labels)
                                            print(f"  Split: Insoluble={insoluble_count}({insoluble_count/total*100:.0f}%) | Soluble={soluble_count}({soluble_count/total*100:.0f}%)")
                                    else:
                                        print(f"  Threshold: {moe_model.threshold:.3f}")
                                        # Force a quick summary fit to show threshold info
                                        if len(result['true_values']) > 0:
                                            gate_labels = (result['true_values'] > moe_model.threshold).astype(int)
                                            low_count = (gate_labels == 0).sum()
                                            high_count = (gate_labels == 1).sum()
                                            total = len(gate_labels)
                                            print(f"  Split: Low={low_count}({low_count/total*100:.0f}%) | High={high_count}({high_count/total*100:.0f}%)")
                    except Exception as e:
                        print(f"  Error analyzing MoE model: {e}")
        
        # OpenCOSMO impact analysis
        print("\n" + "="*60)
        print("OPENCOSMO IMPACT ANALYSIS")
        print("="*60)
        
        cosmo_comparison = results_df.groupby(['matrix_type', 'model']).agg({
            'metric_rmse': ['mean', 'std'],
            'metric_r2': ['mean', 'std'],
            'metric_mae': ['mean', 'std']
        }).round(4)
        
        print("OpenCOSMO Feature Impact:")
        print(cosmo_comparison)
        
        # CV Strategy comparison
        print("\n" + "="*60)
        print("CROSS-VALIDATION STRATEGY COMPARISON")
        print("="*60)
        
        cv_comparison = results_df.groupby(['cv_strategy', 'model']).agg({
            'metric_rmse': ['mean', 'std'],
            'metric_r2': ['mean', 'std'],
            'metric_mae': ['mean', 'std']
        }).round(4)
        
        print("CV Strategy Comparison:")
        print(cv_comparison)
        
        # Solvent-specific analysis
        results_by_solvent = self.analyze_by_solvent(all_results)
        
        # Overall best results
        print("\n" + "="*60)
        print("BEST OVERALL RESULTS")
        print("="*60)
        
        best_result = results_df.loc[results_df['metric_rmse'].idxmin()]
        print(f"Best Configuration:")
        print(f"  Model: {best_result['model']}")
        print(f"  Matrix: {best_result['matrix_type']}")
        print(f"  CV: {best_result['cv_strategy']}")
        print(f"  PCA: {best_result['pca_config']}")
        print(f"  RMSE: {best_result['metric_rmse']:.4f}")
        print(f"  R²: {best_result['metric_r2']:.4f}")
        
        # Print solvent analysis
        if results_by_solvent:
            print(f"\nSOLVENT ANALYSIS:")
            print("-" * 20)
            valid_solvents = {k: v for k, v in results_by_solvent.items() if not np.isnan(v['rmse'])}
            for solvent, metrics in sorted(valid_solvents.items(), key=lambda x: x[1]['rmse']):
                print(f"{solvent}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        # Save detailed results
        results_df.to_csv(self.output_dir / "enhanced_detailed_results.csv", index=False)
        
        # Create enhanced visualizations
        if self.config.create_visualizations:
            self.create_enhanced_visualizations(all_results, results_df)
        
        # Enhanced summary statistics
        summary_stats = {
            'overall_best_rmse': float(results_df['metric_rmse'].min()),
            'overall_best_r2': float(results_df['metric_r2'].max()),
            'opencosmo_comparison': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in cosmo_comparison.to_dict().items()},
            'cv_strategy_comparison': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in cv_comparison.to_dict().items()},
            'best_configuration': {
                'matrix_type': best_result['matrix_type'],
                'cv_strategy': best_result['cv_strategy'],
                'model': best_result['model'],
                'pca_config': str(best_result['pca_config']),
                'rmse': float(best_result['metric_rmse']),
                'r2': float(best_result['metric_r2'])
            },
            'total_configurations_tested': len(results_df),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "enhanced_summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Generate enhanced report with solvent analysis
        if self.config.create_report:
            self.generate_enhanced_report(summary_stats, results_df, results_by_solvent)
        
        print(f"\nEnhanced analysis complete! Results saved to: {self.output_dir}")
        
        return summary_stats
    
    def analyze_by_solvent(self, all_results):
        """Analyze performance by solvent using names for readability"""
        print(f"\nAnalyzing performance by solvent...")
        
        solvent_results = {}
        
        for result in all_results:
            test_indices = result['test_indices']
            y_true = result['true_values']
            y_pred = result['predictions']
            
            # Get solvent names for test indices (for readable output)
            test_solvent_names = self.df.iloc[test_indices][self.config.solvent_name_col]
            
            for i, solvent_name in enumerate(test_solvent_names):
                if solvent_name not in solvent_results:
                    solvent_results[solvent_name] = {'true': [], 'pred': []}
                
                solvent_results[solvent_name]['true'].append(y_true[i])
                solvent_results[solvent_name]['pred'].append(y_pred[i])
        
        # Calculate metrics by solvent
        solvent_metrics = {}
        for solvent_name, data in solvent_results.items():

            true_vals = np.array(data['true'])
            pred_vals = np.array(data['pred'])
            
            solvent_metrics[solvent_name] = {
                'count': len(true_vals),
                'rmse': rmse(true_vals, pred_vals),
                'r2': r2(true_vals, pred_vals),
                'mae': mae(true_vals, pred_vals)}
                
        
        # Save solvent analysis
        solvent_df = pd.DataFrame.from_dict(solvent_metrics, orient='index')
        if not solvent_df.empty:
            solvent_df.to_csv(self.output_dir / "solvent_analysis.csv")
            print(f"Analyzed {len(solvent_metrics)} solvents with sufficient data")
        
        return solvent_metrics
    
    def analyze_feature_importance(self, fold_results):
        """Analyze feature importance across folds"""
        print(f"\nAnalyzing feature importance...")
        
        importances_by_fold = {}
        
        for fold_idx, fold_result in fold_results.items():
            model = fold_result.get('best_estimator')
            if model and hasattr(model.named_steps.get('model'), 'feature_importances_'):
                importances = model.named_steps['model'].feature_importances_
                importances_by_fold[ fold_idx] = importances
        
        if importances_by_fold:
            # Average importance across folds
            all_importances = np.array(list(importances_by_fold.values()))
            mean_importance = np.mean(all_importances, axis=0)
            std_importance = np.std(all_importances, axis=0)
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(mean_importance)],
                'importance_mean': mean_importance,
                'importance_std': std_importance
            }).sort_values('importance_mean', ascending=False)
            
            importance_df.to_csv(self.output_dir / "feature_importance.csv", index=False)
            print(f"Saved feature importance analysis for {len(importance_df)} features")
    
    def create_comprehensive_visualizations(self, all_results, fold_results):
        """Create comprehensive visualization suite"""
        print(f"\nCreating visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model comparison plot
        self.plot_model_comparison(all_results, viz_dir)
        
        # 2. Predictions vs actual
        self.plot_predictions_vs_actual(all_results, viz_dir)
        
        # 3. Feature importance
        self.plot_feature_importance(viz_dir)
        
        # 4. Solvent analysis
        self.plot_solvent_analysis(viz_dir)
        
        # 5. Dimensionality reduction visualization
        if self.config.use_dscribe:
            self.plot_dimensionality_reduction(viz_dir)
        
        print(f"Visualizations saved to: {viz_dir}")
    
    def plot_model_comparison(self, all_results, viz_dir):
        """Plot model comparison"""
        results_df = pd.DataFrame([
            {
                'model': result['model'],
                'rmse': result['metrics']['rmse'],
                'r2': result['metrics']['r2'],
                'mae': result['metrics']['mae']
            }
            for result in all_results
        ])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RMSE comparison
        sns.boxplot(data=results_df, x='model', y='rmse', ax=axes[0])
        axes[0].set_title('RMSE by Model')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R² comparison  
        sns.boxplot(data=results_df, x='model', y='r2', ax=axes[1])
        axes[1].set_title('R² by Model')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        sns.boxplot(data=results_df, x='model', y='mae', ax=axes[2])
        axes[2].set_title('MAE by Model')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions_vs_actual(self, all_results, viz_dir):
        """Plot predictions vs actual values"""
        all_true = []
        all_pred = []
        
        for result in all_results:
            all_true.extend(result['true_values'])
            all_pred.extend(result['predictions'])
        
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(all_true, all_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs True Values (All Models)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        overall_r2 = r2(all_true, all_pred)
        overall_rmse = rmse(all_true, all_pred)
        plt.text(0.05, 0.95, f'R² = {overall_r2:.3f}\nRMSE = {overall_rmse:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, viz_dir):
        """Plot feature importance if available"""
        importance_file = self.output_dir / "feature_importance.csv"
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
            
            # Plot top 30 features
            top_features = importance_df.head(30)
            
            plt.figure(figsize=(12, 10))
            plt.barh(range(len(top_features)), top_features['importance_mean'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 30 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(viz_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_solvent_analysis(self, viz_dir):
        """Plot solvent analysis if available"""
        solvent_file = self.output_dir / "solvent_analysis.csv"
        if solvent_file.exists():
            solvent_df = pd.read_csv(solvent_file, index_col=0)
            
            if len(solvent_df) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # RMSE by solvent
                solvent_df_sorted = solvent_df.sort_values('rmse')
                axes[0,0].bar(range(len(solvent_df_sorted)), solvent_df_sorted['rmse'])
                axes[0,0].set_title('RMSE by Solvent')
                axes[0,0].set_xticks(range(len(solvent_df_sorted)))

                axes[0,0].set_xticklabels(solvent_df_sorted.index, rotation=45, ha='right')
                
                # R² by solvent
                axes[0,1].bar(range(len(solvent_df_sorted)), solvent_df_sorted['r2'])
                axes[0,1].set_title('R² by Solvent')
                axes[0,1].set_xticks(range(len(solvent_df_sorted)))
                axes[0,1].set_xticklabels(solvent_df_sorted.index, rotation=45, ha='right')
                
                # Count by solvent
                axes[1,0].bar(range(len(solvent_df_sorted)), solvent_df_sorted['count'])
                axes[1,0].set_title('Prediction Count by Solvent')
                axes[1,0].set_xticks(range(len(solvent_df_sorted)))
                axes[1,0].set_xticklabels(solvent_df_sorted.index, rotation=45, ha='right')
                
                # Scatter plot
                axes[1,1].scatter(solvent_df['r2'], solvent_df['rmse'], 
                                s=solvent_df['count']*2, alpha=0.7)
                axes[1,1].set_xlabel('R²')
                axes[1,1].set_ylabel('RMSE')
                axes[1,1].set_title('RMSE vs R² (size = count)')
                
                plt.tight_layout()
                plt.savefig(viz_dir / "solvent_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def plot_dimensionality_reduction(self, viz_dir):
        """Plot dimensionality reduction visualization"""
        if len(self.X) > 1000:  # Sample for visualization
            indices = np.random.choice(len(self.X), 1000, replace=False)
            X_sample = self.X[indices]
            y_sample = self.y[indices]
        else:
            X_sample = self.X
            y_sample = self.y
        
        # t-SNE
        if self.config.use_gpu_viz and USE_CUML:
            tsne = TSNE(n_components=2, perplexity=self.config.tsne_perplexity, 
                         #random_state=SEED
                         )
            X_tsne = tsne.fit_transform(X_sample)
            if hasattr(X_tsne, 'get'):  # Convert from CuPy if needed
                X_tsne = X_tsne.get()
        else:
            tsne = TSNE(n_components=2, perplexity=self.config.tsne_perplexity, 
                       #random_state=SEED, 
                       n_jobs=self.config.n_jobs)
            X_tsne = tsne.fit_transform(X_sample)
        
       
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                            cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Solubility')
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # UMAP (if available)
        if USE_UMAP or (self.config.use_gpu_viz and USE_CUML):
            if self.config.use_gpu_viz and USE_CUML:
                umap_model = UMAP(n_neighbors=self.config.umap_neighbors, 
                                   #random_state=SEED
                                   )
                X_umap = umap_model.fit_transform(X_sample)
                if hasattr(X_umap, 'get'):
                    X_umap = X_umap.get()
            else:
                umap_model = UMAP(n_neighbors=self.config.umap_neighbors, 
                                 #random_state=SEED
                                 )
                X_umap = umap_model.fit_transform(X_sample)
            
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_sample, 
                                cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Solubility')
            plt.title('UMAP Visualization')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "dimensionality_reduction.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, summary_stats, solvent_analysis):
        """Generate comprehensive Markdown report"""
        report_path = self.output_dir / "comprehensive_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Solubility Prediction Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Dataset:** {self.config.tsv_path}\n")
            f.write(f"- **Target:** {self.config.target_col}\n")
            f.write(f"- **Models tested:** {', '.join(self.config.models)}\n")
            f.write(f"- **DScribe descriptors:** {'Yes' if self.config.use_dscribe else 'No'}\n")
            f.write(f"- **GPU acceleration:** {'Yes' if self.config.use_gpu else 'No'}\n")
            f.write(f"- **Cross-validation:** {self.config.outer_folds}-fold outer, {self.config.inner_folds}-fold inner\n\n")
            
            f.write("## Dataset Summary\n\n")
            f.write(f"- **Total samples:** {len(self.df)}\n")
            f.write(f"- **Features:** {self.X.shape[1]}\n")
            f.write(f"- **Unique solvents:** {self.df[self.config.solvent_col].nunique()}\n")
            f.write(f"- **Target range:** {self.y.min():.3f} to {self.y.max():.3f}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write(f"- **Best RMSE:** {summary_stats['overall_best_rmse']:.4f}\n")
            f.write(f"- **Best R²:** {summary_stats['overall_best_r2']:.4f}\n")
            f.write(f"- **Configurations tested:** {summary_stats['total_configurations_tested']}\n\n")
            
            if solvent_analysis:
                f.write("## Solvent-Specific Performance\n\n")
                f.write("| Solvent | RMSE | R² | MAE | Count |\n")
                f.write("|---------|------|----|----|-------|\n")
                
                sorted_solvents = sorted(solvent_analysis.items(), 
                                       key=lambda x: x[1]['rmse'])
                for solvent, metrics in sorted_solvents[:10]:  # Top 10
                    f.write(f"| {solvent} | {metrics['rmse']:.4f} | "
                           f"{metrics['r2']:.4f} | {metrics['mae']:.4f} | "
                           f"{metrics['count']} |\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `detailed_results.csv` - Complete results for all configurations\n")
            f.write("- `summary_statistics.json` - Summary statistics\n")
            f.write("- `feature_importance.csv` - Feature importance analysis\n")
            f.write("- `solvent_analysis.csv` - Per-solvent performance\n")
            f.write("- `visualizations/` - All plots and charts\n")
            f.write("- `best_model_fold_*.pkl` - Best models from each fold\n")
        
        print(f"Comprehensive report saved to: {report_path}")
    
    def generate_enhanced_report(self, summary_stats, results_df, results_by_solvent=None):
        """Generate enhanced Markdown report"""
        report_path = self.output_dir / "enhanced_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Solubility Prediction Analysis Report (Detailed)\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Dataset:** {self.config.tsv_path}\n")
            f.write(f"- **Target:** {self.config.target_col}\n")
            f.write(f"- **Models tested:** {', '.join(self.config.models)}\n")
            f.write(f"- **DScribe descriptors:** {'Yes' if self.config.use_dscribe else 'No'}\n")
            f.write(f"- **GPU acceleration:** {'Yes' if self.config.use_gpu else 'No'}\n")
            f.write(f"- **Cross-validation:** {self.config.outer_folds}-fold outer, {self.config.inner_folds}-fold inner\n\n")
            
            f.write("## Dataset Summary\n\n")
            f.write(f"- **Total samples:** {len(self.df)}\n")
            f.write(f"- **Features:** {self.X.shape[1]}\n")
            f.write(f"- **Unique solvents:** {self.df[self.config.solvent_col].nunique()}\n")
            f.write(f"- **Target range:** {self.y.min():.3f} to {self.y.max():.3f}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write(f"- **Best RMSE:** {summary_stats['overall_best_rmse']:.4f}\n")
            f.write(f"- **Best R²:** {summary_stats['overall_best_r2']:.4f}\n")
            f.write(f"- **Configurations tested:** {summary_stats['total_configurations_tested']}\n\n")
            
            f.write("## OpenCOSMO Impact Analysis\n\n")
            f.write("| Matrix Type | Model | RMSE (mean ± std) | R² (mean ± std) | MAE (mean ± std) |\n")
            f.write("|-------------|-------|-------------------|-----------------|-----------------|\n")
            
            for (matrix_type, model), group in results_df.groupby(['matrix_type', 'model']):
                rmse_mean = group['metric_rmse'].mean()
                rmse_std = group['metric_rmse'].std()
                r2_mean = group['metric_r2'].mean()
                r2_std = group['metric_r2'].std()
                mae_mean = group['metric_mae'].mean()
                mae_std = group['metric_mae'].std()
                
                f.write(f"| {matrix_type} | {model} | {rmse_mean:.4f} ± {rmse_std:.4f} | "
                       f"{r2_mean:.4f} ± {r2_std:.4f} | {mae_mean:.4f} ± {mae_std:.4f} |\n")
            
            f.write("\n## Cross-Validation Strategy Comparison\n\n")
            f.write("| CV Strategy | Model | RMSE (mean ± std) | R² (mean ± std) | MAE (mean ± std) |\n")
            f.write("|-------------|-------|-------------------|-----------------|-----------------|\n")
            
            for (cv_strategy, model), group in results_df.groupby(['cv_strategy', 'model']):
                rmse_mean = group['metric_rmse'].mean()
                rmse_std = group['metric_rmse'].std()
                r2_mean = group['metric_r2'].mean()
                r2_std = group['metric_r2'].std()
                mae_mean = group['metric_mae'].mean()
                mae_std = group['metric_mae'].std()
                
                f.write(f"| {cv_strategy} | {model} | {rmse_mean:.4f} ± {rmse_std:.4f} | "
                       f"{r2_mean:.4f} ± {r2_std:.4f} | {mae_mean:.4f} ± {mae_std:.4f} |\n")
            
            f.write("\n## Best Overall Configuration\n\n")
            f.write(f"- **Matrix Type:** {summary_stats['best_configuration']['matrix_type']}\n")
            f.write(f"- **CV Strategy:** {summary_stats['best_configuration']['cv_strategy']}\n")
            f.write(f"- **Model:** {summary_stats['best_configuration']['model']}\n")
            f.write(f"- **PCA Config:** {summary_stats['best_configuration']['pca_config']}\n")
            f.write(f"- **RMSE:** {summary_stats['best_configuration']['rmse']:.4f}\n")
            f.write(f"- **R²:** {summary_stats['best_configuration']['r2']:.4f}\n")
            
            # Add solvent analysis section
            if results_by_solvent:
                f.write("\n## Solvent-Specific Performance\n\n")
                f.write("| Solvent | RMSE | R² | MAE | Count |\n")
                f.write("|---------|------|----|----|-------|\n")
                
                valid_solvents = {k: v for k, v in results_by_solvent.items() if not np.isnan(v['rmse'])}
                sorted_solvents = sorted(valid_solvents.items(), key=lambda x: x[1]['rmse'])
                
                for solvent, metrics in sorted_solvents[:20]:  # Top 20 best performing solvents
                    f.write(f"| {solvent} | {metrics['rmse']:.4f} | "
                           f"{metrics['r2']:.4f} | {metrics['mae']:.4f} | "
                           f"{metrics['count']} |\n")
        
        print(f"Enhanced report saved to: {report_path}")
    
    def create_enhanced_visualizations(self, all_results, results_df):
        """Create enhanced visualization suite with MoE and OpenCOSMO analysis"""
        print(f"\nCreating enhanced visualizations...")
        
        viz_dir = self.output_dir / "enhanced_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model comparison with OpenCOSMO breakdown
        self.plot_enhanced_model_comparison(results_df, viz_dir)
        
        # 2. MoE vs Standard model comparison
        self.plot_moe_comparison(results_df, viz_dir)
        
        # 3. OpenCOSMO impact visualization
        self.plot_opencosmo_impact(results_df, viz_dir)
        
        # 4. CV Strategy comparison
        self.plot_cv_strategy_comparison(results_df, viz_dir)
        
        # 5. Performance distribution
        self.plot_performance_distribution(results_df, viz_dir)
        
        # 6. Enhanced predictions vs actual
        self.plot_enhanced_predictions_vs_actual(all_results, viz_dir)
        
        print(f"Enhanced visualizations saved to: {viz_dir}")
    
    def plot_enhanced_model_comparison(self, results_df, viz_dir):
        """Enhanced model comparison with OpenCOSMO breakdown"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE by model and matrix type
        sns.boxplot(data=results_df, x='model', y='metric_rmse', hue='matrix_type', ax=axes[0,0])
        axes[0,0].set_title('RMSE by Model and Matrix Type')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend(title='Matrix Type')
        
        # R² by model and matrix type
        sns.boxplot(data=results_df, x='model', y='metric_r2', hue='matrix_type', ax=axes[0,1])
        axes[0,1].set_title('R² by Model and Matrix Type')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(title='Matrix Type')
        
        # MAE by model and matrix type
        sns.boxplot(data=results_df, x='model', y='metric_mae', hue='matrix_type', ax=axes[1,0])
        axes[1,0].set_title('MAE by Model and Matrix Type')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Matrix Type')
        
        # Performance scatter plot
        sns.scatterplot(data=results_df, x='metric_rmse', y='metric_r2', 
                       hue='model', style='matrix_type', s=100, ax=axes[1,1])
        axes[1,1].set_title('RMSE vs R² Performance')
        axes[1,1].set_xlabel('RMSE')
        axes[1,1].set_ylabel('R²')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "enhanced_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_moe_comparison(self, results_df, viz_dir):
        """Compare MoE vs standard models"""
        # Separate MoE and standard models
        moe_results = results_df[results_df['model'].str.contains('moe', case=False)]
        standard_results = results_df[~results_df['model'].str.contains('moe', case=False)]
        
        if len(moe_results) > 0 and len(standard_results) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # RMSE comparison
            moe_rmse = moe_results['metric_rmse'].values
            std_rmse = standard_results['metric_rmse'].values
            
            axes[0].boxplot([std_rmse, moe_rmse], labels=['Standard', 'MoE'])
            axes[0].set_title('RMSE: Standard vs MoE')
            axes[0].set_ylabel('RMSE')
            
            # R² comparison
            moe_r2 = moe_results['metric_r2'].values
            std_r2 = standard_results['metric_r2'].values
            
            axes[1].boxplot([std_r2, moe_r2], labels=['Standard', 'MoE'])
            axes[1].set_title('R²: Standard vs MoE')
            axes[1].set_ylabel('R²')
            
            # Scatter plot comparison
            axes[2].scatter(std_rmse, std_r2, label='Standard', alpha=0.7, s=60)
            axes[2].scatter(moe_rmse, moe_r2, label='MoE', alpha=0.7, s=60)
            axes[2].set_xlabel('RMSE')
            axes[2].set_ylabel('R²')
            axes[2].set_title('Performance Comparison')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "moe_vs_standard_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_opencosmo_impact(self, results_df, viz_dir):
        """Visualize OpenCOSMO feature impact"""
        if 'matrix_type' in results_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # RMSE impact
            cosmo_pivot_rmse = results_df.pivot_table(
                values='metric_rmse', 
                index='model', 
                columns='matrix_type', 
                aggfunc='mean'
            )
            
            cosmo_pivot_rmse.plot(kind='bar', ax=axes[0])
            axes[0].set_title('OpenCOSMO Impact on RMSE')
            axes[0].set_ylabel('RMSE')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].legend(title='Matrix Type')
            
            # R² impact
            cosmo_pivot_r2 = results_df.pivot_table(
                values='metric_r2', 
                index='model', 
                columns='matrix_type', 
                aggfunc='mean'
            )
            
            cosmo_pivot_r2.plot(kind='bar', ax=axes[1])
            axes[1].set_title('OpenCOSMO Impact on R²')
            axes[1].set_ylabel('R²')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend(title='Matrix Type')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "opencosmo_impact.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_cv_strategy_comparison(self, results_df, viz_dir):
        """Compare different CV strategies"""
        if 'cv_strategy' in results_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # RMSE by CV strategy
            sns.boxplot(data=results_df, x='cv_strategy', y='metric_rmse', ax=axes[0])
            axes[0].set_title('RMSE by CV Strategy')
            axes[0].tick_params(axis='x', rotation=45)
            
            # R² by CV strategy
            sns.boxplot(data=results_df, x='cv_strategy', y='metric_r2', ax=axes[1])
            axes[1].set_title('R² by CV Strategy')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "cv_strategy_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_performance_distribution(self, results_df, viz_dir):
        """Plot performance metric distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE distribution
        axes[0,0].hist(results_df['metric_rmse'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('RMSE Distribution')
        axes[0,0].set_xlabel('RMSE')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(results_df['metric_rmse'].mean(), color='red', linestyle='--', label='Mean')
        axes[0,0].legend()
        
        # R² distribution
        axes[0,1].hist(results_df['metric_r2'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('R² Distribution')
        axes[0,1].set_xlabel('R²')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(results_df['metric_r2'].mean(), color='red', linestyle='--', label='Mean')
        axes[0,1].legend()
        
        # MAE distribution
        axes[1,0].hist(results_df['metric_mae'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('MAE Distribution')
        axes[1,0].set_xlabel('MAE')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(results_df['metric_mae'].mean(), color='red', linestyle='--', label='Mean')
        axes[1,0].legend()
        
        # Performance correlation
        axes[1,1].scatter(results_df['metric_rmse'], results_df['metric_r2'], alpha=0.6)
        axes[1,1].set_xlabel('RMSE')
        axes[1,1].set_ylabel('R²')
        axes[1,1].set_title('RMSE vs R² Correlation')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = results_df['metric_rmse'].corr(results_df['metric_r2'])
        axes[1,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                      transform=axes[1,1].transAxes, 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_enhanced_predictions_vs_actual(self, all_results, viz_dir):
        """Enhanced predictions vs actual with model breakdown"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Combined predictions vs actual
        all_true = []
        all_pred = []
        all_models = []
        
        for result in all_results:
            all_true.extend(result['true_values'])
            all_pred.extend(result['predictions'])
            all_models.extend([result['model']] * len(result['true_values']))
        
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        
        # Left plot: All predictions
        axes[0].scatter(all_true, all_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predictions vs True Values (All Models)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        overall_r2 = r2(all_true, all_pred)
        overall_rmse = rmse(all_true, all_pred)
        axes[0].text(0.05, 0.95, f'R² = {overall_r2:.3f}\nRMSE = {overall_rmse:.3f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Right plot: Residuals
        residuals = all_pred - all_true
        axes[1].scatter(all_true, residuals, alpha=0.6, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', label='Perfect Prediction')
        axes[1].set_xlabel('True Values')
        axes[1].set_ylabel('Residuals (Predicted - True)')
        axes[1].set_title('Residual Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add residual statistics
        residual_std = np.std(residuals)
        axes[1].text(0.05, 0.95, f'Residual Std: {residual_std:.3f}', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "enhanced_predictions_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_moe_routing(self, moe_model, X_test, y_test):
        """Analyze how well the MoE model routes samples to appropriate experts"""
        if not isinstance(moe_model, MixtureOfExpertsRegressor):
            return None
        
        try:
            # Get expert assignments and predictions
            routing_info = moe_model.predict_expert_assignment(X_test)
            assignments = routing_info['assignments']
            confidences = routing_info['confidence']
            
            # Get predictions from each method
            hard_preds = moe_model.predict(X_test)  # Hard routing
            soft_preds = moe_model.predict_soft_routing(X_test)  # Soft routing
            
            # Analyze routing accuracy based on true solubility
            true_labels = (y_test > moe_model.threshold).astype(int)
            routing_accuracy = np.mean(assignments == true_labels)
            
            # Performance by expert
            low_mask = assignments == 0
            high_mask = assignments == 1
            
            results = {
                'routing_accuracy': routing_accuracy,
                'avg_confidence': np.mean(confidences),
                'hard_routing_rmse': rmse(y_test, hard_preds),
                'soft_routing_rmse': rmse(y_test, soft_preds),
                'hard_routing_r2': r2(y_test, hard_preds),
                'soft_routing_r2': r2(y_test, soft_preds),
            }
            
            # Expert-specific performance
            if np.any(low_mask):
                results['low_expert_rmse'] = rmse(y_test[low_mask], hard_preds[low_mask])
                results['low_expert_r2'] = r2(y_test[low_mask], hard_preds[low_mask])
                results['low_expert_count'] = low_mask.sum()
            
            if np.any(high_mask):
                results['high_expert_rmse'] = rmse(y_test[high_mask], hard_preds[high_mask])
                results['high_expert_r2'] = r2(y_test[high_mask], hard_preds[high_mask])
                results['high_expert_count'] = high_mask.sum()
            
            return results
            
        except Exception as e:
            print(f"Error analyzing MoE routing: {e}")
            return None
