import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


class EnhancedVisualizer:
    """Create publication-quality visualizations for solubility predictions"""
    
    def __init__(self, config):
        self.config = config
        self._set_plotting_style()
    
    def _set_plotting_style(self):
        """Set publication-quality plotting parameters"""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def create_all_visualizations(self, all_results, output_dir, df=None):
        """Create comprehensive visualization suite"""
        viz_dir = output_dir / "enhanced_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        print("\nCreating enhanced visualizations...")
        
        # 1. Master prediction plot
        self.create_master_prediction_plot(all_results, viz_dir)
        
        # 2. Model performance comparison
        self.create_model_comparison_plot(all_results, viz_dir)
        
        # 3. Feature importance visualization
        self.create_feature_importance_plot(all_results, viz_dir)
        
        # 4. Solvent-specific analysis
        self.create_solvent_analysis_plots(all_results, viz_dir, df)
        
        # 5. Error distribution analysis
        self.create_error_analysis_plots(all_results, viz_dir)
        
        # 6. Cross-validation performance
        self.create_cv_performance_plot(all_results, viz_dir)
        
        # 7. Model complexity vs performance
        self.create_complexity_analysis_plot(all_results, viz_dir)
        
        print(f"Enhanced visualizations saved to: {viz_dir}")
    
    def create_master_prediction_plot(self, all_results, viz_dir):
        """Create comprehensive prediction analysis plot"""
        best_result = min(all_results, key=lambda x: x['metrics']['rmse'])
        
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1], 
                               hspace=0.3, wspace=0.3)
        
        y_true = best_result['true_values']
        y_pred = best_result['predictions']
        residuals = y_true - y_pred
        
        # 1. Main hexbin plot
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Create density hexbin
        hb = ax1.hexbin(y_true, y_pred, gridsize=40, cmap='YlOrRd', mincnt=1, alpha=0.8)
        cb = plt.colorbar(hb, ax=ax1)
        cb.set_label('Sample Density', rotation=270, labelpad=20)
        
        # Add reference lines
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax1.plot(lims, lims, 'b-', lw=2, label='Perfect prediction', zorder=10)
        
        # Add error bands
        x_smooth = np.linspace(lims[0], lims[1], 100)
        ax1.fill_between(x_smooth, x_smooth - 0.5, x_smooth + 0.5, 
                        alpha=0.2, color='gray', label='±0.5 log units')
        ax1.fill_between(x_smooth, x_smooth - 1.0, x_smooth + 1.0, 
                        alpha=0.1, color='gray', label='±1.0 log units')
        
        # Metrics text
        metrics_text = (f"R² = {best_result['metrics']['r2']:.4f}\n"
                       f"RMSE = {best_result['metrics']['rmse']:.4f}\n"
                       f"MAE = {best_result['metrics']['mae']:.4f}")
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('True Solubility (log g/100g)')
        ax1.set_ylabel('Predicted Solubility (log g/100g)')
        ax1.set_title(f"Best Model: {best_result['model'].upper()} - Predictions vs Actual")
        ax1.legend(loc='lower right')
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. Residual plot
        ax2 = fig.add_subplot(gs[0, 2])
        scatter = ax2.scatter(y_pred, residuals, c=np.abs(residuals), 
                            cmap='RdYlBu_r', alpha=0.6, s=30)
        ax2.axhline(y=0, color='black', linestyle='-', lw=1.5)
        ax2.axhline(y=residuals.std(), color='red', linestyle='--', lw=1, alpha=0.7)
        ax2.axhline(y=-residuals.std(), color='red', linestyle='--', lw=1, alpha=0.7)
        
        ax2.set_xlabel('Predicted Solubility')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Analysis')
        
        # 3. Residual histogram with normal fit
        ax3 = fig.add_subplot(gs[1, 2])
        n, bins, patches = ax3.hist(residuals, bins=30, density=True, 
                                   alpha=0.7, color='skyblue', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
                label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Density')
        ax3.set_title('Residual Distribution')
        ax3.legend()
        
        # 4. Q-Q plot
        ax4 = fig.add_subplot(gs[2, 0])
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Normal Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        # 5. Error by magnitude
        ax5 = fig.add_subplot(gs[2, 1])
        abs_error = np.abs(residuals)
        bins = np.percentile(y_true, np.linspace(0, 100, 6))
        bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        bin_errors = []
        
        for i in range(len(bins)-1):
            mask = (y_true >= bins[i]) & (y_true < bins[i+1])
            bin_errors.append(abs_error[mask].mean() if mask.any() else 0)
        
        ax5.bar(range(len(bin_centers)), bin_errors, color='coral', edgecolor='black')
        ax5.set_xticks(range(len(bin_centers)))
        ax5.set_xticklabels([f'{bc:.1f}' for bc in bin_centers], rotation=45)
        ax5.set_xlabel('Solubility Bins (log g/100g)')
        ax5.set_ylabel('Mean Absolute Error')
        ax5.set_title('Error by Solubility Range')
        
        # 6. Outlier analysis
        ax6 = fig.add_subplot(gs[2, 2])
        outlier_threshold = 2 * residuals.std()
        outliers = np.abs(residuals) > outlier_threshold
        
        ax6.scatter(y_true[~outliers], y_pred[~outliers], alpha=0.5, s=20, 
                   label=f'Normal ({(~outliers).sum()})')
        ax6.scatter(y_true[outliers], y_pred[outliers], color='red', s=40, 
                   marker='^', label=f'Outliers ({outliers.sum()})')
        ax6.plot(lims, lims, 'b--', lw=1, alpha=0.5)
        
        ax6.set_xlabel('True Solubility')
        ax6.set_ylabel('Predicted Solubility')
        ax6.set_title(f'Outlier Detection (|residual| > {outlier_threshold:.2f})')
        ax6.legend()
        
        plt.suptitle('Comprehensive Prediction Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(viz_dir / "master_prediction_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_comparison_plot(self, all_results, viz_dir):
        """Create model performance comparison plots"""
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                'model': r['model'],
                'cv_strategy': r.get('cv_strategy', 'standard'),
                'rmse': r['metrics']['rmse'],
                'r2': r['metrics']['r2'],
                'mae': r['metrics']['mae'],
                'mape': r['metrics'].get('mape', np.nan)
            }
            for r in all_results
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # 1. Box plots for each metric
        metrics = ['rmse', 'r2', 'mae', 'mape']
        titles = ['RMSE (Lower is Better)', 'R² (Higher is Better)', 
                  'MAE (Lower is Better)', 'MAPE (Lower is Better)']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            if metric in results_df.columns and not results_df[metric].isna().all():
                data_by_model = [results_df[results_df['model'] == model][metric].values 
                               for model in results_df['model'].unique()]
                
                bp = ax.boxplot(data_by_model, labels=results_df['model'].unique(), patch_artist=True)
                
                # Color boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_title(title)
                ax.set_ylabel(metric.upper())
                ax.grid(True, alpha=0.3)
                
                # Add mean values
                for i, (data, model) in enumerate(zip(data_by_model, results_df['model'].unique())):
                    if len(data) > 0:
                        mean_val = np.mean(data)
                        ax.text(i+1, mean_val, f'{mean_val:.3f}', ha='center', 
                               va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_comparison_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance heatmap
        pivot_table = results_df.pivot_table(
            values=['rmse', 'r2'], 
            index='model', 
            aggfunc=['mean', 'std']
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Mean performance heatmap
        mean_data = pivot_table['mean']
        sns.heatmap(mean_data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                   center=0.5, ax=ax1, cbar_kws={'label': 'Value'})
        ax1.set_title('Mean Performance by Model')
        
        # Std performance heatmap
        std_data = pivot_table['std']
        sns.heatmap(std_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax2, cbar_kws={'label': 'Std Dev'})
        ax2.set_title('Performance Variability by Model')
        
        plt.suptitle('Model Performance Heatmaps', fontsize=14)
        plt.tight_layout()
        plt.savefig(viz_dir / "model_performance_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_importance_plot(self, all_results, viz_dir):
        """Create feature importance visualizations"""
        # Extract feature importance from results
        importance_data = []
        
        for result in all_results:
            if 'feature_importance' in result and result['feature_importance'] is not None:
                imp = result['feature_importance']
                if 'top_features' in imp:
                    for feat in imp['top_features'][:20]:  # Top 20 features
                        importance_data.append({
                            'model': result['model'],
                            'feature': feat['name'],
                            'importance': feat['importance']
                        })
        
        if not importance_data:
            print("No feature importance data available")
            return
        
        importance_df = pd.DataFrame(importance_data)
        
        # Aggregate importance across models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Top features by average importance
        avg_importance = importance_df.groupby('feature')['importance'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(20)
        
        y_pos = np.arange(len(avg_importance))
        ax1.barh(y_pos, avg_importance['mean'], xerr=avg_importance['std'], 
                color='steelblue', alpha=0.8, capsize=5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(avg_importance.index, fontsize=9)
        ax1.set_xlabel('Average Importance')
        ax1.set_title('Top 20 Most Important Features')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # 2. Feature importance by model
        pivot_importance = importance_df.pivot_table(
            values='importance', 
            index='feature', 
            columns='model', 
            aggfunc='mean'
        ).fillna(0)
        
        # Select top features
        top_features = pivot_importance.sum(axis=1).nlargest(15).index
        plot_data = pivot_importance.loc[top_features]
        
        sns.heatmap(plot_data, cmap='YlOrRd', annot=True, fmt='.3f', 
                   ax=ax2, cbar_kws={'label': 'Importance'})
        ax2.set_title('Feature Importance by Model')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Feature')
        
        plt.suptitle('Feature Importance Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(viz_dir / "feature_importance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_solvent_analysis_plots(self, all_results, viz_dir, df):
        """Create solvent-specific performance visualizations"""
        if df is None:
            return
        
        # Analyze by solvent
        solvent_performance = {}
        
        for result in all_results:
            test_indices = result['test_indices']
            y_true = result['true_values']
            y_pred = result['predictions']
            
            # Get solvent names
            test_solvents = df.iloc[test_indices]['solvent_name'].values
            
            for i, solvent in enumerate(test_solvents):
                if solvent not in solvent_performance:
                    solvent_performance[solvent] = {'true': [], 'pred': []}
                
                solvent_performance[solvent]['true'].append(y_true[i])
                solvent_performance[solvent]['pred'].append(y_pred[i])
        
        # Calculate metrics by solvent
        solvent_metrics = []
        
        for solvent, data in solvent_performance.items():
            if len(data['true']) >= 5:  # Need sufficient data
                true_vals = np.array(data['true'])
                pred_vals = np.array(data['pred'])
                
                try:
                    r2_val = stats.pearsonr(true_vals, pred_vals)[0]**2
                    rmse_val = np.sqrt(np.mean((true_vals - pred_vals)**2))
                    
                    solvent_metrics.append({
                        'solvent': solvent,
                        'n_samples': len(true_vals),
                        'rmse': rmse_val,
                        'r2': r2_val,
                        'mean_true': true_vals.mean(),
                        'std_true': true_vals.std()
                    })
                except:
                    pass
        
        if not solvent_metrics:
            return
        
        metrics_df = pd.DataFrame(solvent_metrics).sort_values('rmse')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Solvent-Specific Performance Analysis', fontsize=16)
        
        # 1. RMSE by solvent
        ax1 = axes[0, 0]
        colors = plt.cm.RdYlGn_r(metrics_df['rmse'] / metrics_df['rmse'].max())
        bars1 = ax1.bar(range(len(metrics_df)), metrics_df['rmse'], color=colors)
        ax1.set_xticks(range(len(metrics_df)))
        ax1.set_xticklabels(metrics_df['solvent'], rotation=90, ha='right')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE by Solvent (sorted)')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add sample count annotations
        for i, (bar, n) in enumerate(zip(bars1, metrics_df['n_samples'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'n={n}', ha='center', va='bottom', fontsize=8)
        
        # 2. R² by solvent
        ax2 = axes[0, 1]
        colors = plt.cm.RdYlGn(metrics_df['r2'])
        bars2 = ax2.bar(range(len(metrics_df)), metrics_df['r2'], color=colors)
        ax2.set_xticks(range(len(metrics_df)))
        ax2.set_xticklabels(metrics_df['solvent'], rotation=90, ha='right')
        ax2.set_ylabel('R²')
        ax2.set_title('R² by Solvent')
        ax2.set_ylim(0, 1)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 3. Scatter: RMSE vs Data variability
        ax3 = axes[1, 0]
        scatter = ax3.scatter(metrics_df['std_true'], metrics_df['rmse'], 
                            s=metrics_df['n_samples']*5, alpha=0.6, c=metrics_df['r2'], 
                            cmap='viridis')
        ax3.set_xlabel('Solubility Standard Deviation')
        ax3.set_ylabel('RMSE')
        ax3.set_title('RMSE vs Data Variability (size = n_samples)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('R²', rotation=270, labelpad=15)
        
        # Add trend line
        z = np.polyfit(metrics_df['std_true'], metrics_df['rmse'], 1)
        p = np.poly1d(z)
        ax3.plot(metrics_df['std_true'].sort_values(), 
                p(metrics_df['std_true'].sort_values()), "r--", alpha=0.8)
        
        # 4. Top/Bottom performers
        ax4 = axes[1, 1]
        top_n = 10
        top_solvents = metrics_df.nsmallest(top_n, 'rmse')
        bottom_solvents = metrics_df.nlargest(top_n, 'rmse')
        
        y_pos = np.arange(top_n)
        width = 0.35
        
        ax4.barh(y_pos - width/2, top_solvents['rmse'], width, 
                label='Best', color='green', alpha=0.7)
        ax4.barh(y_pos + width/2, bottom_solvents['rmse'], width, 
                label='Worst', color='red', alpha=0.7)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"{s[:15]}..." if len(s) > 15 else s 
                           for s in top_solvents['solvent']], fontsize=9)
        ax4.set_xlabel('RMSE')
        ax4.set_title(f'Top {top_n} Best Performing Solvents')
        ax4.legend()
        ax4.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "solvent_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_error_analysis_plots(self, all_results, viz_dir):
        """Create detailed error analysis plots"""
        # Collect all errors
        all_errors = []
        all_true = []
        all_pred = []
        
        for result in all_results:
            errors = result['true_values'] - result['predictions']
            all_errors.extend(errors)
            all_true.extend(result['true_values'])
            all_pred.extend(result['predictions'])
        
        all_errors = np.array(all_errors)
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        abs_errors = np.abs(all_errors)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Comprehensive Error Analysis', fontsize=16)
        
        # 1. Error distribution with KDE
        ax1 = axes[0, 0]
        ax1.hist(all_errors, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(all_errors)
        x_range = np.linspace(all_errors.min(), all_errors.max(), 200)
        ax1.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        
        ax1.set_xlabel('Error (True - Predicted)')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative error distribution
        ax2 = axes[0, 1]
        sorted_abs_errors = np.sort(abs_errors)
        cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
        
        ax2.plot(sorted_abs_errors, cumulative, 'b-', lw=2)
        ax2.fill_between(sorted_abs_errors, 0, cumulative, alpha=0.3)
        
        # Add percentile markers
        percentiles = [50, 75, 90, 95]
        for p in percentiles:
            val = np.percentile(abs_errors, p)
            ax2.axvline(val, color='red', linestyle='--', alpha=0.5)
            ax2.text(val, 0.05, f'{p}%\n{val:.2f}', ha='center', fontsize=9)
        
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Error vs prediction magnitude
        ax3 = axes[0, 2]
        hexbin = ax3.hexbin(all_pred, abs_errors, gridsize=30, cmap='YlOrRd', mincnt=1)
        plt.colorbar(hexbin, ax=ax3, label='Count')
        
        # Add trend line
        z = np.polyfit(all_pred, abs_errors, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(all_pred.min(), all_pred.max(), 100)
        ax3.plot(x_smooth, p(x_smooth), 'b-', lw=2, label='Trend')
        
        ax3.set_xlabel('Predicted Value')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Error vs Prediction Magnitude')
        ax3.legend()
        
        # 4. Bland-Altman plot
        ax4 = axes[1, 0]
        mean_values = (all_true + all_pred) / 2
        differences = all_true - all_pred
        
        ax4.scatter(mean_values, differences, alpha=0.5, s=20)
        ax4.axhline(y=0, color='black', linestyle='-', lw=1.5)
        
        # Add limits of agreement
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        ax4.axhline(y=mean_diff, color='red', linestyle='--', lw=1.5, 
                   label=f'Mean: {mean_diff:.3f}')
        ax4.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle=':', lw=1.5, 
                   label=f'±1.96 SD: {1.96*std_diff:.3f}')
        ax4.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle=':', lw=1.5)
        
        ax4.set_xlabel('Mean of True and Predicted')
        ax4.set_ylabel('Difference (True - Predicted)')
        ax4.set_title('Bland-Altman Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Error by data range
        ax5 = axes[1, 1]
        n_bins = 10
        bins = np.percentile(all_true, np.linspace(0, 100, n_bins + 1))
        bin_centers = [(bins[i] + bins[i+1])/2 for i in range(n_bins)]
        bin_errors = []
        bin_stds = []
        
        for i in range(n_bins):
            mask = (all_true >= bins[i]) & (all_true < bins[i+1])
            if mask.any():
                bin_errors.append(abs_errors[mask].mean())
                bin_stds.append(abs_errors[mask].std())
            else:
                bin_errors.append(0)
                bin_stds.append(0)
        
        ax5.bar(range(n_bins), bin_errors, yerr=bin_stds, capsize=5, 
               color='coral', edgecolor='black', alpha=0.7)
        ax5.set_xticks(range(n_bins))
        ax5.set_xticklabels([f'{bc:.1f}' for bc in bin_centers], rotation=45)
        ax5.set_xlabel('Solubility Range (log g/100g)')
        ax5.set_ylabel('Mean Absolute Error')
        ax5.set_title('Error by Solubility Range')
        ax5.grid(True, axis='y', alpha=0.3)
        
        # 6. Error statistics summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""Error Statistics Summary
        
Mean Error: {np.mean(all_errors):.4f}
Std Error: {np.std(all_errors):.4f}
Mean Abs Error: {np.mean(abs_errors):.4f}
Median Abs Error: {np.median(abs_errors):.4f}

Percentiles:
  25th: {np.percentile(abs_errors, 25):.4f}
  50th: {np.percentile(abs_errors, 50):.4f}
  75th: {np.percentile(abs_errors, 75):.4f}
  90th: {np.percentile(abs_errors, 90):.4f}
  95th: {np.percentile(abs_errors, 95):.4f}
  99th: {np.percentile(abs_errors, 99):.4f}

Within ±0.5 log: {(abs_errors <= 0.5).sum() / len(abs_errors) * 100:.1f}%
Within ±1.0 log: {(abs_errors <= 1.0).sum() / len(abs_errors) * 100:.1f}%
"""
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "error_analysis_comprehensive.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_cv_performance_plot(self, all_results, viz_dir):
        """Create cross-validation performance visualization"""
        # Group results by fold
        fold_performance = {}
        
        for result in all_results:
            fold = result.get('fold', 0)
            model = result['model']
            
            if fold not in fold_performance:
                fold_performance[fold] = {}
            
            if model not in fold_performance[fold]:
                fold_performance[fold][model] = {
                    'rmse': result['metrics']['rmse'],
                    'r2': result['metrics']['r2']
                }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot RMSE across folds
        for model in set(r['model'] for r in all_results):
            folds = []
            rmses = []
            
            for fold in sorted(fold_performance.keys()):
                if model in fold_performance[fold]:
                    folds.append(fold)
                    rmses.append(fold_performance[fold][model]['rmse'])
            
            ax1.plot(folds, rmses, 'o-', label=model, markersize=8)
        
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE Across Cross-Validation Folds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot R² across folds
        for model in set(r['model'] for r in all_results):
            folds = []
            r2s = []
            
            for fold in sorted(fold_performance.keys()):
                if model in fold_performance[fold]:
                    folds.append(fold)
                    r2s.append(fold_performance[fold][model]['r2'])
            
            ax2.plot(folds, r2s, 'o-', label=model, markersize=8)
        
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('R²')
        ax2.set_title('R² Across Cross-Validation Folds')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Cross-Validation Performance Stability', fontsize=14)
        plt.tight_layout()
        plt.savefig(viz_dir / "cv_performance_stability.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_complexity_analysis_plot(self, all_results, viz_dir):
        """Create model complexity vs performance plots"""
        complexity_data = []
        
        for result in all_results:
            if 'metrics' in result:
                metrics = result['metrics']
                
                # Extract complexity metrics if available
                complexity_data.append({
                    'model': result['model'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'aic': metrics.get('aic', np.nan),
                    'bic': metrics.get('bic', np.nan),
                    'n_parameters': metrics.get('n_parameters', np.nan),
                    'complexity_penalty': metrics.get('complexity_penalty', np.nan)
                })
        
        if not complexity_data:
            return
        
        df = pd.DataFrame(complexity_data)
        
        # Remove rows with all NaN complexity metrics
        df = df.dropna(subset=['aic', 'bic', 'n_parameters'], how='all')
        
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Complexity Analysis', fontsize=16)
        
        # 1. AIC vs Performance
        ax1 = axes[0, 0]
        if not df['aic'].isna().all():
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                ax1.scatter(model_data['aic'], model_data['rmse'], 
                          label=model, s=100, alpha=0.7)
            
            ax1.set_xlabel('AIC (Lower is Better)')
            ax1.set_ylabel('RMSE (Lower is Better)')
            ax1.set_title('AIC vs RMSE')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. BIC vs Performance
        ax2 = axes[0, 1]
        if not df['bic'].isna().all():
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                ax2.scatter(model_data['bic'], model_data['r2'], 
                          label=model, s=100, alpha=0.7)
            
            ax2.set_xlabel('BIC (Lower is Better)')
            ax2.set_ylabel('R² (Higher is Better)')
            ax2.set_title('BIC vs R²')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Parameter count vs Performance
        ax3 = axes[1, 0]
        if not df['n_parameters'].isna().all():
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                ax3.scatter(model_data['n_parameters'], model_data['rmse'], 
                          label=model, s=100, alpha=0.7)
            
            ax3.set_xlabel('Number of Parameters')
            ax3.set_ylabel('RMSE')
            ax3.set_title('Model Complexity vs Performance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add log scale if parameter range is large
            if df['n_parameters'].max() / df['n_parameters'].min() > 100:
                ax3.set_xscale('log')
        
        # 4. Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary statistics
        summary = df.groupby('model').agg({
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'n_parameters': 'mean'
        }).round(3)
        
        # Create table
        table_data = []
        for model in summary.index:
            row = [
                model,
                f"{summary.loc[model, ('rmse', 'mean')]:.3f} ± {summary.loc[model, ('rmse', 'std')]:.3f}",
                f"{summary.loc[model, ('r2', 'mean')]:.3f} ± {summary.loc[model, ('r2', 'std')]:.3f}",
                f"{summary.loc[model, ('n_parameters', 'mean')]:.0f}" if not np.isnan(summary.loc[model, ('n_parameters', 'mean')]) else "N/A"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Model', 'RMSE', 'R²', 'Parameters'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax4.set_title('Model Summary Statistics', pad=20)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_complexity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()