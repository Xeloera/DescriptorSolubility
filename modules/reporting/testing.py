from modules.config import Config
from modules.analysis import SolubilityAnalyzer



class Reporter:
    def __init__(self, config: Config, analyzer: SolubilityAnalyzer):
        self.config = config
        self.analyzer = analyzer
    def sot(self):
        print("="*80)
        print("ENHANCED SOLUBILITY WORKBENCH 2025 - ADVANCED OPTIMIZATION")
        print("="*80)
        print(f"Configuration: {self.config.models} models, {self.config.dscribe_descriptors} descriptors")
        print(f"MoE Threshold Method: {self.config.moe_threshold_method}")
        print(f"SOAP configs: {len(self.config.soap_configs)}")
        print(f"Sine permutations: {self.config.sine_permutations}")
        print(f"Fingerprint configs: {len(self.config.fp_configs)}")
        print(f"PCA configs: {len(self.config.pca_configs)}")
        print(f"GPU acceleration: {self.config.use_gpu}")
        print(f"Search iterations: {self.config.search_iterations}")
        print(f"OpenCOSMO testing: {self.config.test_without_open_cosmo}")
        print("="*80)
        
    def eot(self, all_results: list, total_time: float):
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Results directory: {self.analyzer.output_dir}")
        print(f"Configurations tested: {len(all_results)}")
        
        # Print best results
        if all_results:
            best_result = min(all_results, key=lambda x: x['metrics']['rmse'])
            print(f"\nBest configuration:")
            print(f"  Model: {best_result['model']}")
            print(f"  Matrix type: {best_result['matrix_type']}")
            print(f"  CV strategy: {best_result['cv_strategy']}")
            print(f"  PCA: {best_result['pca_config']}")
            print(f"  RMSE: {best_result['metrics']['rmse']:.4f}")
            print(f"  R²: {best_result['metrics']['r2']:.4f}")
            
            # Analyze MoE performance if available
            moe_results = [r for r in all_results if 'moe' in r['model']]
            if moe_results:
                print(f"\nMoE Models tested: {len(moe_results)}")
                best_moe = min(moe_results, key=lambda x: x['metrics']['rmse'])
                print(f"Best MoE performance:")
                print(f"  Model: {best_moe['model']}")
                print(f"  RMSE: {best_moe['metrics']['rmse']:.4f}")
                print(f"  R²: {best_moe['metrics']['r2']:.4f}")
            
            # Analyze regularization impact
            print(f"\n{'='*60}")
            print("REGULARIZATION ANALYSIS")
            print(f"{'='*60}")
            print(f"Regularization strength used: {self.config.regularization_strength}")
            print(f"Regularized models enabled: {self.config.enable_regularized_models}")
            print(f"Feature selection enabled: {self.config.use_feature_selection}")
            if self.config.use_feature_selection:
                print(f"Feature selection methods: {self.config.feature_selection_methods}")
                print(f"Variance threshold: {self.config.variance_threshold}")
            
            # Compare regularized vs non-regularized models
            tree_models = [r for r in all_results if r['model'] in ['xgboost', 'lightgbm', 'random_forest']]
            linear_models = [r for r in all_results if r['model'] in ['ridge', 'lasso', 'elastic_net', 'svr', 'bayesian_ridge']]
            
            if tree_models:
                best_tree = min(tree_models, key=lambda x: x['metrics']['rmse'])
                print(f"\nBest tree-based model (with regularization):")
                print(f"  Model: {best_tree['model']}")
                print(f"  RMSE: {best_tree['metrics']['rmse']:.4f}")
                print(f"  R²: {best_tree['metrics']['r2']:.4f}")
            
            if linear_models:
                best_linear = min(linear_models, key=lambda x: x['metrics']['rmse'])
                print(f"\nBest linear regularized model:")
                print(f"  Model: {best_linear['model']}")
                print(f"  RMSE: {best_linear['metrics']['rmse']:.4f}")
                print(f"  R²: {best_linear['metrics']['r2']:.4f}")
                
                # Show regularization effectiveness
                print(f"\nRegularization effectiveness:")
                for model_type in ['ridge', 'lasso', 'elastic_net']:
                    model_results = [r for r in all_results if r['model'] == model_type]
                    if model_results:
                        best_reg = min(model_results, key=lambda x: x['metrics']['rmse'])
                        print(f"  {model_type.upper()}: RMSE={best_reg['metrics']['rmse']:.4f}, R²={best_reg['metrics']['r2']:.4f}")
                        
            # Feature selection impact (if available)
            if self.config.use_feature_selection:
                print(f"\nFeature selection impact:")
                print(f"  Methods tested: {', '.join(self.config.feature_selection_methods)}")
                print(f"  Expected benefits: Reduced overfitting, faster training, better generalization")