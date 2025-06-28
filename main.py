from pathlib import Path
import time
from modules.config import Config
from modules.analysis import SolubilityAnalyzer
import os
from modules.reporting.testing import Reporter

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['XGB_DEVICE'] = 'GPU'
try:
    import cupy as cp
    from cuml.decomposition import PCA as CuPCA
    from cuml.preprocessing import StandardScaler as CuStandardScaler
    from cuml.manifold import TSNE as CuTSNE
    from cuml.manifold import UMAP as CuUMAP
    USE_CUML = True
except ImportError:
    USE_CUML = False
    cp = None
    
def main():
    """Main execution function"""
    # ENHANCED configuration for MAXIMUM PERFORMANCE with advanced MoE
    config = Config(
        tsv_path="data.tsv",
        use_dscribe=True,  # ENABLE DScribe - critical for top performance!
        dscribe_descriptors=["soap", 
                             "sine_matrix",
                             #"mbtr"
                             ],  # Use comprehensive DScribe descriptors
        
        # Enhanced SOAP configurations - multiple scales for better coverage
        soap_configs=[
             #{"r_cut": 6.0, "n_max": 8, "l_max": 6, "permutation": "none"},   # Primary config
             #{"r_cut": 4.0, "n_max": 6, "l_max": 4, "permutation": "none"},   # Local features
             {"r_cut": 8.0, "n_max": 10, "l_max": 8}, # Long-range (commented for speed)
        ],
        
        # Multiple Sine Matrix permutations for robustness
        sine_permutations=[
            "sorted_l2", 
            "eigenspectrum"
            ],
        
        # Mixture of Experts enabled with better threshold
        use_mixture_of_experts=False,
        moe_threshold_method="solubility_cutoff",  # Use natural solubility threshold (logS = 0)
        
        models=["xgboost", 
                #"neural_network",
                #"random_forest", 
                #"ridge"
                ],  # Add random_forest back, remove lasso for now
        fp_configs=[
            #{"type": "morgan", "n_bits": 4096, "radius": 3},  # Standard Morgan
            {"type": "combined", "n_bits": 4096, "radius": 3}, # Combined fingerprints
            {"type": "rdkit_descriptors"},  # RDKit descriptors
        ],
        pca_configs=[
            {"use_pca": False},  # No PCA for direct feature interpretation
           #{"use_pca": True, "n_components": 500}  # High-dimensional PCA for noise reduction
        ],
        outer_folds=5,  # Standard 10-fold CV
        inner_folds=5,  # Reduced inner folds for speed
        search_iterations=6,  # More iterations for better hyperparameter search
        create_visualizations=True,  # Enable enhanced visualizations
        use_gpu=USE_CUML,  # Maximum GPU acceleration
        use_open_cosmo=True,  # Use OpenCOSMO features - critical for performance!
        test_without_open_cosmo=False,  # Focus on WITH OpenCOSMO first
        use_drip_cv=False,  # Disable for this optimization run
        drip_strategies=["standard_cv"],  # Focus on standard CV
        n_jobs=32,  # Use maximum CPU cores for parallel processing
        verbose=1,
        
        # Enhanced Regularization Settings - Lighter for better performance
        regularization_strength="light",  # Reduced from medium - less aggressive
        enable_regularized_models=True,  # Enable Ridge, Lasso, ElasticNet
        use_feature_selection=True,  # Disable feature selection initially to isolate issues
        feature_selection_methods=["variance","elastic_net","bayesian_ridge"],  # Simplified
        variance_threshold=0.001  # Lower threshold
    )
    
    
 # Print configuration details
    
    # Check if dataset exists
    if not Path(config.tsv_path).exists():
        print(f"ERROR: Dataset not found: {config.tsv_path}")
        return
    
    # Initialize analyzer
    analyzer = SolubilityAnalyzer(config)
    r = Reporter(config, analyzer)
    r.sot
    
    # Run analysis
    start_time = time.time()
    try:
        all_results = analyzer.analysis()
        total_time = time.time() - start_time
        
        r.eot(config, analyzer, all_results, total_time)

        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    main()