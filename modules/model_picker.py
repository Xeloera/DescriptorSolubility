from modules.config import Config


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from modules.models.mixtureofexpertsregressor import MixtureOfExpertsRegressor
from modules.models.pca import RobustPCA

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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



class ModelSuite:
    """Comprehensive model comparison suite"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def _get_regularization_multipliers(self):
        """Get regularization parameter multipliers based on strength setting"""
        multipliers = {
            "light": {
                "xgb_reg_lambda": [0.1, 0.5, 1.0],  # Much lighter
                "xgb_reg_alpha": [0, 0.01, 0.05],  # Much lighter
                "lgb_reg_lambda": [0.1, 0.5, 1.0],  # Much lighter
                "lgb_reg_alpha": [0, 0.01, 0.05],  # Much lighter
                "nn_alpha": [0.0001, 0.001, 0.01],
                "linear_alpha": [0.001, 0.01, 0.1, 1.0]  # Much lighter for Ridge/Lasso
            },
            "medium": {
                "xgb_reg_lambda": [1, 3, 5, 10],
                "xgb_reg_alpha": [0, 0.1, 0.5, 1.0],
                "lgb_reg_lambda": [1, 3, 5, 10],
                "lgb_reg_alpha": [0, 0.1, 0.5, 1.0],
                "nn_alpha": [0.0001, 0.001, 0.01, 0.1],
                "linear_alpha": [0.1, 1.0, 10.0, 50.0]
            },
            "strong": {
                "xgb_reg_lambda": [5, 10, 15, 25],
                "xgb_reg_alpha": [0.5, 1.0, 2.0, 5.0],
                "lgb_reg_lambda": [5, 10, 15, 25],
                "lgb_reg_alpha": [0.5, 1.0, 2.0, 5.0],
                "nn_alpha": [0.01, 0.1, 1.0, 10.0],
                "linear_alpha": [10.0, 50.0, 100.0, 500.0]
            },
            "very_strong": {
                "xgb_reg_lambda": [10, 25, 50, 100],
                "xgb_reg_alpha": [1.0, 5.0, 10.0, 25.0],
                "lgb_reg_lambda": [10, 25, 50, 100],
                "lgb_reg_alpha": [1.0, 5.0, 10.0, 25.0],
                "nn_alpha": [0.1, 1.0, 10.0, 100.0],
                "linear_alpha": [100.0, 500.0, 1000.0, 5000.0]
            }
        }
        return multipliers.get(self.config.regularization_strength, multipliers["medium"])
    
    def get_model_configs(self):
        """Get model configurations for hyperparameter search including MoE"""
        configs = {}
        
        # Get regularization multipliers based on strength setting
        reg_multipliers = self._get_regularization_multipliers()
        
        # Base XGBoost configuration with improved defaults - avoid parameter conflicts
        xgb_base_params = {
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist' if self.config.use_gpu else 'hist',
            'n_jobs': self.config.n_jobs,
            'eval_metric': 'rmse',
            'verbosity': 0,  # Reduce output noise
            # Removed seed/random_state to avoid parameter conflicts in XGBoost 3.0+
        }
        
        # Add GPU settings with conservative memory management
        if self.config.use_gpu:
            try:
                # Try to use GPU with very conservative memory settings to avoid CUBLAS errors
                xgb_base_params.update({
                    'gpu_id': 0,
                    'predictor': 'gpu_predictor',
                    'max_bin': 64,  # Very low to reduce memory usage
                    'single_precision_histogram': True,  # Use single precision
                })
            except Exception as e:
                print(f"Warning: GPU setup failed ({e}), falling back to CPU")
                xgb_base_params['tree_method'] = 'hist'
        
        if "xgboost" in self.config.models:
            try:
                configs["xgboost"] = {
                    "model": XGBRegressor(**xgb_base_params),
                    "params": {
                        'model__n_estimators': [300, 500, 800, 1000],  # More estimators
                        'model__learning_rate': [0.01, 0.05, 0.1, 0.15],  # More learning rates
                        'model__max_depth': [4, 6, 8, 10],  # Deeper trees
                        'model__subsample': [0.6, 0.7, 0.8, 0.9],  # Enhanced subsampling for regularization
                        'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],  # Enhanced column sampling
                        'model__colsample_bylevel': [0.6, 0.8, 1.0],  # Additional column regularization
                        'model__colsample_bynode': [0.6, 0.8, 1.0],   # Node-level column regularization
                        'model__reg_lambda': reg_multipliers["xgb_reg_lambda"],  # Dynamic L2 regularization
                        'model__reg_alpha': reg_multipliers["xgb_reg_alpha"],  # Dynamic L1 regularization
                        'model__min_child_weight': [1, 3, 5, 7],  # Minimum sum of instance weight in child
                        'model__gamma': [0, 0.1, 0.2, 0.5],  # Minimum loss reduction for splits
                        'model__max_delta_step': [0, 1, 5, 10],  # Maximum delta step for weight updates
                    }
                }
            except Exception as e:
                print(f"Warning: XGBoost model creation failed ({e}), skipping XGBoost")
                # Try CPU fallback
                xgb_cpu_params = {k: v for k, v in xgb_base_params.items() 
                                if k not in ['gpu_id', 'predictor']}
                xgb_cpu_params['tree_method'] = 'hist'
                try:
                    configs["xgboost"] = {
                        "model": XGBRegressor(**xgb_cpu_params),
                        "params": {
                            'model__n_estimators': [300, 500, 800, 1000],
                            'model__learning_rate': [0.01, 0.05, 0.1, 0.15],
                            'model__max_depth': [4, 6, 8, 10],
                            'model__subsample': [0.6, 0.7, 0.8, 0.9],
                            'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                            'model__colsample_bylevel': [0.6, 0.8, 1.0],
                            'model__colsample_bynode': [0.6, 0.8, 1.0],
                            'model__reg_lambda': reg_multipliers["xgb_reg_lambda"],
                            'model__reg_alpha': reg_multipliers["xgb_reg_alpha"],
                            'model__min_child_weight': [1, 3, 5, 7],
                            'model__gamma': [0, 0.1, 0.2, 0.5],
                            'model__max_delta_step': [0, 1, 5, 10],
                        }
                    }
                    print("Successfully created XGBoost model with CPU fallback")
                except Exception as e2:
                    print(f"Error: Both GPU and CPU XGBoost failed ({e2}), skipping XGBoost entirely")
            
            # Add Mixture of Experts XGBoost if enabled with multiple strategies
            if self.config.use_mixture_of_experts:
                try:
                    # Create base params for MoE models - use CPU to avoid CUBLAS errors
                    xgb_moe_params = {k: v for k, v in xgb_base_params.items() 
                                    if k not in ['gpu_id', 'predictor']}
                    xgb_moe_params['tree_method'] = 'hist'  # Force CPU for MoE to avoid GPU memory issues
                    
                    # Standard MoE with median threshold - no seed to avoid parameter conflicts
                    low_model = XGBRegressor(**xgb_moe_params)
                    high_model = XGBRegressor(**xgb_moe_params)
                
                    moe_model = MixtureOfExpertsRegressor(
                        low_model=low_model,
                        high_model=high_model,
                        threshold_method=self.config.moe_threshold_method,
                        threshold_value=self.config.moe_threshold_value,
                        verbose=False  # Suppress during hyperparameter search
                    )
                    
                    configs["xgboost_moe"] = {
                        "model": moe_model,
                        "params": {
                            'model__low_model__n_estimators': [200, 400, 600],
                            'model__low_model__learning_rate': [0.05, 0.1, 0.15],
                            'model__low_model__max_depth': [4, 6, 8],
                            'model__low_model__reg_lambda': [1, 3, 5, 10],  # L2 regularization
                            'model__low_model__reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
                            'model__low_model__subsample': [0.7, 0.8, 0.9],
                            'model__low_model__colsample_bytree': [0.7, 0.8, 0.9],
                            'model__high_model__n_estimators': [200, 400, 600],
                            'model__high_model__learning_rate': [0.05, 0.1, 0.15],
                            'model__high_model__max_depth': [4, 6, 8],
                            'model__high_model__reg_lambda': [1, 3, 5, 10],  # L2 regularization
                            'model__high_model__reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
                            'model__high_model__subsample': [0.7, 0.8, 0.9],
                            'model__high_model__colsample_bytree': [0.7, 0.8, 0.9],
                        }
                    }
                    
                    # Add IQR-optimized MoE for comparison
                    low_model_iqr = XGBRegressor(**xgb_moe_params)
                    high_model_iqr = XGBRegressor(**xgb_moe_params)
                    
                    moe_model_iqr = MixtureOfExpertsRegressor(
                        low_model=low_model_iqr,
                        high_model=high_model_iqr,
                        threshold_method="iqr_optimized",
                        threshold_value=0.5,
                        verbose=False
                    )
                    
                    configs["xgboost_moe_iqr"] = {
                        "model": moe_model_iqr,
                        "params": {
                            'model__low_model__n_estimators': [200, 400],
                            'model__low_model__learning_rate': [0.05, 0.1],
                            'model__low_model__max_depth': [4, 6],
                            'model__low_model__reg_lambda': [1, 3, 5],
                            'model__low_model__reg_alpha': [0, 0.1, 0.5],
                            'model__low_model__subsample': [0.7, 0.8],
                            'model__high_model__n_estimators': [200, 400],
                            'model__high_model__learning_rate': [0.05, 0.1],
                            'model__high_model__max_depth': [4, 6],
                            'model__high_model__reg_lambda': [1, 3, 5],
                            'model__high_model__reg_alpha': [0, 0.1, 0.5],
                            'model__high_model__subsample': [0.7, 0.8],
                        }
                    }
                except Exception as e:
                    print(f"Warning: MoE model creation failed ({e}), trying CPU fallback")
                    # Try CPU fallback for MoE
                    try:
                        xgb_moe_cpu_params = {k: v for k, v in xgb_base_params.items() 
                                            if k not in ['gpu_id', 'predictor']}
                        xgb_moe_cpu_params['tree_method'] = 'hist'
                        
                        low_model_cpu = XGBRegressor(**xgb_moe_cpu_params)
                        high_model_cpu = XGBRegressor(**xgb_moe_cpu_params)
                        
                        moe_model_cpu = MixtureOfExpertsRegressor(
                            low_model=low_model_cpu,
                            high_model=high_model_cpu,
                            threshold_method=self.config.moe_threshold_method,
                            threshold_value=self.config.moe_threshold_value,
                            verbose=False
                        )
                        
                        configs["xgboost_moe"] = {
                            "model": moe_model_cpu,
                            "params": {
                                'model__low_model__n_estimators': [200, 400],
                                'model__low_model__learning_rate': [0.05, 0.1],
                                'model__low_model__max_depth': [4, 6],
                                'model__low_model__reg_lambda': [1, 3, 5],
                                'model__low_model__reg_alpha': [0, 0.1, 0.5],
                                'model__low_model__subsample': [0.7, 0.8],
                                'model__high_model__n_estimators': [200, 400],
                                'model__high_model__learning_rate': [0.05, 0.1],
                                'model__high_model__max_depth': [4, 6],
                                'model__high_model__reg_lambda': [1, 3, 5],
                                'model__high_model__reg_alpha': [0, 0.1, 0.5],
                                'model__high_model__subsample': [0.7, 0.8],
                            }
                        }
                        print("Successfully created MoE model with CPU fallback")
                    except Exception as e2:
                        print(f"Error: Both GPU and CPU MoE failed ({e2}), skipping MoE entirely")
        
        if "lightgbm" in self.config.models:
            lgb_base_params = {
                'objective': 'regression',
                'device': 'gpu' if self.config.use_gpu else 'cpu',
                'n_jobs': self.config.n_jobs,
                'verbose': -1,
                #'random_state': SEED
            }
            
            configs["lightgbm"] = {
                "model": LGBMRegressor(**lgb_base_params),
                "params": {
                    'model__n_estimators': [200, 400, 600, 800],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'model__max_depth': [4, 6, 8, 10],
                    'model__num_leaves': [31, 63, 127, 255],  # Controls model complexity
                    'model__subsample': [0.6, 0.7, 0.8, 0.9],
                    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                    'model__subsample_freq': [1, 3, 5],  # Frequency of subsampling
                    'model__reg_lambda': reg_multipliers["lgb_reg_lambda"],  # Dynamic L2 regularization
                    'model__reg_alpha': reg_multipliers["lgb_reg_alpha"],  # Dynamic L1 regularization
                    'model__min_child_weight': [0.001, 0.01, 0.1, 1.0],  # Minimum sum of hessian in leaf
                    'model__min_child_samples': [5, 10, 20, 50],  # Minimum samples in leaf
                    'model__min_split_gain': [0.0, 0.1, 0.2, 0.5],  # Minimum gain to split
                    'model__max_bin': [255, 127, 63],  # Maximum number of bins (regularization)
                }
            }
            
            # Add LightGBM MoE
            if self.config.use_mixture_of_experts:
                low_lgb = LGBMRegressor(**lgb_base_params)
                high_lgb = LGBMRegressor(**lgb_base_params)
                
                moe_lgb = MixtureOfExpertsRegressor(
                    low_model=low_lgb,
                    high_model=high_lgb,
                    threshold_method=self.config.moe_threshold_method,
                    threshold_value=self.config.moe_threshold_value,
                    verbose=False  # Suppress during hyperparameter search
                )
                
                configs["lightgbm_moe"] = {
                    "model": moe_lgb,
                    "params": {
                        'model__low_model__n_estimators': [200, 400],
                        'model__low_model__learning_rate': [0.05, 0.1],
                        'model__low_model__max_depth': [4, 6],
                        'model__low_model__reg_lambda': [1, 3, 5],
                        'model__low_model__reg_alpha': [0, 0.1, 0.5],
                        'model__low_model__subsample': [0.7, 0.8],
                        'model__low_model__colsample_bytree': [0.7, 0.8],
                        'model__high_model__n_estimators': [200, 400],
                        'model__high_model__learning_rate': [0.05, 0.1],
                        'model__high_model__max_depth': [4, 6],
                        'model__high_model__reg_lambda': [1, 3, 5],
                        'model__high_model__reg_alpha': [0, 0.1, 0.5],
                        'model__high_model__subsample': [0.7, 0.8],
                        'model__high_model__colsample_bytree': [0.7, 0.8],
                    }
                }
        
        if "random_forest" in self.config.models:
            configs["random_forest"] = {
                "model": RandomForestRegressor(
                    #random_state=SEED,
                    n_jobs=self.config.n_jobs,
                    bootstrap=True,  # Enable bootstrap sampling for regularization
                    oob_score=True   # Out-of-bag scoring for model validation
                ),
                "params": {
                    'model__n_estimators': [100, 200, 300, 500],
                    'model__max_depth': [5, 10, 15, 20, None],
                    'model__min_samples_split': [2, 5, 10, 15, 20],  # Regularization via split control
                    'model__min_samples_leaf': [1, 2, 4, 8, 10],     # Regularization via leaf control
                    'model__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],  # Feature sampling regularization
                    'model__min_weight_fraction_leaf': [0.0, 0.01, 0.05],  # Weighted regularization
                    'model__max_leaf_nodes': [None, 50, 100, 200, 500],  # Tree complexity control
                    'model__min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],  # Split quality threshold
                    'model__ccp_alpha': [0.0, 0.01, 0.05, 0.1],  # Cost complexity pruning (post-pruning)
                }
            }
        
        if "neural_network" in self.config.models:
            configs["neural_network"] = {
                "model": MLPRegressor(
                    #random_state=SEED,
                    max_iter=2000,  # Increased for better convergence
                    early_stopping=True,  # Early stopping regularization
                    validation_fraction=0.1,  # Fraction for early stopping
                    n_iter_no_change=20,  # Patience for early stopping
                    tol=1e-6  # Tolerance for optimization
                ),
                "params": {
                    'model__hidden_layer_sizes': [
                        (100,), (200,), (300,),  # Single layer
                        (100, 50), (200, 100), (300, 150),  # Two layers
                        (200, 100, 50), (300, 150, 75),  # Three layers
                        (100, 100), (150, 150)  # Equal layers
                    ],
                    'model__activation': ['relu', 'tanh', 'logistic'],  # Added logistic
                    'model__alpha': reg_multipliers["nn_alpha"],  # Dynamic L2 regularization
                    'model__learning_rate': ['constant', 'adaptive', 'invscaling'],  # Added invscaling
                    'model__learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                    'model__solver': ['adam', 'lbfgs'],  # Different optimizers
                    'model__batch_size': ['auto', 32, 64, 128],  # Batch size regularization
                    'model__beta_1': [0.9, 0.95, 0.99],  # Adam parameter (momentum)
                    'model__beta_2': [0.999, 0.9999],  # Adam parameter (second moment)
                    'model__epsilon': [1e-8, 1e-7, 1e-6],  # Numerical stability
                }
            }
        
        # Add regularized linear models for comparison and interpretability (if enabled)
        if self.config.enable_regularized_models:
            if "ridge" in self.config.models:
                configs["ridge"] = {
                    "model": Ridge(),
                    "params": {
                        'model__alpha': reg_multipliers["linear_alpha"],  # Dynamic L2 regularization
                        'model__fit_intercept': [True, False],
                        'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag'],
                        'model__max_iter': [1000, 2000, 5000],
                        'model__tol': [1e-3, 1e-4, 1e-5]
                    }
                }
            
            if "lasso" in self.config.models:
                configs["lasso"] = {
                    "model": Lasso(max_iter=2000),
                    "params": {
                        'model__alpha': reg_multipliers["linear_alpha"][:4],  # Dynamic L1 regularization (subset)
                        'model__fit_intercept': [True, False],
                        'model__selection': ['cyclic', 'random'],  # Coordinate descent selection
                        'model__tol': [1e-3, 1e-4, 1e-5],
                        'model__positive': [False, True]  # Constrain coefficients to be positive
                    }
                }
            
            if "elastic_net" in self.config.models:
                configs["elastic_net"] = {
                    "model": ElasticNet(max_iter=2000),
                    "params": {
                        'model__alpha': reg_multipliers["linear_alpha"][:4],  # Dynamic regularization (subset)
                        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # L1 vs L2 ratio
                        'model__fit_intercept': [True, False],
                        'model__selection': ['cyclic', 'random'],
                        'model__tol': [1e-3, 1e-4, 1e-5],
                        'model__positive': [False, True]
                    }
                }
            
            # Support Vector Regression with different kernels and regularization
            if "svr" in self.config.models:
                configs["svr"] = {
                    "model": SVR(),
                    "params": {
                        'model__kernel': ['linear', 'rbf', 'poly'],
                        'model__C': reg_multipliers["linear_alpha"],  # Regularization parameter (inverse)
                        'model__epsilon': [0.01, 0.1, 0.2, 0.5],  # Epsilon-tube
                        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient
                        'model__degree': [2, 3, 4],  # For polynomial kernel
                        'model__coef0': [0.0, 0.1, 1.0],  # Independent term in kernel
                        'model__shrinking': [True, False],  # Use shrinking heuristic
                        'model__tol': [1e-3, 1e-4, 1e-5]
                    }
                }
            
            # Bayesian Ridge Regression with automatic regularization
            if "bayesian_ridge" in self.config.models:
                configs["bayesian_ridge"] = {
                    "model": BayesianRidge(),
                    "params": {
                        'model__alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],  # Gamma prior on alpha
                        'model__alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],  # Gamma prior on alpha
                        'model__lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],  # Gamma prior on lambda
                        'model__lambda_2': [1e-6, 1e-5, 1e-4, 1e-3],  # Gamma prior on lambda
                        'model__fit_intercept': [True, False],
                        'model__n_iter': [300, 500, 1000],
                        'model__tol': [1e-3, 1e-4, 1e-5]
                    }
                }
        
        return configs
    
    def create_pipeline(self, model, pca_config, scaler_type="standard", 
                       feature_selection=None, variance_threshold=0.01):
        """Create a complete pipeline with preprocessing, feature selection, and robust PCA fallback"""
        steps = []
        
        # Variance threshold feature selection (removes constant/low-variance features)
        if feature_selection == "variance" or feature_selection is True:
            from sklearn.feature_selection import VarianceThreshold
            steps.append(('variance_selector', VarianceThreshold(threshold=variance_threshold)))
        
        # Scaler
        if scaler_type == "standard":
            if self.config.use_gpu and USE_CUML:
                steps.append(('scaler', CuStandardScaler()))
            else:
                steps.append(('scaler', StandardScaler()))
        elif scaler_type == "minmax":
            steps.append(('scaler', MinMaxScaler()))
        elif scaler_type == "robust":
            steps.append(('scaler', RobustScaler()))
        
        # Advanced feature selection (after scaling)
        if feature_selection == "univariate":
            from sklearn.feature_selection import SelectKBest, f_regression
            steps.append(('feature_selector', SelectKBest(score_func=f_regression, k='all')))
        elif feature_selection == "lasso_selection":
            from sklearn.feature_selection import SelectFromModel
            from sklearn.linear_model import LassoCV
            steps.append(('lasso_selector', SelectFromModel(
                LassoCV(cv=3, max_iter=1000), threshold='median')))
        elif feature_selection == "rfe":
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor
            steps.append(('rfe_selector', RFE(
                RandomForestRegressor(n_estimators=50, n_jobs=self.config.n_jobs),
                n_features_to_select=0.7)))
        
        # PCA with robust fallback mechanism
        if pca_config.get("use_pca", False):
            n_components = pca_config.get("n_components", 100)
            
            # Use RobustPCA wrapper that automatically falls back to CPU if GPU fails
            steps.append(('pca', RobustPCA(
                n_components=n_components,
                use_gpu=self.config.use_gpu and USE_CUML,
                verbose=self.config.verbose > 0
            )))
        
        # Model
        steps.append(('model', model))
        
        return Pipeline(steps)
