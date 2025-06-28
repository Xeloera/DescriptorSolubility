from modules.config import Config
from modules.dataset import Fingerprint
from modules.model_picker import ModelSuite
from modules.models.mixtureofexpertsregressor import MixtureOfExpertsRegressor
from modules.functions.metrics import (rmse, mae, mape, r2, explained_var)
from modules.dataset import DripFeedingCV
from modules.reporting.results import Results

from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

import pandas as pd

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
    
from sklearn.model_selection import KFold, RandomizedSearchCV



import joblib 
    

class SolubilityAnalyzer:
    """Main analysis engine with comprehensive features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.fingerprint_calc = Fingerprint(config)
        self.model_suite = ModelSuite(config)
        self.results = {}
        self.output_dir = self.config.output_dir
        self.reporting = Results(config)
        
        print(f"Results will be saved to: {self.output_dir}")
    
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.config.tsv_path, sep='\t', low_memory=False)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target column: {self.config.target_col}")
        print(f"Unique solvents (by name): {self.df[self.config.solvent_name_col].nunique()}")
        print(f"Unique solvents (by SMILES): {self.df[self.config.solvent_col].nunique()}")
        print(f"Target statistics:\n{self.df[self.config.target_col].describe()}")
        
        # Extract target values
        self.y = self.df[self.config.target_col].astype(float).values
        self.smiles = self.df[self.config.solute_col].tolist()
        self.solvents = self.df[self.config.solvent_col].tolist()  # SMILES for descriptors
        self.solvent_names = self.df[self.config.solvent_name_col].tolist()  # Names for output
        self.solutes = self.df[self.config.solute_col].tolist()
        
        print(f"Sample solvent mapping:")
        print(f"  SMILES: {self.solvents[0]} -> Name: {self.solvent_names[0]}")
        print(f"  SMILES: {self.solvents[1]} -> Name: {self.solvent_names[1]}")
        print(f"Unique solvent SMILES: {len(set(self.solvents))}")
        print(f"Unique solvent names: {len(set(self.solvent_names))}")
        
        # Build feature matrices for different OpenCOSMO settings
        self.feature_matrices = {}
        
        if self.config.test_without_open_cosmo:
            print("\nBuilding feature matrix WITHOUT OpenCOSMO...")
            X_no_cosmo, names_no_cosmo = self.fingerprint_calc.build_feature_matrix(
                self.smiles, self.df, use_open_cosmo=False
            )
            self.feature_matrices['no_cosmo'] = (X_no_cosmo, names_no_cosmo)
            print(f"Feature matrix (no OpenCOSMO) shape: {X_no_cosmo.shape}")
        
        if self.config.use_open_cosmo:
            print("\nBuilding feature matrix WITH OpenCOSMO...")
            X_with_cosmo, names_with_cosmo = self.fingerprint_calc.build_feature_matrix(
                self.smiles, self.df, use_open_cosmo=True
            )
            self.feature_matrices['with_cosmo'] = (X_with_cosmo, names_with_cosmo)
            print(f"Feature matrix (with OpenCOSMO) shape: {X_with_cosmo.shape}")
        
        # Set default feature matrix
        if 'with_cosmo' in self.feature_matrices:
            self.X, self.feature_names = self.feature_matrices['with_cosmo']
        else:
            self.X, self.feature_names = self.feature_matrices['no_cosmo']
        
        return self.X, self.y
    
    def analysis(self):
        """Run the complete analysis pipeline with OpenCOSMO and drip-feeding CV"""
        # Load data
        X, y = self.load_and_prepare_data()
        
        # Get model configurations
        model_configs = self.model_suite.get_model_configs()
        
        # Results storage
        all_results = []
        best_models = {}
        
        # Test different feature matrices and CV strategies
        for matrix_type, (X_matrix, feature_names) in self.feature_matrices.items():
            print(f"\n{'='*80}")
            print(f"TESTING FEATURE MATRIX: {matrix_type.upper()}")
            print(f"{'='*80}")
            
            for cv_strategy in self.config.drip_strategies:
                print(f"\n{'='*60}")
                print(f"CV STRATEGY: {cv_strategy.upper()}")
                print(f"{'='*60}")
                
                # Set up cross-validation
                if cv_strategy == "standard_cv":
                    cv_splitter = KFold(n_splits=self.config.outer_folds, 
                                        shuffle=True, #random_state=SEED
                                        )
                    cv_splits = list(cv_splitter.split(X_matrix, y))
                    groups = None
                elif cv_strategy == "drip_solvent":
                    drip_cv = DripFeedingCV(strategy="drip_solvent")
                    cv_splits = list(drip_cv.split(X_matrix, y, groups=self.solvent_names))
                    groups = self.solvent_names
                elif cv_strategy == "drip_solute":
                    drip_cv = DripFeedingCV(strategy="drip_solute")
                    cv_splits = list(drip_cv.split(X_matrix, y, groups=self.solutes))
                    groups = self.solutes
                
                if not cv_splits:
                    print(f"Warning: No valid splits for {cv_strategy}")
                    continue
                
                fold_results = {}
                for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, 1):
                    print(f"\nFold {fold_idx}/{len(cv_splits)}")
                    
                    X_train, X_test = X_matrix[train_idx], X_matrix[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    fold_best = None
                    fold_best_score = float('inf')
                    
                    # Test each model type
                    for model_name, model_config in model_configs.items():
                        print(f"  Testing {model_name}...")
                        
                        # Test different PCA configurations
                        for pca_config in self.config.pca_configs:
                            try:
                                # Create pipeline
                                pipeline = self.model_suite.create_pipeline(
                                    model_config["model"], pca_config
                                )
                                
                                # Hyperparameter search
                                search = RandomizedSearchCV(
                                    pipeline,
                                    model_config["params"],
                                    n_iter=self.config.search_iterations,
                                    cv=self.config.inner_folds,
                                    
                                    scoring='neg_root_mean_squared_error',
                                    n_jobs=self.config.n_jobs,
                                    verbose=0  # Removed random_state to avoid conflicts with XGBoost
                                )
                                
                                # Fit model
                                search.fit(X_train, y_train)
                                
                                # Predict
                                y_pred = search.predict(X_test)
                                
                                # Calculate metrics
                                metrics = {
                                    'rmse': rmse(y_test, y_pred),
                                    'mae': mae(y_test, y_pred),
                                    'r2': r2(y_test, y_pred),
                                    'mape': mape(y_test, y_pred),
                                    'explained_variance': explained_var(y_test, y_pred)
                                }
                                
                                # Store result
                                result = {
                                    'matrix_type': matrix_type,
                                    'cv_strategy': cv_strategy,
                                    'fold': fold_idx,
                                    'model': model_name,
                                    'pca_config': pca_config,
                                    'best_params': search.best_params_,
                                    'metrics': metrics,
                                    'predictions': y_pred,
                                    'true_values': y_test,
                                    'test_indices': test_idx,
                                    'best_estimator': search.best_estimator_,
                                    'feature_names': feature_names
                                }
                                all_results.append(result)
                                
                                # Track best model for this fold
                                if metrics['rmse'] < fold_best_score:
                                    fold_best_score = metrics['rmse']
                                    fold_best = result
                                
                                print(f"    {model_name} (PCA: {pca_config.get('use_pca', False)}): RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                                
                            except Exception as e:
                                print(f"    Failed: {e}")
                    
                    # Save best model for this fold
                    if fold_best and self.config.save_models:
                        model_path = self.output_dir / f"best_model_{matrix_type}_{cv_strategy}_{fold_idx}.pkl"
                        joblib.dump(fold_best['best_estimator'], model_path)
                    
                    fold_results[fold_idx] = fold_best
        
        # Analyze results using the dedicated reporting class
        self.reporting.analyze_and_report_results_enhanced(
            all_results, 
            self.output_dir,
            self.df,
            self.feature_matrices
        )
        
        return all_results
    