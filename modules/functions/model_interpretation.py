import numpy as np
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class ModelInterpreter:
    """Model interpretation and complexity metrics calculation"""
    
    def __init__(self, config):
        self.config = config
        
    def calculate_complexity_metrics(self, model, X_train, y_train, X_test, y_test):
        """Calculate BIC/AIC-like metrics for model selection"""
        n_train = len(y_train)
        n_test = len(y_test)
        
        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate residual sum of squares
        rss_train = np.sum((y_train - y_pred_train) ** 2)
        rss_test = np.sum((y_test - y_pred_test) ** 2)
        
        # Estimate number of parameters
        n_params = self._estimate_parameters(model)
        
        # Calculate log likelihood (assuming normal errors)
        # log L = -n/2 * log(2π) - n/2 * log(RSS/n) - n/2
        log_likelihood_train = -n_train/2 * np.log(2*np.pi) - n_train/2 * np.log(rss_train/n_train + 1e-10) - n_train/2
        
        # AIC = 2k - 2 log L
        aic = 2 * n_params - 2 * log_likelihood_train
        
        # BIC = k log n - 2 log L
        bic = n_params * np.log(n_train) - 2 * log_likelihood_train
        
        # Adjusted AIC for small samples
        # AICc = AIC + 2k(k+1)/(n-k-1)
        if n_train > n_params + 1:
            aicc = aic + 2 * n_params * (n_params + 1) / (n_train - n_params - 1)
        else:
            aicc = np.inf
        
        # Mallows Cp (for linear models)
        # Cp = RSS_p/σ² + 2p - n
        sigma2_full = rss_train / (n_train - n_params) if n_train > n_params else 1.0
        mallows_cp = rss_test / sigma2_full + 2 * n_params - n_test
        
        # Generalized Cross-Validation (GCV)
        # GCV = RSS / (n * (1 - p/n)²)
        gcv = rss_train / (n_train * (1 - n_params/n_train)**2) if n_train > n_params else np.inf
        
        # Model complexity penalty (custom metric)
        # Penalizes complex models more heavily
        complexity_penalty = n_params / np.sqrt(n_train)
        
        # Information criterion for prediction (custom)
        # Balances in-sample fit with out-of-sample performance
        prediction_ic = rss_test / n_test + complexity_penalty * np.log(n_train)
        
        return {
            'aic': aic,
            'bic': bic,
            'aicc': aicc,
            'mallows_cp': mallows_cp,
            'gcv': gcv,
            'n_parameters': n_params,
            'complexity_penalty': complexity_penalty,
            'prediction_ic': prediction_ic,
            'log_likelihood': log_likelihood_train
        }
    
    def _estimate_parameters(self, model):
        """Estimate number of parameters in the model"""
        if isinstance(model, Pipeline):
            # Get the final estimator from pipeline
            model = model.steps[-1][1]
        
        # Model-specific parameter counting
        model_name = type(model).__name__.lower()
        
        if 'xgb' in model_name:
            # XGBoost: number of trees * average leaves per tree
            if hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    n_trees = booster.num_boosted_rounds() if hasattr(booster, 'num_boosted_rounds') else 100
                    # Approximate leaves per tree
                    avg_leaves = 2 ** model.max_depth if hasattr(model, 'max_depth') and model.max_depth else 32
                    return n_trees * avg_leaves
                except:
                    return model.n_estimators * 32 if hasattr(model, 'n_estimators') else 3200
            
        elif 'lgb' in model_name or 'lightgbm' in model_name:
            # LightGBM: similar to XGBoost
            n_trees = model.n_estimators if hasattr(model, 'n_estimators') else 100
            num_leaves = model.num_leaves if hasattr(model, 'num_leaves') else 31
            return n_trees * num_leaves
            
        elif 'randomforest' in model_name:
            # Random Forest: trees * average nodes per tree
            n_trees = model.n_estimators if hasattr(model, 'n_estimators') else 100
            # Approximate nodes based on max_depth
            if hasattr(model, 'max_depth') and model.max_depth:
                avg_nodes = 2 ** (model.max_depth + 1) - 1
            else:
                avg_nodes = 100  # Default estimate
            return n_trees * avg_nodes
            
        elif 'mlp' in model_name:
            # Neural Network: count weights and biases
            if hasattr(model, 'coefs_'):
                n_params = sum(coef.size for coef in model.coefs_)
                n_params += sum(bias.size for bias in model.intercepts_)
                return n_params
            else:
                # Estimate from layer sizes
                layers = model.hidden_layer_sizes if hasattr(model, 'hidden_layer_sizes') else (100,)
                if isinstance(layers, int):
                    layers = (layers,)
                n_params = 0
                prev_size = None  # Will be set after knowing input dimension
                for layer_size in layers:
                    if prev_size is not None:
                        n_params += prev_size * layer_size + layer_size
                    prev_size = layer_size
                return n_params or 1000  # Default estimate
                
        elif 'ridge' in model_name or 'lasso' in model_name or 'elasticnet' in model_name:
            # Linear models: number of features + intercept
            if hasattr(model, 'coef_'):
                return model.coef_.size + 1
            else:
                return 100  # Default estimate
                
        elif 'svr' in model_name or 'svm' in model_name:
            # SVM: number of support vectors
            if hasattr(model, 'support_vectors_'):
                return len(model.support_vectors_)
            else:
                return 100  # Default estimate
                
        elif 'mixture' in model_name:
            # Mixture of Experts: sum of sub-model parameters
            n_params = 0
            if hasattr(model, 'low_model'):
                n_params += self._estimate_parameters(model.low_model)
            if hasattr(model, 'high_model'):
                n_params += self._estimate_parameters(model.high_model)
            # Add gating network parameters
            n_params += 100  # Approximate for logistic regression gate
            return n_params
            
        else:
            # Default: assume moderate complexity
            return 100
    
    def get_feature_importance(self, model, feature_names):
        """Extract feature importance from various model types"""
        if isinstance(model, Pipeline):
            # Get the final estimator from pipeline
            model = model.steps[-1][1]
            
        model_name = type(model).__name__.lower()
        
        # Try to get feature importance
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (XGBoost, LightGBM, RandomForest)
            importance = model.feature_importances_
            
        elif hasattr(model, 'coef_'):
            # Linear models - use absolute coefficients
            coef = model.coef_
            if coef.ndim > 1:
                # Multi-output or multi-class - average across outputs
                importance = np.mean(np.abs(coef), axis=0)
            else:
                importance = np.abs(coef)
                
        elif 'mixture' in model_name and hasattr(model, 'low_model'):
            # Mixture of Experts - average importance from sub-models
            low_imp = self.get_feature_importance(model.low_model, feature_names)
            high_imp = self.get_feature_importance(model.high_model, feature_names)
            if low_imp is not None and high_imp is not None:
                importance = (low_imp + high_imp) / 2
                
        if importance is not None and len(importance) == len(feature_names):
            # Create importance dictionary
            importance_dict = {
                'features': feature_names,
                'importance': importance.tolist() if hasattr(importance, 'tolist') else importance,
                'top_features': self._get_top_features(feature_names, importance)
            }
            return importance_dict
        
        return None
    
    def _get_top_features(self, feature_names, importance, n_top=20):
        """Get top N most important features"""
        if len(feature_names) != len(importance):
            return []
        
        # Sort by importance
        indices = np.argsort(importance)[::-1][:n_top]
        
        top_features = []
        for idx in indices:
            if importance[idx] > self.config.importance_threshold:
                top_features.append({
                    'name': feature_names[idx],
                    'importance': float(importance[idx])
                })
        
        return top_features