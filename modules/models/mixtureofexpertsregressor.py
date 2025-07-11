from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import numpy as np

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


class MixtureOfExpertsRegressor(BaseEstimator, TransformerMixin):
    """Mixture of Experts with gating network for high/low solubility prediction"""
    
    def __init__(self, low_model, high_model, threshold_method="median", threshold_value=0.25, 
                 n_experts=2, gate_type="soft", regularization=0.01, verbose=False):
        self.low_model = low_model
        self.high_model = high_model
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.n_experts = n_experts
        self.gate_type = gate_type
        self.regularization = regularization
        self.verbose = verbose
        
        # Initialize gating network with regularization
        self.gating_network = LogisticRegression(
            max_iter=5000,  # Increased iterations
            tol=1e-4,       # Better tolerance
            solver='lbfgs',  # LBFGS solver
            C=1.0/self.regularization if self.regularization > 0 else 1e10,  # Convert to C parameter
            multi_class='ovr' if n_experts > 2 else 'auto'
        )
        
        # For multi-expert scenario
        if n_experts > 2:
            self.expert_models = [low_model, high_model]  # Will extend dynamically
        
        self.threshold = None
        self.is_fitted = False
        self._fit_count = 0  # Track how many times fit() is called
    
    def _determine_threshold(self, y):
        """Determine threshold for high/low solubility split"""
        if self.threshold_method == "median":
            return np.median(y)
        elif self.threshold_method == "percentile":
            return np.percentile(y, self.threshold_value * 100)
        elif self.threshold_method == "solubility_cutoff":
            # Use logS = 0 as the natural solubility threshold
            # logS < 0: poorly soluble/insoluble compounds
            # logS >= 0: soluble compounds
            return 0.0
        elif self.threshold_method == "quartile":
            # Use first and third quartiles for more balanced split
            q1, q3 = np.percentile(y, [25, 75])
            return (q1 + q3) / 2
        elif self.threshold_method == "iqr_optimized":
            # Find optimal threshold within IQR using MSE minimization
            q1, q3 = np.percentile(y, [25, 75])
            thresholds = np.linspace(q1, q3, 20)
            best_threshold = q1
            best_score = float('inf')
            
            for thresh in thresholds:
                low_mask = y <= thresh
                high_mask = y > thresh
                
                if low_mask.sum() > 10 and high_mask.sum() > 10:
                    low_var = np.var(y[low_mask])
                    high_var = np.var(y[high_mask])
                    weighted_var = (low_mask.sum() * low_var + high_mask.sum() * high_var) / len(y)
                    
                    if weighted_var < best_score:
                        best_score = weighted_var
                        best_threshold = thresh
            
            return best_threshold
        elif self.threshold_method == "gradient_based":
            # Use gradient of sorted values to find natural breaks
            sorted_y = np.sort(y)
            gradients = np.diff(sorted_y)
            # Find largest gradient (biggest jump)
            max_grad_idx = np.argmax(gradients)
            return sorted_y[max_grad_idx]
        elif self.threshold_method == "kmeans":
            # Use K-means to find natural split
            kmeans = KMeans(n_clusters=2, 
                            #random_state=SEED, 
                            n_init=10)
            y_reshaped = y.reshape(-1, 1)
            clusters = kmeans.fit_predict(y_reshaped)
            centers = kmeans.cluster_centers_.flatten()
            return np.mean(centers)
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
    
    def fit(self, X, y):
        """Fit the mixture of experts model"""
        self._fit_count += 1
        
        # Determine threshold
        self.threshold = self._determine_threshold(y)
        
        # Create binary labels for gating network (0=low/insoluble, 1=high/soluble)
        gate_labels = (y > self.threshold).astype(int)
        
        # Split data
        low_mask = gate_labels == 0  # Insoluble compounds (logS < 0)
        high_mask = gate_labels == 1  # Soluble compounds (logS >= 0)
        
        # Only print verbose info occasionally to avoid spam during hyperparameter search
        should_print = (self.verbose and 
                       (self._fit_count == 1 or  # First fit
                        self._fit_count % 50 == 0 or  # Every 50th fit
                        len(y) > 5000))  # Final model fits (usually larger datasets)
        
        if should_print:
            if self.threshold_method == "solubility_cutoff":
                print(f"MoE Fit #{self._fit_count}: Solubility T={self.threshold:.1f} | Insoluble={low_mask.sum()}({low_mask.sum()/len(y)*100:.0f}%) | Soluble={high_mask.sum()}({high_mask.sum()/len(y)*100:.0f}%)")
            else:
                print(f"MoE Fit #{self._fit_count}: T={self.threshold:.3f} | Low={low_mask.sum()}({low_mask.sum()/len(y)*100:.0f}%) | High={high_mask.sum()}({high_mask.sum()/len(y)*100:.0f}%)")
        
        # Fit gating network
        self.gating_network.fit(X, gate_labels)
        
        # Fit expert models on their respective data
        if low_mask.sum() > 0:
            self.low_model.fit(X[low_mask], y[low_mask])
        if high_mask.sum() > 0:
            self.high_model.fit(X[high_mask], y[high_mask])
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using mixture of experts - route to appropriate expert based on gating network"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get gating network hard predictions (which expert to use)
        gate_assignments = self.gating_network.predict(X)  # 0=low expert, 1=high expert
        
        # Initialize predictions array
        final_preds = np.zeros(len(X))
        
        # Route to low solubility expert (insoluble compounds, logS < 0)
        low_mask = gate_assignments == 0
        if np.any(low_mask):
            final_preds[low_mask] = self.low_model.predict(X[low_mask])
        
        # Route to high solubility expert (soluble compounds, logS >= 0)
        high_mask = gate_assignments == 1
        if np.any(high_mask):
            final_preds[high_mask] = self.high_model.predict(X[high_mask])
        
        return final_preds
    
    def predict_expert_assignment(self, X):
        """Return which expert each sample would be assigned to with confidence scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        gate_assignments = self.gating_network.predict(X)  # Hard assignments
        gate_probs = self.gating_network.predict_proba(X)  # Soft probabilities
        
        return {
            'assignments': gate_assignments,  # 0=low expert, 1=high expert
            'probabilities': gate_probs,      # [prob_low, prob_high]
            'confidence': np.max(gate_probs, axis=1)  # Confidence in assignment
        }
    
    def predict_soft_routing(self, X):
        """Alternative prediction using soft routing (weighted combination)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get gating network probabilities
        gate_probs = self.gating_network.predict_proba(X)[:, 1]  # Probability of high solubility
        
        # Get predictions from both experts
        low_preds = self.low_model.predict(X)
        high_preds = self.high_model.predict(X)
        
        # Weighted combination based on gating network
        final_preds = (1 - gate_probs) * low_preds + gate_probs * high_preds
        
        return final_preds
    
    def set_verbose_mode(self, verbose=True):
        """Set verbose mode and reset fit counter for final model training"""
        self.verbose = verbose
        self._fit_count = 0  # Reset counter for cleaner final output
        return self
