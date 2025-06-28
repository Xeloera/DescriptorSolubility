from sklearn.metrics import (make_scorer, r2_score, mean_squared_error, 
                           mean_absolute_error, explained_variance_score)

import numpy as np


# ═════════════════════════ ENHANCED METRICS ═══════════════════════════════════

def rmse(y_true, y_pred): 
    """Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred): 
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred): 
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def r2_robust(y_true, y_pred):
    """
    Robust R² calculation that handles edge cases gracefully.
    Returns NaN for problematic cases instead of extreme negative values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check for edge cases
    if len(y_true) < 2:
        return np.nan
    
    # Check for constant true values (no variance to explain)
    if np.var(y_true) == 0:
        return np.nan if not np.allclose(y_true, y_pred) else 1.0
    
    # Check for constant predictions
    if np.var(y_pred) == 0:
        return 0.0 if np.allclose(y_pred, np.mean(y_true)) else -np.inf
    
    # Calculate standard R²
    r2_val = r2_score(y_true, y_pred)
    
    # Cap extremely negative values that indicate degenerate cases
    if r2_val < -10:  # Threshold for "unreasonably bad" R²
        return np.nan
    
    return r2_val

def r2(y_true, y_pred): 
    """Legacy function that calls robust R² calculation"""
    return r2_robust(y_true, y_pred)

def explained_var(y_true, y_pred): 
    """Explained Variance Score"""
    return explained_variance_score(y_true, y_pred)

# Enhanced scorer collection
ENHANCED_SCORERS = {
    'rmse': make_scorer(rmse, greater_is_better=False),
    'mae': make_scorer(mae, greater_is_better=False),
    'r2': make_scorer(r2_score),
    'explained_variance': make_scorer(explained_variance_score),
}