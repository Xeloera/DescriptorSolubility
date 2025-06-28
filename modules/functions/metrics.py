from sklearn.metrics import (make_scorer, r2_score, mean_squared_error, 
                           mean_absolute_error, explained_variance_score)

import numpy as np


# ═════════════════════════ ENHANCED METRICS ═══════════════════════════════════

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred): return mean_absolute_error(y_true, y_pred)
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
def r2(y_true, y_pred): return r2_score(y_true, y_pred)
def explained_var(y_true, y_pred): return explained_variance_score(y_true, y_pred)

# Enhanced scorer collection
ENHANCED_SCORERS = {
    'rmse': make_scorer(rmse, greater_is_better=False),
    'mae': make_scorer(mae, greater_is_better=False),
    'r2': make_scorer(r2_score),
    'explained_variance': make_scorer(explained_variance_score),
}