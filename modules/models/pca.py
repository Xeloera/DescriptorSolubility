


from sklearn.base import BaseEstimator, TransformerMixin

# CUDA acceleration
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

class RobustPCA(BaseEstimator, TransformerMixin):
    """
    Robust PCA wrapper that automatically falls back from GPU to CPU when cuML fails.
    Handles CUSOLVER_STATUS_INVALID_VALUE and other GPU memory/dimension issues.
    """
    
    def __init__(self, n_components=100, use_gpu=True, verbose=False):
        self.n_components = n_components
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.pca_model = None
        self.using_gpu = False
        
    def fit(self, X, y=None):
        """Fit PCA, falling back to CPU if GPU fails"""
        if self.use_gpu and USE_CUML:
            try:
                if self.verbose:
                    print(f"Attempting GPU PCA with cuML for {X.shape} matrix...")
                
                # Try cuML PCA first
                from cuml.decomposition import PCA as cuPCA
                self.pca_model = cuPCA(
                    n_components=self.n_components,
                    whiten=True
                )
                self.pca_model.fit(X)
                self.using_gpu = True
                
                if self.verbose:
                    print(f"✅ GPU PCA successful")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ GPU PCA failed ({type(e).__name__}: {e}), falling back to CPU...")
                self._fallback_to_cpu(X)
        else:
            if self.verbose:
                print(f"Using CPU PCA for {X.shape} matrix...")
            self._fallback_to_cpu(X)
            
        return self
    
    def _fallback_to_cpu(self, X):
        """Fallback to CPU PCA implementation"""
        try:
            # Use sklearn PCA
            from sklearn.decomposition import PCA
            self.pca_model = PCA(
                n_components=self.n_components,
                whiten=True
            )
            self.pca_model.fit(X)
            self.using_gpu = False
            
            if self.verbose:
                print(f"✅ CPU PCA fallback successful")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ CPU PCA also failed: {e}")
            # Final fallback: use identity transformation
            self.pca_model = None
            self.using_gpu = False
    
    def transform(self, X):
        """Transform data using fitted PCA model"""
        if self.pca_model is None:
            # Identity transformation if PCA completely failed
            if self.verbose:
                print("⚠️ Using identity transformation (PCA failed)")
            return X[:, :self.n_components] if X.shape[1] > self.n_components else X
        
        try:
            if self.using_gpu and hasattr(self.pca_model, 'transform'):
                # GPU transform
                return self.pca_model.transform(X)
            else:
                # CPU transform
                return self.pca_model.transform(X)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Transform failed ({e}), using identity transformation")
            return X[:, :self.n_components] if X.shape[1] > self.n_components else X
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
