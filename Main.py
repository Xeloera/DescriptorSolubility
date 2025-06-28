#!/usr/bin/env python3
"""
================================================================

A complete solubility prediction workbench featuring:
- DScribe descriptors (SOAP, MBTR, Sine Matrix, Coulomb Matrix)
- Multi-model comparison (XGBoost, LightGBM, Random Forest, Neural Networks)
- GPU-accelerated visualization (t-SNE, UMAP with CuML)
- Comprehensive statistical analysis and reporting
- Advanced fingerprint combinations
- Per-solvent, per-solute, per-cluster analysis

© 2025 · Enhanced Version with Full DScribe Integration
"""

import os, time, warnings, itertools, json, random, joblib, pickle
import numpy as np, pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import traceback

# Set CUDA devices early
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['XGB_DEVICE'] = 'GPU'

# ── Chemistry toolkits ────────────────────────────────────────────────────────
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import (Descriptors, AllChem, DataStructs,
                        rdFingerprintGenerator, ChemicalFeatures)
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

# DScribe descriptors - Full suite
from dscribe.descriptors import SOAP, MBTR, SineMatrix, CoulombMatrix, ACSF
from ase import Atoms

# ── ML toolchain with CUDA acceleration ──────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (KFold, RandomizedSearchCV, GridSearchCV,
                                     cross_validate, StratifiedKFold)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (make_scorer, r2_score, mean_squared_error, 
                           mean_absolute_error, explained_variance_score)
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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

# ── Parallel processing ───────────────────────────────────────────────────────
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count

# ── Visualization and analysis ───────────────────────────────────────────────
from sklearn.manifold import TSNE
try:
    import umap.umap_ as UMAP
    USE_UMAP = True
except ImportError:
    USE_UMAP = False

# ── Global settings ───────────────────────────────────────────────────────────
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=FutureWarning, module="dscribe")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")  # Suppress sklearn pipeline warnings

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if cp is not None:
    cp.random.seed(SEED)

# Performance settings - Maximize CPU cores for high performance
N_JOBS = min(cpu_count(), 128)  # Use all available cores up to 128
BATCH_SIZE = 1000

print(f"CUDA Devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
if USE_CUML:
    print(f"Available GPUs: {cp.cuda.runtime.getDeviceCount()}")
print(f"CPU cores available: {N_JOBS}")
print(f"CuML available: {USE_CUML}")
print(f"UMAP available: {USE_UMAP}")
OPEN_COSMO = [
    'opencosmo_solubility_mole_frac',
    'opencosmo_solubility_mass_fraction',
    'opencosmo_solubility_g_100g',
    'opencosmo_solubility_g_100g_log',
    'cosmo_solubility_mole_frac',
    'cosmo_solubility_mass_fraction',
    'cosmo_solubility_g_100g',
    'cosmo_solubility_g_100g_log'
]

PRECOMPUTED_SOLVENT_FEATURES = [
    # EState Indices
    'solvent_MaxAbsEStateIndex', 'solvent_MaxEStateIndex', 'solvent_MinAbsEStateIndex', 'solvent_MinEStateIndex',
    # General Descriptors
    'solvent_qed', 'solvent_SPS', 'solvent_MolWt', 'solvent_HeavyAtomMolWt', 'solvent_ExactMolWt',
    'solvent_NumValenceElectrons', 'solvent_NumRadicalElectrons', 'solvent_MaxPartialCharge',
    'solvent_MinPartialCharge', 'solvent_MaxAbsPartialCharge', 'solvent_MinAbsPartialCharge',
    'solvent_FpDensityMorgan1', 'solvent_FpDensityMorgan2', 'solvent_FpDensityMorgan3',
    # BCUT Descriptors
    'solvent_BCUT2D_MWHI', 'solvent_BCUT2D_MWLOW', 'solvent_BCUT2D_CHGHI', 'solvent_BCUT2D_CHGLO',
    'solvent_BCUT2D_LOGPHI', 'solvent_BCUT2D_LOGPLOW', 'solvent_BCUT2D_MRHI', 'solvent_BCUT2D_MRLOW',
    # Topological Descriptors
    'solvent_AvgIpc', 'solvent_BalabanJ', 'solvent_BertzCT', 'solvent_Chi0', 'solvent_Chi0n', 'solvent_Chi0v',
    'solvent_Chi1', 'solvent_Chi1n', 'solvent_Chi1v', 'solvent_Chi2n', 'solvent_Chi2v', 'solvent_Chi3n',
    'solvent_Chi3v', 'solvent_Chi4n', 'solvent_Chi4v', 'solvent_HallKierAlpha', 'solvent_Ipc', 'solvent_Kappa1',
    'solvent_Kappa2', 'solvent_Kappa3', 'solvent_LabuteASA',
    # VSA Descriptors
    'solvent_PEOE_VSA1', 'solvent_PEOE_VSA10', 'solvent_PEOE_VSA11', 'solvent_PEOE_VSA12', 'solvent_PEOE_VSA13',
    'solvent_PEOE_VSA14', 'solvent_PEOE_VSA2', 'solvent_PEOE_VSA3', 'solvent_PEOE_VSA4', 'solvent_PEOE_VSA5',
    'solvent_PEOE_VSA6', 'solvent_PEOE_VSA7', 'solvent_PEOE_VSA8', 'solvent_PEOE_VSA9',
    'solvent_SMR_VSA1', 'solvent_SMR_VSA10', 'solvent_SMR_VSA2', 'solvent_SMR_VSA3', 'solvent_SMR_VSA4',
    'solvent_SMR_VSA5', 'solvent_SMR_VSA6', 'solvent_SMR_VSA7', 'solvent_SMR_VSA8', 'solvent_SMR_VSA9',
    'solvent_SlogP_VSA1', 'solvent_SlogP_VSA10', 'solvent_SlogP_VSA11', 'solvent_SlogP_VSA12',
    'solvent_SlogP_VSA2', 'solvent_SlogP_VSA3', 'solvent_SlogP_VSA4', 'solvent_SlogP_VSA5',
    'solvent_SlogP_VSA6', 'solvent_SlogP_VSA7', 'solvent_SlogP_VSA8', 'solvent_SlogP_VSA9',
    'solvent_TPSA',
    'solvent_EState_VSA1', 'solvent_EState_VSA10', 'solvent_EState_VSA11', 'solvent_EState_VSA2',
    'solvent_EState_VSA3', 'solvent_EState_VSA4', 'solvent_EState_VSA5', 'solvent_EState_VSA6',
    'solvent_EState_VSA7', 'solvent_EState_VSA8', 'solvent_EState_VSA9',
    'solvent_VSA_EState1', 'solvent_VSA_EState10', 'solvent_VSA_EState2', 'solvent_VSA_EState3',
    'solvent_VSA_EState4', 'solvent_VSA_EState5', 'solvent_VSA_EState6', 'solvent_VSA_EState7',
    'solvent_VSA_EState8', 'solvent_VSA_EState9',
    # Property Counts / Ratios
    'solvent_FractionCSP3', 'solvent_HeavyAtomCount', 'solvent_NHOHCount', 'solvent_NOCount',
    'solvent_NumAliphaticCarbocycles', 'solvent_NumAliphaticHeterocycles', 'solvent_NumAliphaticRings',
    'solvent_NumAmideBonds', 'solvent_NumAromaticCarbocycles', 'solvent_NumAromaticHeterocycles',
    'solvent_NumAromaticRings', 'solvent_NumAtomStereoCenters', 'solvent_NumBridgeheadAtoms',
    'solvent_NumHAcceptors', 'solvent_NumHDonors', 'solvent_NumHeteroatoms', 'solvent_NumHeterocycles',
    'solvent_NumRotatableBonds', 'solvent_NumSaturatedCarbocycles', 'solvent_NumSaturatedHeterocycles',
    'solvent_NumSaturatedRings', 'solvent_NumSpiroAtoms', 'solvent_NumUnspecifiedAtomStereoCenters',
    'solvent_Phi', 'solvent_RingCount', 'solvent_MolLogP', 'solvent_MolMR',
    # Fragment Counts (fr_*) - Assuming all fr_* columns are relevant and numeric
    'solvent_fr_Al_COO', 'solvent_fr_Al_OH', 'solvent_fr_Al_OH_noTert', 'solvent_fr_ArN', 'solvent_fr_Ar_COO',
    'solvent_fr_Ar_N', 'solvent_fr_Ar_NH', 'solvent_fr_Ar_OH', 'solvent_fr_COO', 'solvent_fr_COO2',
    'solvent_fr_C_O', 'solvent_fr_C_O_noCOO', 'solvent_fr_C_S', 'solvent_fr_HOCCN', 'solvent_fr_Imine',
    'solvent_fr_NH0', 'solvent_fr_NH1', 'solvent_fr_NH2', 'solvent_fr_N_O', 'solvent_fr_Ndealkylation1',
    'solvent_fr_Ndealkylation2', 'solvent_fr_Nhpyrrole', 'solvent_fr_SH', 'solvent_fr_aldehyde',
    'solvent_fr_alkyl_carbamate', 'solvent_fr_alkyl_halide', 'solvent_fr_allylic_oxid', 'solvent_fr_amide',
    'solvent_fr_amidine', 'solvent_fr_aniline', 'solvent_fr_aryl_methyl', 'solvent_fr_azide', 'solvent_fr_azo',
    'solvent_fr_barbitur', 'solvent_fr_benzene', 'solvent_fr_benzodiazepine', 'solvent_fr_bicyclic',
    'solvent_fr_diazo', 'solvent_fr_dihydropyridine', 'solvent_fr_epoxide', 'solvent_fr_ester',
    'solvent_fr_ether', 'solvent_fr_furan', 'solvent_fr_guanido', 'solvent_fr_halogen', 'solvent_fr_hdrzine',
    'solvent_fr_hdrzone', 'solvent_fr_imidazole', 'solvent_fr_imide', 'solvent_fr_isocyan',
    'solvent_fr_isothiocyan', 'solvent_fr_ketone', 'solvent_fr_ketone_Topliss', 'solvent_fr_lactam',
    'solvent_fr_lactone', 'solvent_fr_methoxy', 'solvent_fr_morpholine', 'solvent_fr_nitrile',
    'solvent_fr_nitro', 'solvent_fr_nitro_arom', 'solvent_fr_nitro_arom_nonortho', 'solvent_fr_nitroso',
    'solvent_fr_oxazole', 'solvent_fr_oxime', 'solvent_fr_para_hydroxylation', 'solvent_fr_phenol',
    'solvent_fr_phenol_noOrthoHbond', 'solvent_fr_phos_acid', 'solvent_fr_phos_ester', 'solvent_fr_piperdine',
    'solvent_fr_piperzine', 'solvent_fr_priamide', 'solvent_fr_prisulfonamd', 'solvent_fr_pyridine',
    'solvent_fr_quatN', 'solvent_fr_sulfide', 'solvent_fr_sulfonamd', 'solvent_fr_sulfone',
    'solvent_fr_term_acetylene', 'solvent_fr_tetrazole', 'solvent_fr_thiazole', 'solvent_fr_thiocyan',
    'solvent_fr_thiophene', 'solvent_fr_unbrch_alkane', 'solvent_fr_urea'
]
PRECOMPUTED_SOLUTE_FEATURES = [
    # EState Indices
    'solute_MaxAbsEStateIndex', 'solute_MaxEStateIndex', 'solute_MinAbsEStateIndex', 'solute_MinEStateIndex',
    # General Descriptors
    'solute_qed', 'solute_SPS', 'solute_MolWt', 'solute_HeavyAtomMolWt', 'solute_ExactMolWt',
    'solute_NumValenceElectrons', 'solute_NumRadicalElectrons', 'solute_MaxPartialCharge',
    'solute_MinPartialCharge', 'solute_MaxAbsPartialCharge', 'solute_MinAbsPartialCharge',
    'solute_FpDensityMorgan1', 'solute_FpDensityMorgan2', 'solute_FpDensityMorgan3',
    # BCUT Descriptors
    'solute_BCUT2D_MWHI', 'solute_BCUT2D_MWLOW', 'solute_BCUT2D_CHGHI', 'solute_BCUT2D_CHGLO',
    'solute_BCUT2D_LOGPHI', 'solute_BCUT2D_LOGPLOW', 'solute_BCUT2D_MRHI', 'solute_BCUT2D_MRLOW',
    # Topological Descriptors
    'solute_AvgIpc', 'solute_BalabanJ', 'solute_BertzCT', 'solute_Chi0', 'solute_Chi0n', 'solute_Chi0v',
    'solute_Chi1', 'solute_Chi1n', 'solute_Chi1v', 'solute_Chi2n', 'solute_Chi2v', 'solute_Chi3n',
    'solute_Chi3v', 'solute_Chi4n', 'solute_Chi4v', 'solute_HallKierAlpha', 'solute_Ipc', 'solute_Kappa1',
    'solute_Kappa2', 'solute_Kappa3', 'solute_LabuteASA',
    # VSA Descriptors
    'solute_PEOE_VSA1', 'solute_PEOE_VSA10', 'solute_PEOE_VSA11', 'solute_PEOE_VSA12', 'solute_PEOE_VSA13',
    'solute_PEOE_VSA14', 'solute_PEOE_VSA2', 'solute_PEOE_VSA3', 'solute_PEOE_VSA4', 'solute_PEOE_VSA5',
    'solute_PEOE_VSA6', 'solute_PEOE_VSA7', 'solute_PEOE_VSA8', 'solute_PEOE_VSA9',
    'solute_SMR_VSA1', 'solute_SMR_VSA10', 'solute_SMR_VSA2', 'solute_SMR_VSA3', 'solute_SMR_VSA4',
    'solute_SMR_VSA5', 'solute_SMR_VSA6', 'solute_SMR_VSA7', 'solute_SMR_VSA8', 'solute_SMR_VSA9',
    'solute_SlogP_VSA1', 'solute_SlogP_VSA10', 'solute_SlogP_VSA11', 'solute_SlogP_VSA12',
    'solute_SlogP_VSA2', 'solute_SlogP_VSA3', 'solute_SlogP_VSA4', 'solute_SlogP_VSA5',
    'solute_SlogP_VSA6', 'solute_SlogP_VSA7', 'solute_SlogP_VSA8', 'solute_SlogP_VSA9',
    'solute_TPSA',
    'solute_EState_VSA1', 'solute_EState_VSA10', 'solute_EState_VSA11', 'solute_EState_VSA2',
    'solute_EState_VSA3', 'solute_EState_VSA4', 'solute_EState_VSA5', 'solute_EState_VSA6',
    'solute_EState_VSA7', 'solute_EState_VSA8', 'solute_EState_VSA9',
    'solute_VSA_EState1', 'solute_VSA_EState10', 'solute_VSA_EState2', 'solute_VSA_EState3',
    'solute_VSA_EState4', 'solute_VSA_EState5', 'solute_VSA_EState6', 'solute_VSA_EState7',
    'solute_VSA_EState8', 'solute_VSA_EState9',
    # Property Counts / Ratios
    'solute_FractionCSP3', 'solute_HeavyAtomCount', 'solute_NHOHCount', 'solute_NOCount',
    'solute_NumAliphaticCarbocycles', 'solute_NumAliphaticHeterocycles', 'solute_NumAliphaticRings',
    'solute_NumAmideBonds', 'solute_NumAromaticCarbocycles', 'solute_NumAromaticHeterocycles',
    'solute_NumAromaticRings', 'solute_NumAtomStereoCenters', 'solute_NumBridgeheadAtoms',
    'solute_NumHAcceptors', 'solute_NumHDonors', 'solute_NumHeteroatoms', 'solute_NumHeterocycles',
    'solute_NumRotatableBonds', 'solute_NumSaturatedCarbocycles', 'solute_NumSaturatedHeterocycles',
    'solute_NumSaturatedRings', 'solute_NumSpiroAtoms', 'solute_NumUnspecifiedAtomStereoCenters',
    'solute_Phi', 'solute_RingCount', 'solute_MolLogP', 'solute_MolMR',
    # Fragment Counts (fr_*) - Assuming all fr_* columns are relevant and numeric
    'solute_fr_Al_COO', 'solute_fr_Al_OH', 'solute_fr_Al_OH_noTert', 'solute_fr_ArN', 'solute_fr_Ar_COO',
    'solute_fr_Ar_N', 'solute_fr_Ar_NH', 'solute_fr_Ar_OH', 'solute_fr_COO', 'solute_fr_COO2',
    'solute_fr_C_O', 'solute_fr_C_O_noCOO', 'solute_fr_C_S', 'solute_fr_HOCCN', 'solute_fr_Imine',
    'solute_fr_NH0', 'solute_fr_NH1', 'solute_fr_NH2', 'solute_fr_N_O', 'solute_fr_Ndealkylation1',
    'solute_fr_Ndealkylation2', 'solute_fr_Nhpyrrole', 'solute_fr_SH', 'solute_fr_aldehyde',
    'solute_fr_alkyl_carbamate', 'solute_fr_alkyl_halide', 'solute_fr_allylic_oxid', 'solute_fr_amide',
    'solute_fr_amidine', 'solute_fr_aniline', 'solute_fr_aryl_methyl', 'solute_fr_azide', 'solute_fr_azo',
    'solute_fr_barbitur', 'solute_fr_benzene', 'solute_fr_benzodiazepine', 'solute_fr_bicyclic',
    'solute_fr_diazo', 'solute_fr_dihydropyridine', 'solute_fr_epoxide', 'solute_fr_ester',
    'solute_fr_ether', 'solute_fr_furan', 'solute_fr_guanido', 'solute_fr_halogen', 'solute_fr_hdrzine',
    'solute_fr_hdrzone', 'solute_fr_imidazole', 'solute_fr_imide', 'solute_fr_isocyan',
    'solute_fr_isothiocyan', 'solute_fr_ketone', 'solute_fr_ketone_Topliss', 'solute_fr_lactam',
    'solute_fr_lactone', 'solute_fr_methoxy', 'solute_fr_morpholine', 'solute_fr_nitrile',
    'solute_fr_nitro', 'solute_fr_nitro_arom', 'solute_fr_nitro_arom_nonortho', 'solute_fr_nitroso',
    'solute_fr_oxazole', 'solute_fr_oxime', 'solute_fr_para_hydroxylation', 'solute_fr_phenol',
    'solute_fr_phenol_noOrthoHbond', 'solute_fr_phos_acid', 'solute_fr_phos_ester', 'solute_fr_piperdine',
    'solute_fr_piperzine', 'solute_fr_priamide', 'solute_fr_prisulfonamd', 'solute_fr_pyridine',
    'solute_fr_quatN', 'solute_fr_sulfide', 'solute_fr_sulfonamd', 'solute_fr_sulfone',
    'solute_fr_term_acetylene', 'solute_fr_tetrazole', 'solute_fr_thiazole', 'solute_fr_thiocyan',
    'solute_fr_thiophene', 'solute_fr_unbrch_alkane', 'solute_fr_urea'
]
SOLVENT_COLS = OPEN_COSMO + PRECOMPUTED_SOLVENT_FEATURES + PRECOMPUTED_SOLUTE_FEATURES

# ═════════════════════════ ENHANCED CONFIGURATION ═══════════════════════════════
@dataclass
class EnhancedConfig:
    """Enhanced configuration with comprehensive options"""
    # Dataset - IMPORTANT: Dual solvent handling for best of both worlds
    tsv_path: str = "converted_2025-04-25_104236.tsv"
    target_col: str = "solubility_g_100g_log"
    solute_col: str = "solute_smiles"
    solvent_col: str = "solvent_smiles"  # FIXED: Use SMILES for molecular descriptors!
    solvent_name_col: str = "solvent_name"  # Keep names for readable output/analysis
    # ↑ This gives us the best of both: 
    #   - SMILES for accurate molecular fingerprints/descriptors
    #   - Names for human-readable logs and analysis
    
    # Fingerprint configurations
    fp_configs: List[Dict] = field(default_factory=lambda: [
        #{"type": "morgan", "n_bits": 2048, "radius": 2},
        #{"type": "morgan", "n_bits": 2048, "radius": 3},
        {"type": "morgan", "n_bits": 4096, "radius": 3},
        {"type": "combined", "n_bits": 4096, "radius": 3},  # Morgan + Avalon + Torsion
        {"type": "rdkit_descriptors"},
    ])
    
    # DScribe descriptors - Enhanced with multiple configurations
    use_dscribe: bool = True
    dscribe_descriptors: List[str] = field(default_factory=lambda: [
        "soap", "sine_matrix", "mbtr"  # Added MBTR for completeness
    ])
    
    # Enhanced DScribe configurations with more diversity
    soap_configs: List[Dict] = field(default_factory=lambda: [
        {"r_cut": 6.0, "n_max": 8, "l_max": 6, "permutation": "none"},   # Primary config
        {"r_cut": 8.0, "n_max": 10, "l_max": 8, "permutation": "none"},  # Larger cutoff for long-range
        {"r_cut": 4.0, "n_max": 6, "l_max": 4, "permutation": "none"},   # Smaller, faster for local
        {"r_cut": 6.0, "n_max": 12, "l_max": 8, "permutation": "none"},  # Higher resolution
    ])
    
    sine_permutations: List[str] = field(default_factory=lambda: [
        "sorted_l2",      # Primary - best performing
        "eigenspectrum",  # Alternative eigenvalue ordering  
        "random"          # Random permutation for robustness
    ])
    
    # Mixture of Experts configuration
    use_mixture_of_experts: bool = True
    moe_threshold_method: str = "solubility_cutoff"  # "median", "percentile", "kmeans", "solubility_cutoff"
    moe_threshold_value: float = 0.5  # For percentile method (ignored for solubility_cutoff)
    
    # OpenCOSMO and solvent features
    use_open_cosmo: bool = True
    test_without_open_cosmo: bool = True  # Test both with and without OpenCOSMO
    
    # Model configurations
    models: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "random_forest", "neural_network"
    ])
    
    # Dimensionality reduction
    pca_configs: List[Dict] = field(default_factory=lambda: [
        {"use_pca": False},
        {"use_pca": True, "n_components": 100},
        {"use_pca": True, "n_components": 200},
        {"use_pca": True, "n_components": 500},
    ])
    
    # Cross-validation options
    outer_folds: int = 5
    inner_folds: int = 3
    search_iterations: int = 50
    
    # Drip-feeding cross-validation
    use_drip_cv: bool = True
    drip_strategies: List[str] = field(default_factory=lambda: [
        "standard_cv", "drip_solvent", "drip_solute"
    ])
    
    # Clustering and gating
    use_clustering: bool = True
    clustering_methods: List[str] = field(default_factory=lambda: [
        "none", "median_gate", "percentile_gate", "kmeans"
    ])
    
    # Visualization
    create_visualizations: bool = True
    use_gpu_viz: bool = USE_CUML
    tsne_perplexity: int = 30
    umap_neighbors: int = 15
    
    # Output
    output_dir: str = "enhanced_solubility_analysis"
    save_models: bool = True
    create_report: bool = True
    
    # Performance
    use_gpu: bool = USE_CUML
    n_jobs: int = N_JOBS
    verbose: int = 1
    
    # Regularization Control
    regularization_strength: str = "medium"  # "light", "medium", "strong", "very_strong"
    enable_regularized_models: bool = True  # Enable Ridge, Lasso, ElasticNet, SVR, BayesianRidge
    
    # Feature Selection for Regularization
    use_feature_selection: bool = True  # Enable feature selection
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        "variance", "lasso_selection"  # "variance", "univariate", "lasso_selection", "rfe"
    ])
    variance_threshold: float = 0.01  # Threshold for variance-based feature selection

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

# ═════════════════════════ OPTIMIZED DSCRIBE DESCRIPTORS ═══════════════════════════════
SPECIES = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
N_ATOMS_MAX = 200
SINE_EIGS = 50
SOAP_DIM = None
_soap_, _sine_, _mbtr_, _coulomb_ = None, None, None, None

def _get_soap(r_cut=6.0, n_max=8, l_max=6, permutation="none"):
    """Get SOAP descriptor with customizable parameters"""
    global _soap_, SOAP_DIM
    
    # Create unique key for different parameter combinations
    soap_key = f"soap_{r_cut}_{n_max}_{l_max}_{permutation}"
    
    if soap_key not in globals():
        # Enhanced SOAP configurations
        if permutation == "none":
            soap_desc = SOAP(species=SPECIES, r_cut=r_cut,
                           n_max=n_max, l_max=l_max,
                           periodic=False, average="inner")
        else:
            # For permutation-aware SOAP (if needed)
            soap_desc = SOAP(species=SPECIES, r_cut=r_cut,
                           n_max=n_max, l_max=l_max,
                           periodic=False, average="inner")
        
        globals()[soap_key] = soap_desc
        globals()[f"{soap_key}_dim"] = soap_desc.get_number_of_features()
    
    return globals()[soap_key]

def _get_sine(permutation="sorted_l2"):
    """Get Sine Matrix with permutation options"""
    sine_key = f"sine_{permutation}"
    
    if sine_key not in globals():
        globals()[sine_key] = SineMatrix(n_atoms_max=N_ATOMS_MAX,
                                       permutation=permutation, sparse=False)
    return globals()[sine_key]

def _get_mbtr():
    global _mbtr_
    if _mbtr_ is None:
        _mbtr_ = MBTR(
            species=SPECIES,
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 50, "sigma": 0.1},  # Reduced for speed
            weighting={"function": "exp", "scale": 0.5, "cutoff": 1e-3, "threshold": 1e-3},
            periodic=False,
            sparse=False
        )
    return _mbtr_

def _get_coulomb():
    global _coulomb_
    if _coulomb_ is None:
        _coulomb_ = CoulombMatrix(
            n_atoms_max=N_ATOMS_MAX,
            permutation="sorted_l2",
            sparse=False
        )
    return _coulomb_

class DScribeDescriptorSuite:
    """Optimized DScribe descriptor calculator"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.descriptor_names = config.dscribe_descriptors if config.use_dscribe else []
    
    def _mol_to_atoms(self, mol):
        """Convert RDKit molecule to ASE Atoms object with explicit hydrogens"""
        if mol is None:
            return None
        
        try:
            # Add explicit hydrogens
            mol_with_h = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol_with_h, randomSeed=SEED) == -1:
                # If embedding fails, try without optimization
                if AllChem.EmbedMolecule(mol_with_h, randomSeed=SEED, useRandomCoords=True) == -1:
                    return None
            
            # Optimize geometry
            try:
                AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=200)
            except:
                pass  # Continue even if optimization fails
            
            # Extract atomic information
            conf = mol_with_h.GetConformer()
            positions = conf.GetPositions()
            symbols = [atom.GetSymbol() for atom in mol_with_h.GetAtoms()]
            
            # Filter to known species only
            valid_indices = [i for i, s in enumerate(symbols) if s in SPECIES]
            if not valid_indices:
                return None
            
            filtered_symbols = [symbols[i] for i in valid_indices]
            filtered_positions = positions[valid_indices]
            
            # Create ASE Atoms object with proper cell
            atoms = Atoms(symbols=filtered_symbols, positions=filtered_positions)
            
            # Set a large enough cell to avoid periodicity issues
            # This is crucial for DScribe to work properly
            max_coord = np.max(np.abs(filtered_positions)) + 10.0
            atoms.set_cell([max_coord*2, max_coord*2, max_coord*2])
            atoms.set_pbc([False, False, False])  # Non-periodic
            
            return atoms
            
        except Exception as e:
            if self.config.verbose > 1:
                print(f"Warning: Failed to convert molecule to atoms: {e}")
            return None
    
    def calculate_single(self, mol):
        """Calculate all DScribe descriptors for a single molecule with multiple configurations"""
        atoms = self._mol_to_atoms(mol)
        if atoms is None:
            return None
        
        features = {}
        
        try:
            # SOAP descriptor with multiple configurations
            if "soap" in self.descriptor_names:
                for i, soap_config in enumerate(self.config.soap_configs):
                    soap_desc = _get_soap(**soap_config)
                    soap_vec = soap_desc.create(atoms).astype(np.float32)
                    features[f'soap_config_{i}'] = soap_vec
            
            # MBTR descriptor (if enabled)
            if "mbtr" in self.descriptor_names:
                mbtr_desc = _get_mbtr()
                mbtr_vec = mbtr_desc.create(atoms)
                if mbtr_vec.ndim > 1:
                    mbtr_vec = mbtr_vec.flatten()
                features['mbtr'] = mbtr_vec.astype(np.float32)
            
            # Sine Matrix with multiple permutations
            if "sine_matrix" in self.descriptor_names:
                for perm in self.config.sine_permutations:
                    try:
                        sine_desc = _get_sine(permutation=perm)
                        sine_matrix = sine_desc.create(atoms)
                        
                        # Handle different output formats from Sine Matrix
                        if sine_matrix.ndim == 1:
                            eigenvals = sine_matrix
                        else:
                            eigenvals = np.linalg.eigvalsh(sine_matrix)
                        
                        # Sort descending and filter
                        eigenvals = np.sort(eigenvals)[::-1]
                        eigenvals = eigenvals[eigenvals > 1e-6]
                        
                        # Pad or truncate to fixed size
                        if len(eigenvals) > SINE_EIGS:
                            eigenvals = eigenvals[:SINE_EIGS]
                        else:
                            padded = np.zeros(SINE_EIGS)
                            padded[:len(eigenvals)] = eigenvals
                            eigenvals = padded
                        
                        features[f'sine_matrix_{perm}'] = eigenvals.astype(np.float32)
                    except Exception as e:
                        if self.config.verbose > 1:
                            print(f"Warning: Failed to calculate Sine Matrix with {perm}: {e}")
            
            # Coulomb Matrix (if enabled)
            if "coulomb_matrix" in self.descriptor_names:
                coulomb_desc = _get_coulomb()
                coulomb_vec = coulomb_desc.create(atoms)
                if coulomb_vec.ndim > 1:
                    coulomb_vec = coulomb_vec.flatten()
                features['coulomb_matrix'] = coulomb_vec.astype(np.float32)
                
        except Exception as e:
            if self.config.verbose > 1:
                print(f"Warning: Failed to calculate descriptors: {e}")
            return None
        
        return features if features else None
    
    def calculate_batch(self, smiles_list):
        """Calculate DScribe descriptors for a batch of SMILES"""
        print(f"Calculating DScribe descriptors for {len(smiles_list)} molecules...")
        
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        
        with ProcessPoolExecutor(max_workers=min(self.config.n_jobs, 8)) as executor:
            results = list(tqdm(
                executor.map(self.calculate_single, mols),
                total=len(mols),
                desc="DScribe descriptors"
            ))
        
        # Organize results by descriptor type
        descriptor_arrays = {}
        valid_indices = []
        
        for i, result in enumerate(results):
            if result is not None and all(v is not None for v in result.values()):
                valid_indices.append(i)
                for desc_name, desc_vec in result.items():
                    if desc_name not in descriptor_arrays:
                        descriptor_arrays[desc_name] = []
                    descriptor_arrays[desc_name].append(desc_vec)
        
        # Convert to numpy arrays
        final_arrays = {}
        for desc_name, vectors in descriptor_arrays.items():
            if vectors:
                final_arrays[desc_name] = np.vstack(vectors)
        
        print(f"Successfully calculated DScribe descriptors for {len(valid_indices)} molecules")
        return final_arrays, valid_indices

# ═════════════════════════ ENHANCED FINGERPRINT CALCULATOR ═══════════════════════════════
class EnhancedFingerprintCalculator:
    """Enhanced fingerprint calculator with multiple types"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.dscribe_suite = DScribeDescriptorSuite(config) if config.use_dscribe else None
    
    def _calculate_morgan(self, mol, n_bits=2048, radius=3):
        """Calculate Morgan fingerprints with explicit hydrogens"""
        # Add explicit hydrogens for consistency
        mol_with_h = Chem.AddHs(mol)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius, n_bits, True)
        fp = generator.GetFingerprint(mol_with_h)
        return np.frombuffer(fp.ToBitString().encode(), 'c').view(np.uint8) - 48
    
    def _calculate_avalon(self, mol, n_bits=2048):
        """Calculate Avalon fingerprints with explicit hydrogens"""
        mol_with_h = Chem.AddHs(mol)
        fp = GetAvalonFP(mol_with_h, nBits=n_bits)
        return np.frombuffer(fp.ToBitString().encode(), 'c').view(np.uint8) - 48
    
    def _calculate_torsion(self, mol, n_bits=2048):
        """Calculate topological torsion fingerprints with explicit hydrogens"""
        mol_with_h = Chem.AddHs(mol)
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
        fp = generator.GetFingerprint(mol_with_h)
        return np.frombuffer(fp.ToBitString().encode(), 'c').view(np.uint8) - 48
    
    def _calculate_rdkit_descriptors(self, mol):
        """Calculate RDKit molecular descriptors with explicit hydrogens"""
        mol_with_h = Chem.AddHs(mol)
        descriptors = []
        descriptor_names = [name for name, _ in Descriptors.descList]
        
        for name in descriptor_names:
            try:
                desc_fn = getattr(Descriptors, name)
                value = desc_fn(mol_with_h)
                # Handle NaN and infinite values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                descriptors.append(float(value))
            except:
                descriptors.append(0.0)
        
        return np.array(descriptors, dtype=np.float32)
    
    def calculate_fingerprint_batch(self, smiles_list, fp_config):
        """Calculate fingerprints for a batch of SMILES"""
        fp_type = fp_config["type"]
        print(f"Calculating {fp_type} fingerprints for {len(smiles_list)} molecules...")
        
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        valid_mols = [(i, mol) for i, mol in enumerate(mols) if mol is not None]
        
        if not valid_mols:
            raise ValueError("No valid molecules found")
        
        # Calculate fingerprints in parallel
        if fp_type == "morgan":
            def calc_fp(mol):
                try:
                    return self._calculate_morgan(mol, fp_config["n_bits"], fp_config["radius"])
                except:
                    return None
        
        elif fp_type == "combined":
            def calc_fp(mol):
                try:
                    mol_with_h = Chem.AddHs(mol)
                    morgan = self._calculate_morgan(mol, fp_config["n_bits"], fp_config["radius"])
                    avalon = self._calculate_avalon(mol, fp_config["n_bits"])
                    torsion = self._calculate_torsion(mol, fp_config["n_bits"])
                    
                    # Add basic descriptors with explicit hydrogens
                    basic_desc = np.array([
                        Descriptors.MolWt(mol_with_h),
                        Descriptors.MolLogP(mol_with_h),
                        Descriptors.TPSA(mol_with_h),
                        Descriptors.NumHAcceptors(mol_with_h),
                        Descriptors.NumHDonors(mol_with_h),
                        Descriptors.NumRotatableBonds(mol_with_h),
                        Descriptors.NumAromaticRings(mol_with_h),
                        Descriptors.FractionCSP3(mol_with_h)
                    ], dtype=np.float32)
                    
                    return np.hstack([morgan, avalon, torsion, basic_desc])
                except:
                    return None
        
        elif fp_type == "rdkit_descriptors":
            def calc_fp(mol):
                try:
                    return self._calculate_rdkit_descriptors(mol)
                except:
                    return None
        
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        # Calculate in parallel
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            fingerprints = list(tqdm(
                executor.map(calc_fp, [mol for _, mol in valid_mols]),
                total=len(valid_mols),
                desc=f"{fp_type} fingerprints"
            ))
        
        # Filter valid fingerprints
        valid_fps = [fp for fp in fingerprints if fp is not None]
        if not valid_fps:
            raise ValueError("No valid fingerprints calculated")
        
        return np.vstack(valid_fps)
    
    def build_feature_matrix(self, solute_smiles_list, df, use_open_cosmo=True):
        """Build comprehensive feature matrix with BOTH solute and solvent molecular features
        
        Note: Uses SMILES for molecular descriptors/fingerprints but preserves names for analysis
        """
        all_features = []
        feature_names = []
        
        # Extract solvent SMILES from the dataframe (for molecular descriptors)
        solvent_smiles_list = df[self.config.solvent_col].tolist()
        
        print(f"Building features for {len(solute_smiles_list)} solute-solvent pairs...")
        print(f"Unique solutes: {len(set(solute_smiles_list))}")
        print(f"Unique solvents: {len(set(solvent_smiles_list))}")
        print(f"Using SMILES for molecular descriptors, names preserved for analysis")
        
        # Calculate SOLUTE fingerprints for all configurations
        for fp_config in self.config.fp_configs:
            try:
                print(f"Calculating SOLUTE {fp_config['type']} fingerprints...")
                fp_matrix = self.calculate_fingerprint_batch(solute_smiles_list, fp_config)
                all_features.append(fp_matrix)
                
                # Generate feature names
                fp_type = fp_config["type"]
                if fp_type == "rdkit_descriptors":
                    names = [f"solute_rdkit_{name}" for name, _ in Descriptors.descList]
                else:
                    n_features = fp_matrix.shape[1]
                    names = [f"solute_{fp_type}_{i}" for i in range(n_features)]
                feature_names.extend(names)
                
                print(f"Added {fp_matrix.shape[1]} SOLUTE features from {fp_type}")
            except Exception as e:
                print(f"Warning: Failed to calculate solute {fp_config['type']}: {e}")
        
        # Calculate SOLVENT fingerprints for all configurations
        for fp_config in self.config.fp_configs:
            try:
                print(f"Calculating SOLVENT {fp_config['type']} fingerprints...")
                fp_matrix = self.calculate_fingerprint_batch(solvent_smiles_list, fp_config)
                all_features.append(fp_matrix)
                
                # Generate feature names
                fp_type = fp_config["type"]
                if fp_type == "rdkit_descriptors":
                    names = [f"solvent_rdkit_{name}" for name, _ in Descriptors.descList]
                else:
                    n_features = fp_matrix.shape[1]
                    names = [f"solvent_{fp_type}_{i}" for i in range(n_features)]
                feature_names.extend(names)
                
                print(f"Added {fp_matrix.shape[1]} SOLVENT features from {fp_type}")
            except Exception as e:
                print(f"Warning: Failed to calculate solvent {fp_config['type']}: {e}")
        
        # Add SOLUTE DScribe descriptors if enabled
        if self.config.use_dscribe and self.dscribe_suite:
            try:
                print(f"Calculating SOLUTE DScribe descriptors: {self.config.dscribe_descriptors}")
                dscribe_results, valid_indices = self.dscribe_suite.calculate_batch(solute_smiles_list)
                
                for desc_name, desc_matrix in dscribe_results.items():
                    all_features.append(desc_matrix)
                    names = [f"solute_dscribe_{desc_name}_{i}" for i in range(desc_matrix.shape[1])]
                    feature_names.extend(names)
                    print(f"Added {desc_matrix.shape[1]} SOLUTE features from DScribe {desc_name}")
            except Exception as e:
                print(f"Warning: Failed to calculate solute DScribe descriptors: {e}")
        
        # Add SOLVENT DScribe descriptors if enabled
        if self.config.use_dscribe and self.dscribe_suite:
            try:
                print(f"Calculating SOLVENT DScribe descriptors: {self.config.dscribe_descriptors}")
                dscribe_results, valid_indices = self.dscribe_suite.calculate_batch(solvent_smiles_list)
                
                for desc_name, desc_matrix in dscribe_results.items():
                    all_features.append(desc_matrix)
                    names = [f"solvent_dscribe_{desc_name}_{i}" for i in range(desc_matrix.shape[1])]
                    feature_names.extend(names)
                    print(f"Added {desc_matrix.shape[1]} SOLVENT features from DScribe {desc_name}")
            except Exception as e:
                print(f"Warning: Failed to calculate solvent DScribe descriptors: {e}")
        
        # Add precomputed solvent features with OpenCOSMO control
        precomputed_solvent_features = self._extract_solvent_features(df, use_open_cosmo)
        if precomputed_solvent_features is not None:
            all_features.append(precomputed_solvent_features)
            solvent_names = [f"precomputed_solvent_{i}" for i in range(precomputed_solvent_features.shape[1])]
            feature_names.extend(solvent_names)
            print(f"Added {precomputed_solvent_features.shape[1]} precomputed solvent features (OpenCOSMO: {use_open_cosmo})")
        
        if not all_features:
            raise ValueError("No features calculated successfully")
        
        # Combine all features with feature engineering
        feature_matrix = np.hstack(all_features).astype(np.float32)
        
        #Add feature engineering: polynomial features for key descriptors
        if len(all_features) > 2:  # Only if we have multiple feature types
            # Create interaction features between first two feature types
            feat1 = all_features[0][:, :min(50, all_features[0].shape[1])]  # First 120 features
            feat2 = all_features[1][:, :min(50, all_features[1].shape[1])]  # First 120 features
            
            # Element-wise multiplication (interaction)
            interaction_features = feat1 * feat2
            feature_matrix = np.hstack([feature_matrix, interaction_features])
            
            # Add interaction feature names
            interaction_names = [f"interaction_{i}" for i in range(interaction_features.shape[1])]
            feature_names.extend(interaction_names)
            
            print(f"Added {interaction_features.shape[1]} interaction features")
        
        print(f"Final feature matrix shape: {feature_matrix.shape}")
        print(f"Features breakdown:")
        print(f"  - Solute molecular features: {sum(1 for name in feature_names if name.startswith('solute_'))}")
        print(f"  - Solvent molecular features: {sum(1 for name in feature_names if name.startswith('solvent_') and not name.startswith('precomputed_'))}")
        print(f"  - Precomputed solvent features: {sum(1 for name in feature_names if name.startswith('precomputed_'))}")
        
        return feature_matrix, feature_names
    
    def _extract_solvent_features(self, df, use_open_cosmo=True):
        """Extract solvent features from dataframe with OpenCOSMO control"""
        # Select columns based on OpenCOSMO flag
        if use_open_cosmo:
            solvent_cols = SOLVENT_COLS  # Includes both OpenCOSMO and precomputed features
        else:
            solvent_cols = PRECOMPUTED_SOLVENT_FEATURES  # Only precomputed features
        
        # Check available columns
        available_cols = [col for col in solvent_cols if col in df.columns]
        if not available_cols:
            print(f"Warning: No solvent features found (use_open_cosmo={use_open_cosmo})")
            return None
        
        print(f"Using {len(available_cols)} solvent features out of {len(solvent_cols)} requested")
        
        solvent_features = df[available_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        return solvent_features.values.astype(np.float32)

# ═════════════════════════ ROBUST PCA CLASS ═════════════════════════════════════
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

# ═════════════════════════ MIXTURE OF EXPERTS ══════════════════════════════════════════
# Prefer GPU-accelerated implementations when available, but fall back to sklearn
try:
    from cuml.linear_model import LogisticRegression
    from cuml import KMeans
except Exception:  # cuml not available
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans

class MixtureOfExpertsRegressor(BaseEstimator, TransformerMixin):
    """Mixture of Experts with gating network for high/low solubility prediction"""
    
    def __init__(self, low_model, high_model, threshold_method="median", threshold_value=0.25, verbose=False):
        self.low_model = low_model
        self.high_model = high_model
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.verbose = verbose
        self.gating_network = LogisticRegression(
            max_iter=5000,  # Increased iterations
            tol=1e-4,       # Better tolerance
            solver='qn',    # Quasi-Newton solver
            C=1.0)          # Regularization
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

# ═════════════════════════ ENHANCED MODEL SUITE ═══════════════════════════════
class EnhancedModelSuite:
    """Comprehensive model comparison suite"""
    
    def __init__(self, config: EnhancedConfig):
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

# ═════════════════════════ DRIP-FEEDING CROSS-VALIDATION ═══════════════════════════════
class DripFeedingCV:
    """Drip-feeding cross-validation for solvent and solute-based splits"""
    
    def __init__(self, strategy="drip_solvent", min_test_size=10):
        self.strategy = strategy
        self.min_test_size = min_test_size
    
    def split(self, X, y=None, groups=None):
        """Generate train/test splits based on drip-feeding strategy"""
        if groups is None:
            raise ValueError("Groups must be provided for drip-feeding CV")
        
        if self.strategy == "standard_cv":
            # Standard K-fold
            kf = KFold(n_splits=5, shuffle=True, #random_state=SEED
                       )
            yield from kf.split(X, y)
            
        elif self.strategy == "drip_solvent":
            yield from self._drip_solvent_splits(groups)
            
        elif self.strategy == "drip_solute":
            yield from self._drip_solute_splits(groups)
            
        else:
            raise ValueError(f"Unknown drip strategy: {self.strategy}")
    
    def _drip_solvent_splits(self, solvents):
        """Generate progressive drip-feeding splits based on solvents"""
        # Get unique solvents ordered by frequency (most frequent first)
        solvent_counts = pd.Series(solvents).value_counts()
        sorted_solvents = solvent_counts.index.tolist()
        
        # Progressive building: start with top solvents, add one at a time
        for i in range(2, len(sorted_solvents)):  # Start with at least 2 solvents
            # Training set: first i solvents (cumulative)
            train_solvents = set(sorted_solvents[:i])
            # Test set: next solvent (the one we're adding)
            test_solvent = sorted_solvents[i]
            
            train_indices = [idx for idx, solv in enumerate(solvents) if solv in train_solvents]
            test_indices = [idx for idx, solv in enumerate(solvents) if solv == test_solvent]
            
            # Only yield if we have enough data
            if len(train_indices) >= self.min_test_size and len(test_indices) >= self.min_test_size:
                yield np.array(train_indices), np.array(test_indices)
    
    def _drip_solute_splits(self, solutes):
        """Generate progressive drip-feeding splits based on solutes"""
        # Get unique solutes ordered by frequency (most frequent first)
        solute_counts = pd.Series(solutes).value_counts()
        sorted_solutes = solute_counts.index.tolist()
        
        # Progressive building approach
        min_train_solutes = max(10, len(sorted_solutes) // 10)  # Start with 10% of solutes
        step_size = max(1, len(sorted_solutes) // 15)  # Add ~6-7% at a time
        
        for i in range(min_train_solutes, len(sorted_solutes), step_size):
            # Training set: first i solutes (cumulative)
            train_solutes = set(sorted_solutes[:i])
            # Test set: next step_size solutes
            end_idx = min(i + step_size, len(sorted_solutes))
            test_solutes = set(sorted_solutes[i:end_idx])
            
            train_indices = [idx for idx, sol in enumerate(solutes) if sol in train_solutes]
            test_indices = [idx for idx, sol in enumerate(solutes) if sol in test_solutes]
            
            # Only yield if we have enough data
            if len(train_indices) >= self.min_test_size and len(test_indices) >= self.min_test_size:
                yield np.array(train_indices), np.array(test_indices)

# ═════════════════════════ ENHANCED ANALYSIS ENGINE ═══════════════════════════════
class EnhancedSolubilityAnalyzer:
    """Main analysis engine with comprehensive features"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.fingerprint_calc = EnhancedFingerprintCalculator(config)
        self.model_suite = EnhancedModelSuite(config)
        self.results = {}
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"{config.output_dir}_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def run_comprehensive_analysis(self):
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
        
        # Analyze results
        self.analyze_and_report_results_enhanced(all_results)
        
        return all_results
    
    def analyze_and_report_results(self, all_results, fold_results):
        """Comprehensive result analysis and reporting"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                **{k: v for k, v in result.items() if k not in ['predictions', 'true_values', 'test_indices', 'best_estimator']},
                **{f"metric_{k}": v for k, v in result['metrics'].items()}
            }
            for result in all_results
        ])
        
        # Overall statistics by model
        model_stats = results_df.groupby('model').agg({
            'metric_rmse': ['mean', 'std', 'min'],
            'metric_r2': ['mean', 'std', 'max'],
            'metric_mae': ['mean', 'std', 'min'],
            'metric_mape': ['mean', 'std', 'min']
        }).round(4)
        
        print("\nModel Performance Summary:")
        print(model_stats)
        
        # PCA impact analysis
        pca_impact = results_df.groupby(['model', 'pca_config']).agg({
            'metric_rmse': 'mean',
            'metric_r2': 'mean'
        }).round(4)
        
        print(f"\nPCA Impact Analysis:")
        print(pca_impact)
        
        # Solvent-specific analysis
        solvent_analysis = self.analyze_by_solvent(all_results)
        
        # Feature importance analysis
        self.analyze_feature_importance(fold_results)
        
        # Create visualizations
        if self.config.create_visualizations:
            self.create_comprehensive_visualizations(all_results, fold_results)
        
        # Save detailed results
        results_df.to_csv(self.output_dir / "detailed_results.csv", index=False)
        
        # Save summary statistics
        summary_stats = {
            'overall_best_rmse': float(results_df['metric_rmse'].min()),
            'overall_best_r2': float(results_df['metric_r2'].max()),
            'model_comparison': model_stats.to_dict(),
            'pca_impact': pca_impact.to_dict(),
            'total_configurations_tested': len(results_df),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Generate comprehensive report
        if self.config.create_report:
            self.generate_comprehensive_report(summary_stats, solvent_analysis)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
    
    def analyze_and_report_results_enhanced(self, all_results):
        """Enhanced result analysis with OpenCOSMO and drip-feeding comparison"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ENHANCED RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                **{k: v for k, v in result.items() if k not in ['predictions', 'true_values', 'test_indices', 'best_estimator', 'feature_names']},
                **{f"metric_{k}": v for k, v in result['metrics'].items()}
            }
            for result in all_results
        ])
        
        # MoE Analysis - Show detailed info for best MoE models
        moe_results = [r for r in all_results if 'moe' in r['model']]
        if moe_results:
            print("\n" + "="*60)
            print("MIXTURE OF EXPERTS ANALYSIS")
            print("="*60)
            
            for result in moe_results:
                if 'best_estimator' in result:
                    try:
                        estimator = result['best_estimator']
                        if hasattr(estimator, 'named_steps') and 'model' in estimator.named_steps:
                            moe_model = estimator.named_steps['model']
                            if isinstance(moe_model, MixtureOfExpertsRegressor):
                                print(f"\nMoE Model: {result['model']} (Fold {result['fold']})")
                                print(f"  Matrix: {result['matrix_type']}, CV: {result['cv_strategy']}")
                                print(f"  Performance: RMSE={result['metrics']['rmse']:.4f}, R²={result['metrics']['r2']:.4f}")
                                if hasattr(moe_model, 'threshold') and moe_model.threshold is not None:
                                    if moe_model.threshold_method == "solubility_cutoff":
                                        print(f"  Solubility Threshold: logS = {moe_model.threshold:.1f}")
                                        # Force a quick summary fit to show threshold info
                                        if len(result['true_values']) > 0:
                                            gate_labels = (result['true_values'] > moe_model.threshold).astype(int)
                                            insoluble_count = (gate_labels == 0).sum()
                                            soluble_count = (gate_labels == 1).sum()
                                            total = len(gate_labels)
                                            print(f"  Split: Insoluble={insoluble_count}({insoluble_count/total*100:.0f}%) | Soluble={soluble_count}({soluble_count/total*100:.0f}%)")
                                    else:
                                        print(f"  Threshold: {moe_model.threshold:.3f}")
                                        # Force a quick summary fit to show threshold info
                                        if len(result['true_values']) > 0:
                                            gate_labels = (result['true_values'] > moe_model.threshold).astype(int)
                                            low_count = (gate_labels == 0).sum()
                                            high_count = (gate_labels == 1).sum()
                                            total = len(gate_labels)
                                            print(f"  Split: Low={low_count}({low_count/total*100:.0f}%) | High={high_count}({high_count/total*100:.0f}%)")
                    except Exception as e:
                        print(f"  Error analyzing MoE model: {e}")
        
        # OpenCOSMO impact analysis
        print("\n" + "="*60)
        print("OPENCOSMO IMPACT ANALYSIS")
        print("="*60)
        
        cosmo_comparison = results_df.groupby(['matrix_type', 'model']).agg({
            'metric_rmse': ['mean', 'std'],
            'metric_r2': ['mean', 'std'],
            'metric_mae': ['mean', 'std']
        }).round(4)
        
        print("OpenCOSMO Feature Impact:")
        print(cosmo_comparison)
        
        # CV Strategy comparison
        print("\n" + "="*60)
        print("CROSS-VALIDATION STRATEGY COMPARISON")
        print("="*60)
        
        cv_comparison = results_df.groupby(['cv_strategy', 'model']).agg({
            'metric_rmse': ['mean', 'std'],
            'metric_r2': ['mean', 'std'],
            'metric_mae': ['mean', 'std']
        }).round(4)
        
        print("CV Strategy Comparison:")
        print(cv_comparison)
        
        # Solvent-specific analysis
        results_by_solvent = self.analyze_by_solvent(all_results)
        
        # Overall best results
        print("\n" + "="*60)
        print("BEST OVERALL RESULTS")
        print("="*60)
        
        best_result = results_df.loc[results_df['metric_rmse'].idxmin()]
        print(f"Best Configuration:")
        print(f"  Model: {best_result['model']}")
        print(f"  Matrix: {best_result['matrix_type']}")
        print(f"  CV: {best_result['cv_strategy']}")
        print(f"  PCA: {best_result['pca_config']}")
        print(f"  RMSE: {best_result['metric_rmse']:.4f}")
        print(f"  R²: {best_result['metric_r2']:.4f}")
        
        # Print solvent analysis
        if results_by_solvent:
            print(f"\nSOLVENT ANALYSIS:")
            print("-" * 20)
            valid_solvents = {k: v for k, v in results_by_solvent.items() if not np.isnan(v['rmse'])}
            for solvent, metrics in sorted(valid_solvents.items(), key=lambda x: x[1]['rmse']):
                print(f"{solvent}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        # Save detailed results
        results_df.to_csv(self.output_dir / "enhanced_detailed_results.csv", index=False)
        
        # Create enhanced visualizations
        if self.config.create_visualizations:
            self.create_enhanced_visualizations(all_results, results_df)
        
        # Enhanced summary statistics
        summary_stats = {
            'overall_best_rmse': float(results_df['metric_rmse'].min()),
            'overall_best_r2': float(results_df['metric_r2'].max()),
            'opencosmo_comparison': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in cosmo_comparison.to_dict().items()},
            'cv_strategy_comparison': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in cv_comparison.to_dict().items()},
            'best_configuration': {
                'matrix_type': best_result['matrix_type'],
                'cv_strategy': best_result['cv_strategy'],
                'model': best_result['model'],
                'pca_config': str(best_result['pca_config']),
                'rmse': float(best_result['metric_rmse']),
                'r2': float(best_result['metric_r2'])
            },
            'total_configurations_tested': len(results_df),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "enhanced_summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Generate enhanced report with solvent analysis
        if self.config.create_report:
            self.generate_enhanced_report(summary_stats, results_df, results_by_solvent)
        
        print(f"\nEnhanced analysis complete! Results saved to: {self.output_dir}")
        
        return summary_stats
    
    def analyze_by_solvent(self, all_results):
        """Analyze performance by solvent using names for readability"""
        print(f"\nAnalyzing performance by solvent...")
        
        solvent_results = {}
        
        for result in all_results:
            test_indices = result['test_indices']
            y_true = result['true_values']
            y_pred = result['predictions']
            
            # Get solvent names for test indices (for readable output)
            test_solvent_names = self.df.iloc[test_indices][self.config.solvent_name_col]
            
            for i, solvent_name in enumerate(test_solvent_names):
                if solvent_name not in solvent_results:
                    solvent_results[solvent_name] = {'true': [], 'pred': []}
                
                solvent_results[solvent_name]['true'].append(y_true[i])
                solvent_results[solvent_name]['pred'].append(y_pred[i])
        
        # Calculate metrics by solvent
        solvent_metrics = {}
        for solvent_name, data in solvent_results.items():

            true_vals = np.array(data['true'])
            pred_vals = np.array(data['pred'])
            
            solvent_metrics[solvent_name] = {
                'count': len(true_vals),
                'rmse': rmse(true_vals, pred_vals),
                'r2': r2(true_vals, pred_vals),
                'mae': mae(true_vals, pred_vals)}
                
        
        # Save solvent analysis
        solvent_df = pd.DataFrame.from_dict(solvent_metrics, orient='index')
        if not solvent_df.empty:
            solvent_df.to_csv(self.output_dir / "solvent_analysis.csv")
            print(f"Analyzed {len(solvent_metrics)} solvents with sufficient data")
        
        return solvent_metrics
    
    def analyze_feature_importance(self, fold_results):
        """Analyze feature importance across folds"""
        print(f"\nAnalyzing feature importance...")
        
        importances_by_fold = {}
        
        for fold_idx, fold_result in fold_results.items():
            model = fold_result.get('best_estimator')
            if model and hasattr(model.named_steps.get('model'), 'feature_importances_'):
                importances = model.named_steps['model'].feature_importances_
                importances_by_fold[ fold_idx] = importances
        
        if importances_by_fold:
            # Average importance across folds
            all_importances = np.array(list(importances_by_fold.values()))
            mean_importance = np.mean(all_importances, axis=0)
            std_importance = np.std(all_importances, axis=0)
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(mean_importance)],
                'importance_mean': mean_importance,
                'importance_std': std_importance
            }).sort_values('importance_mean', ascending=False)
            
            importance_df.to_csv(self.output_dir / "feature_importance.csv", index=False)
            print(f"Saved feature importance analysis for {len(importance_df)} features")
    
    def create_comprehensive_visualizations(self, all_results, fold_results):
        """Create comprehensive visualization suite"""
        print(f"\nCreating visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model comparison plot
        self.plot_model_comparison(all_results, viz_dir)
        
        # 2. Predictions vs actual
        self.plot_predictions_vs_actual(all_results, viz_dir)
        
        # 3. Feature importance
        self.plot_feature_importance(viz_dir)
        
        # 4. Solvent analysis
        self.plot_solvent_analysis(viz_dir)
        
        # 5. Dimensionality reduction visualization
        if self.config.use_dscribe:
            self.plot_dimensionality_reduction(viz_dir)
        
        print(f"Visualizations saved to: {viz_dir}")
    
    def plot_model_comparison(self, all_results, viz_dir):
        """Plot model comparison"""
        results_df = pd.DataFrame([
            {
                'model': result['model'],
                'rmse': result['metrics']['rmse'],
                'r2': result['metrics']['r2'],
                'mae': result['metrics']['mae']
            }
            for result in all_results
        ])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RMSE comparison
        sns.boxplot(data=results_df, x='model', y='rmse', ax=axes[0])
        axes[0].set_title('RMSE by Model')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R² comparison  
        sns.boxplot(data=results_df, x='model', y='r2', ax=axes[1])
        axes[1].set_title('R² by Model')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        sns.boxplot(data=results_df, x='model', y='mae', ax=axes[2])
        axes[2].set_title('MAE by Model')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions_vs_actual(self, all_results, viz_dir):
        """Plot predictions vs actual values"""
        all_true = []
        all_pred = []
        
        for result in all_results:
            all_true.extend(result['true_values'])
            all_pred.extend(result['predictions'])
        
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(all_true, all_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs True Values (All Models)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        overall_r2 = r2(all_true, all_pred)
        overall_rmse = rmse(all_true, all_pred)
        plt.text(0.05, 0.95, f'R² = {overall_r2:.3f}\nRMSE = {overall_rmse:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, viz_dir):
        """Plot feature importance if available"""
        importance_file = self.output_dir / "feature_importance.csv"
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
            
            # Plot top 30 features
            top_features = importance_df.head(30)
            
            plt.figure(figsize=(12, 10))
            plt.barh(range(len(top_features)), top_features['importance_mean'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 30 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(viz_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_solvent_analysis(self, viz_dir):
        """Plot solvent analysis if available"""
        solvent_file = self.output_dir / "solvent_analysis.csv"
        if solvent_file.exists():
            solvent_df = pd.read_csv(solvent_file, index_col=0)
            
            if len(solvent_df) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # RMSE by solvent
                solvent_df_sorted = solvent_df.sort_values('rmse')
                axes[0,0].bar(range(len(solvent_df_sorted)), solvent_df_sorted['rmse'])
                axes[0,0].set_title('RMSE by Solvent')
                axes[0,0].set_xticks(range(len(solvent_df_sorted)))

                axes[0,0].set_xticklabels(solvent_df_sorted.index, rotation=45, ha='right')
                
                # R² by solvent
                axes[0,1].bar(range(len(solvent_df_sorted)), solvent_df_sorted['r2'])
                axes[0,1].set_title('R² by Solvent')
                axes[0,1].set_xticks(range(len(solvent_df_sorted)))
                axes[0,1].set_xticklabels(solvent_df_sorted.index, rotation=45, ha='right')
                
                # Count by solvent
                axes[1,0].bar(range(len(solvent_df_sorted)), solvent_df_sorted['count'])
                axes[1,0].set_title('Prediction Count by Solvent')
                axes[1,0].set_xticks(range(len(solvent_df_sorted)))
                axes[1,0].set_xticklabels(solvent_df_sorted.index, rotation=45, ha='right')
                
                # Scatter plot
                axes[1,1].scatter(solvent_df['r2'], solvent_df['rmse'], 
                                s=solvent_df['count']*2, alpha=0.7)
                axes[1,1].set_xlabel('R²')
                axes[1,1].set_ylabel('RMSE')
                axes[1,1].set_title('RMSE vs R² (size = count)')
                
                plt.tight_layout()
                plt.savefig(viz_dir / "solvent_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def plot_dimensionality_reduction(self, viz_dir):
        """Plot dimensionality reduction visualization"""
        if len(self.X) > 1000:  # Sample for visualization
            indices = np.random.choice(len(self.X), 1000, replace=False)
            X_sample = self.X[indices]
            y_sample = self.y[indices]
        else:
            X_sample = self.X
            y_sample = self.y
        
        # t-SNE
        if self.config.use_gpu_viz and USE_CUML:
            tsne = CuTSNE(n_components=2, perplexity=self.config.tsne_perplexity, 
                         #random_state=SEED
                         )
            X_tsne = tsne.fit_transform(X_sample)
            if hasattr(X_tsne, 'get'):  # Convert from CuPy if needed
                X_tsne = X_tsne.get()
        else:
            tsne = TSNE(n_components=2, perplexity=self.config.tsne_perplexity, 
                       #random_state=SEED, 
                       n_jobs=self.config.n_jobs)
            X_tsne = tsne.fit_transform(X_sample)
        
       
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                            cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Solubility')
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # UMAP (if available)
        if USE_UMAP or (self.config.use_gpu_viz and USE_CUML):
            if self.config.use_gpu_viz and USE_CUML:
                umap_model = CuUMAP(n_neighbors=self.config.umap_neighbors, 
                                   #random_state=SEED
                                   )
                X_umap = umap_model.fit_transform(X_sample)
                if hasattr(X_umap, 'get'):
                    X_umap = X_umap.get()
            else:
                umap_model = UMAP(n_neighbors=self.config.umap_neighbors, 
                                 #random_state=SEED
                                 )
                X_umap = umap_model.fit_transform(X_sample)
            
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_sample, 
                                cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Solubility')
            plt.title('UMAP Visualization')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "dimensionality_reduction.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, summary_stats, solvent_analysis):
        """Generate comprehensive Markdown report"""
        report_path = self.output_dir / "comprehensive_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Solubility Prediction Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Dataset:** {self.config.tsv_path}\n")
            f.write(f"- **Target:** {self.config.target_col}\n")
            f.write(f"- **Models tested:** {', '.join(self.config.models)}\n")
            f.write(f"- **DScribe descriptors:** {'Yes' if self.config.use_dscribe else 'No'}\n")
            f.write(f"- **GPU acceleration:** {'Yes' if self.config.use_gpu else 'No'}\n")
            f.write(f"- **Cross-validation:** {self.config.outer_folds}-fold outer, {self.config.inner_folds}-fold inner\n\n")
            
            f.write("## Dataset Summary\n\n")
            f.write(f"- **Total samples:** {len(self.df)}\n")
            f.write(f"- **Features:** {self.X.shape[1]}\n")
            f.write(f"- **Unique solvents:** {self.df[self.config.solvent_col].nunique()}\n")
            f.write(f"- **Target range:** {self.y.min():.3f} to {self.y.max():.3f}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write(f"- **Best RMSE:** {summary_stats['overall_best_rmse']:.4f}\n")
            f.write(f"- **Best R²:** {summary_stats['overall_best_r2']:.4f}\n")
            f.write(f"- **Configurations tested:** {summary_stats['total_configurations_tested']}\n\n")
            
            if solvent_analysis:
                f.write("## Solvent-Specific Performance\n\n")
                f.write("| Solvent | RMSE | R² | MAE | Count |\n")
                f.write("|---------|------|----|----|-------|\n")
                
                sorted_solvents = sorted(solvent_analysis.items(), 
                                       key=lambda x: x[1]['rmse'])
                for solvent, metrics in sorted_solvents[:10]:  # Top 10
                    f.write(f"| {solvent} | {metrics['rmse']:.4f} | "
                           f"{metrics['r2']:.4f} | {metrics['mae']:.4f} | "
                           f"{metrics['count']} |\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `detailed_results.csv` - Complete results for all configurations\n")
            f.write("- `summary_statistics.json` - Summary statistics\n")
            f.write("- `feature_importance.csv` - Feature importance analysis\n")
            f.write("- `solvent_analysis.csv` - Per-solvent performance\n")
            f.write("- `visualizations/` - All plots and charts\n")
            f.write("- `best_model_fold_*.pkl` - Best models from each fold\n")
        
        print(f"Comprehensive report saved to: {report_path}")
    
    def generate_enhanced_report(self, summary_stats, results_df, results_by_solvent=None):
        """Generate enhanced Markdown report"""
        report_path = self.output_dir / "enhanced_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Solubility Prediction Analysis Report (Detailed)\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Dataset:** {self.config.tsv_path}\n")
            f.write(f"- **Target:** {self.config.target_col}\n")
            f.write(f"- **Models tested:** {', '.join(self.config.models)}\n")
            f.write(f"- **DScribe descriptors:** {'Yes' if self.config.use_dscribe else 'No'}\n")
            f.write(f"- **GPU acceleration:** {'Yes' if self.config.use_gpu else 'No'}\n")
            f.write(f"- **Cross-validation:** {self.config.outer_folds}-fold outer, {self.config.inner_folds}-fold inner\n\n")
            
            f.write("## Dataset Summary\n\n")
            f.write(f"- **Total samples:** {len(self.df)}\n")
            f.write(f"- **Features:** {self.X.shape[1]}\n")
            f.write(f"- **Unique solvents:** {self.df[self.config.solvent_col].nunique()}\n")
            f.write(f"- **Target range:** {self.y.min():.3f} to {self.y.max():.3f}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write(f"- **Best RMSE:** {summary_stats['overall_best_rmse']:.4f}\n")
            f.write(f"- **Best R²:** {summary_stats['overall_best_r2']:.4f}\n")
            f.write(f"- **Configurations tested:** {summary_stats['total_configurations_tested']}\n\n")
            
            f.write("## OpenCOSMO Impact Analysis\n\n")
            f.write("| Matrix Type | Model | RMSE (mean ± std) | R² (mean ± std) | MAE (mean ± std) |\n")
            f.write("|-------------|-------|-------------------|-----------------|-----------------|\n")
            
            for (matrix_type, model), group in results_df.groupby(['matrix_type', 'model']):
                rmse_mean = group['metric_rmse'].mean()
                rmse_std = group['metric_rmse'].std()
                r2_mean = group['metric_r2'].mean()
                r2_std = group['metric_r2'].std()
                mae_mean = group['metric_mae'].mean()
                mae_std = group['metric_mae'].std()
                
                f.write(f"| {matrix_type} | {model} | {rmse_mean:.4f} ± {rmse_std:.4f} | "
                       f"{r2_mean:.4f} ± {r2_std:.4f} | {mae_mean:.4f} ± {mae_std:.4f} |\n")
            
            f.write("\n## Cross-Validation Strategy Comparison\n\n")
            f.write("| CV Strategy | Model | RMSE (mean ± std) | R² (mean ± std) | MAE (mean ± std) |\n")
            f.write("|-------------|-------|-------------------|-----------------|-----------------|\n")
            
            for (cv_strategy, model), group in results_df.groupby(['cv_strategy', 'model']):
                rmse_mean = group['metric_rmse'].mean()
                rmse_std = group['metric_rmse'].std()
                r2_mean = group['metric_r2'].mean()
                r2_std = group['metric_r2'].std()
                mae_mean = group['metric_mae'].mean()
                mae_std = group['metric_mae'].std()
                
                f.write(f"| {cv_strategy} | {model} | {rmse_mean:.4f} ± {rmse_std:.4f} | "
                       f"{r2_mean:.4f} ± {r2_std:.4f} | {mae_mean:.4f} ± {mae_std:.4f} |\n")
            
            f.write("\n## Best Overall Configuration\n\n")
            f.write(f"- **Matrix Type:** {summary_stats['best_configuration']['matrix_type']}\n")
            f.write(f"- **CV Strategy:** {summary_stats['best_configuration']['cv_strategy']}\n")
            f.write(f"- **Model:** {summary_stats['best_configuration']['model']}\n")
            f.write(f"- **PCA Config:** {summary_stats['best_configuration']['pca_config']}\n")
            f.write(f"- **RMSE:** {summary_stats['best_configuration']['rmse']:.4f}\n")
            f.write(f"- **R²:** {summary_stats['best_configuration']['r2']:.4f}\n")
            
            # Add solvent analysis section
            if results_by_solvent:
                f.write("\n## Solvent-Specific Performance\n\n")
                f.write("| Solvent | RMSE | R² | MAE | Count |\n")
                f.write("|---------|------|----|----|-------|\n")
                
                valid_solvents = {k: v for k, v in results_by_solvent.items() if not np.isnan(v['rmse'])}
                sorted_solvents = sorted(valid_solvents.items(), key=lambda x: x[1]['rmse'])
                
                for solvent, metrics in sorted_solvents[:20]:  # Top 20 best performing solvents
                    f.write(f"| {solvent} | {metrics['rmse']:.4f} | "
                           f"{metrics['r2']:.4f} | {metrics['mae']:.4f} | "
                           f"{metrics['count']} |\n")
        
        print(f"Enhanced report saved to: {report_path}")
    
    def create_enhanced_visualizations(self, all_results, results_df):
        """Create enhanced visualization suite with MoE and OpenCOSMO analysis"""
        print(f"\nCreating enhanced visualizations...")
        
        viz_dir = self.output_dir / "enhanced_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model comparison with OpenCOSMO breakdown
        self.plot_enhanced_model_comparison(results_df, viz_dir)
        
        # 2. MoE vs Standard model comparison
        self.plot_moe_comparison(results_df, viz_dir)
        
        # 3. OpenCOSMO impact visualization
        self.plot_opencosmo_impact(results_df, viz_dir)
        
        # 4. CV Strategy comparison
        self.plot_cv_strategy_comparison(results_df, viz_dir)
        
        # 5. Performance distribution
        self.plot_performance_distribution(results_df, viz_dir)
        
        # 6. Enhanced predictions vs actual
        self.plot_enhanced_predictions_vs_actual(all_results, viz_dir)
        
        print(f"Enhanced visualizations saved to: {viz_dir}")
    
    def plot_enhanced_model_comparison(self, results_df, viz_dir):
        """Enhanced model comparison with OpenCOSMO breakdown"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE by model and matrix type
        sns.boxplot(data=results_df, x='model', y='metric_rmse', hue='matrix_type', ax=axes[0,0])
        axes[0,0].set_title('RMSE by Model and Matrix Type')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend(title='Matrix Type')
        
        # R² by model and matrix type
        sns.boxplot(data=results_df, x='model', y='metric_r2', hue='matrix_type', ax=axes[0,1])
        axes[0,1].set_title('R² by Model and Matrix Type')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(title='Matrix Type')
        
        # MAE by model and matrix type
        sns.boxplot(data=results_df, x='model', y='metric_mae', hue='matrix_type', ax=axes[1,0])
        axes[1,0].set_title('MAE by Model and Matrix Type')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Matrix Type')
        
        # Performance scatter plot
        sns.scatterplot(data=results_df, x='metric_rmse', y='metric_r2', 
                       hue='model', style='matrix_type', s=100, ax=axes[1,1])
        axes[1,1].set_title('RMSE vs R² Performance')
        axes[1,1].set_xlabel('RMSE')
        axes[1,1].set_ylabel('R²')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "enhanced_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_moe_comparison(self, results_df, viz_dir):
        """Compare MoE vs standard models"""
        # Separate MoE and standard models
        moe_results = results_df[results_df['model'].str.contains('moe', case=False)]
        standard_results = results_df[~results_df['model'].str.contains('moe', case=False)]
        
        if len(moe_results) > 0 and len(standard_results) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # RMSE comparison
            moe_rmse = moe_results['metric_rmse'].values
            std_rmse = standard_results['metric_rmse'].values
            
            axes[0].boxplot([std_rmse, moe_rmse], labels=['Standard', 'MoE'])
            axes[0].set_title('RMSE: Standard vs MoE')
            axes[0].set_ylabel('RMSE')
            
            # R² comparison
            moe_r2 = moe_results['metric_r2'].values
            std_r2 = standard_results['metric_r2'].values
            
            axes[1].boxplot([std_r2, moe_r2], labels=['Standard', 'MoE'])
            axes[1].set_title('R²: Standard vs MoE')
            axes[1].set_ylabel('R²')
            
            # Scatter plot comparison
            axes[2].scatter(std_rmse, std_r2, label='Standard', alpha=0.7, s=60)
            axes[2].scatter(moe_rmse, moe_r2, label='MoE', alpha=0.7, s=60)
            axes[2].set_xlabel('RMSE')
            axes[2].set_ylabel('R²')
            axes[2].set_title('Performance Comparison')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "moe_vs_standard_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_opencosmo_impact(self, results_df, viz_dir):
        """Visualize OpenCOSMO feature impact"""
        if 'matrix_type' in results_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # RMSE impact
            cosmo_pivot_rmse = results_df.pivot_table(
                values='metric_rmse', 
                index='model', 
                columns='matrix_type', 
                aggfunc='mean'
            )
            
            cosmo_pivot_rmse.plot(kind='bar', ax=axes[0])
            axes[0].set_title('OpenCOSMO Impact on RMSE')
            axes[0].set_ylabel('RMSE')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].legend(title='Matrix Type')
            
            # R² impact
            cosmo_pivot_r2 = results_df.pivot_table(
                values='metric_r2', 
                index='model', 
                columns='matrix_type', 
                aggfunc='mean'
            )
            
            cosmo_pivot_r2.plot(kind='bar', ax=axes[1])
            axes[1].set_title('OpenCOSMO Impact on R²')
            axes[1].set_ylabel('R²')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend(title='Matrix Type')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "opencosmo_impact.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_cv_strategy_comparison(self, results_df, viz_dir):
        """Compare different CV strategies"""
        if 'cv_strategy' in results_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # RMSE by CV strategy
            sns.boxplot(data=results_df, x='cv_strategy', y='metric_rmse', ax=axes[0])
            axes[0].set_title('RMSE by CV Strategy')
            axes[0].tick_params(axis='x', rotation=45)
            
            # R² by CV strategy
            sns.boxplot(data=results_df, x='cv_strategy', y='metric_r2', ax=axes[1])
            axes[1].set_title('R² by CV Strategy')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "cv_strategy_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_performance_distribution(self, results_df, viz_dir):
        """Plot performance metric distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE distribution
        axes[0,0].hist(results_df['metric_rmse'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('RMSE Distribution')
        axes[0,0].set_xlabel('RMSE')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(results_df['metric_rmse'].mean(), color='red', linestyle='--', label='Mean')
        axes[0,0].legend()
        
        # R² distribution
        axes[0,1].hist(results_df['metric_r2'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('R² Distribution')
        axes[0,1].set_xlabel('R²')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(results_df['metric_r2'].mean(), color='red', linestyle='--', label='Mean')
        axes[0,1].legend()
        
        # MAE distribution
        axes[1,0].hist(results_df['metric_mae'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('MAE Distribution')
        axes[1,0].set_xlabel('MAE')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(results_df['metric_mae'].mean(), color='red', linestyle='--', label='Mean')
        axes[1,0].legend()
        
        # Performance correlation
        axes[1,1].scatter(results_df['metric_rmse'], results_df['metric_r2'], alpha=0.6)
        axes[1,1].set_xlabel('RMSE')
        axes[1,1].set_ylabel('R²')
        axes[1,1].set_title('RMSE vs R² Correlation')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = results_df['metric_rmse'].corr(results_df['metric_r2'])
        axes[1,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                      transform=axes[1,1].transAxes, 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_enhanced_predictions_vs_actual(self, all_results, viz_dir):
        """Enhanced predictions vs actual with model breakdown"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Combined predictions vs actual
        all_true = []
        all_pred = []
        all_models = []
        
        for result in all_results:
            all_true.extend(result['true_values'])
            all_pred.extend(result['predictions'])
            all_models.extend([result['model']] * len(result['true_values']))
        
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        
        # Left plot: All predictions
        axes[0].scatter(all_true, all_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predictions vs True Values (All Models)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        overall_r2 = r2(all_true, all_pred)
        overall_rmse = rmse(all_true, all_pred)
        axes[0].text(0.05, 0.95, f'R² = {overall_r2:.3f}\nRMSE = {overall_rmse:.3f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Right plot: Residuals
        residuals = all_pred - all_true
        axes[1].scatter(all_true, residuals, alpha=0.6, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', label='Perfect Prediction')
        axes[1].set_xlabel('True Values')
        axes[1].set_ylabel('Residuals (Predicted - True)')
        axes[1].set_title('Residual Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add residual statistics
        residual_std = np.std(residuals)
        axes[1].text(0.05, 0.95, f'Residual Std: {residual_std:.3f}', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "enhanced_predictions_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_moe_routing(self, moe_model, X_test, y_test):
        """Analyze how well the MoE model routes samples to appropriate experts"""
        if not isinstance(moe_model, MixtureOfExpertsRegressor):
            return None
        
        try:
            # Get expert assignments and predictions
            routing_info = moe_model.predict_expert_assignment(X_test)
            assignments = routing_info['assignments']
            confidences = routing_info['confidence']
            
            # Get predictions from each method
            hard_preds = moe_model.predict(X_test)  # Hard routing
            soft_preds = moe_model.predict_soft_routing(X_test)  # Soft routing
            
            # Analyze routing accuracy based on true solubility
            true_labels = (y_test > moe_model.threshold).astype(int)
            routing_accuracy = np.mean(assignments == true_labels)
            
            # Performance by expert
            low_mask = assignments == 0
            high_mask = assignments == 1
            
            results = {
                'routing_accuracy': routing_accuracy,
                'avg_confidence': np.mean(confidences),
                'hard_routing_rmse': rmse(y_test, hard_preds),
                'soft_routing_rmse': rmse(y_test, soft_preds),
                'hard_routing_r2': r2(y_test, hard_preds),
                'soft_routing_r2': r2(y_test, soft_preds),
            }
            
            # Expert-specific performance
            if np.any(low_mask):
                results['low_expert_rmse'] = rmse(y_test[low_mask], hard_preds[low_mask])
                results['low_expert_r2'] = r2(y_test[low_mask], hard_preds[low_mask])
                results['low_expert_count'] = low_mask.sum()
            
            if np.any(high_mask):
                results['high_expert_rmse'] = rmse(y_test[high_mask], hard_preds[high_mask])
                results['high_expert_r2'] = r2(y_test[high_mask], hard_preds[high_mask])
                results['high_expert_count'] = high_mask.sum()
            
            return results
            
        except Exception as e:
            print(f"Error analyzing MoE routing: {e}")
            return None

# ═════════════════════════ MAIN EXECUTION ═══════════════════════════════
def main():
    """Main execution function"""
    # ENHANCED configuration for MAXIMUM PERFORMANCE with advanced MoE
    config = EnhancedConfig(
        tsv_path="converted_2025-04-25_104236.tsv",
        use_dscribe=True,  # ENABLE DScribe - critical for top performance!
        dscribe_descriptors=["soap", 
                             "sine_matrix",
                             #"mbtr"
                             ],  # Use comprehensive DScribe descriptors
        
        # Enhanced SOAP configurations - multiple scales for better coverage
        soap_configs=[
             #{"r_cut": 6.0, "n_max": 8, "l_max": 6, "permutation": "none"},   # Primary config
             #{"r_cut": 4.0, "n_max": 6, "l_max": 4, "permutation": "none"},   # Local features
             {"r_cut": 8.0, "n_max": 10, "l_max": 8, "permutation": "none"}, # Long-range (commented for speed)
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
        outer_folds=10,  # Standard 10-fold CV
        inner_folds=10,  # Reduced inner folds for speed
        search_iterations=20,  # More iterations for better hyperparameter search
        create_visualizations=True,  # Enable enhanced visualizations
        use_gpu=USE_CUML,  # Maximum GPU acceleration
        use_open_cosmo=True,  # Use OpenCOSMO features - critical for performance!
        test_without_open_cosmo=False,  # Focus on WITH OpenCOSMO first
        use_drip_cv=False,  # Disable for this optimization run
        drip_strategies=["standard_cv"],  # Focus on standard CV
        n_jobs=12,  # Use maximum CPU cores for parallel processing
        verbose=1,
        
        # Enhanced Regularization Settings - Lighter for better performance
        regularization_strength="light",  # Reduced from medium - less aggressive
        enable_regularized_models=True,  # Enable Ridge, Lasso, ElasticNet
        use_feature_selection=True,  # Disable feature selection initially to isolate issues
        feature_selection_methods=["variance","elastic_net","bayesian_ridge"],  # Simplified
        variance_threshold=0.001  # Lower threshold
    )
    
    print("="*80)
    print("ENHANCED SOLUBILITY WORKBENCH 2025 - ADVANCED OPTIMIZATION")
    print("="*80)
    print(f"Configuration: {config.models} models, {config.dscribe_descriptors} descriptors")
    print(f"MoE Threshold Method: {config.moe_threshold_method}")
    print(f"SOAP configs: {len(config.soap_configs)}")
    print(f"Sine permutations: {config.sine_permutations}")
    print(f"Fingerprint configs: {len(config.fp_configs)}")
    print(f"PCA configs: {len(config.pca_configs)}")
    print(f"GPU acceleration: {config.use_gpu}")
    print(f"Search iterations: {config.search_iterations}")
    print(f"OpenCOSMO testing: {config.test_without_open_cosmo}")
    print("="*80)
    
    # Check if dataset exists
    if not Path(config.tsv_path).exists():
        print(f"ERROR: Dataset not found: {config.tsv_path}")
        return
    
    # Initialize analyzer
    analyzer = EnhancedSolubilityAnalyzer(config)
    
    # Run analysis
    start_time = time.time()
    try:
        all_results = analyzer.run_comprehensive_analysis()
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Results directory: {analyzer.output_dir}")
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
            print(f"Regularization strength used: {config.regularization_strength}")
            print(f"Regularized models enabled: {config.enable_regularized_models}")
            print(f"Feature selection enabled: {config.use_feature_selection}")
            if config.use_feature_selection:
                print(f"Feature selection methods: {config.feature_selection_methods}")
                print(f"Variance threshold: {config.variance_threshold}")
            
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
            if config.use_feature_selection:
                print(f"\nFeature selection impact:")
                print(f"  Methods tested: {', '.join(config.feature_selection_methods)}")
                print(f"  Expected benefits: Reduced overfitting, faster training, better generalization")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    main()
