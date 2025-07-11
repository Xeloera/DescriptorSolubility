
from typing import List, Dict
from dataclasses import dataclass, field

import json

from pydantic import BaseModel, Field
import os
from datetime import datetime
from pathlib import Path
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


@dataclass
class Config:
    # Dataset - IMPORTANT: Dual solvent handling for best of both worlds
    tsv_path: str = "data.tsv"
    
    target_col: str = "solubility_g_100g_log"
    
    solute_col: str = "solute_smiles"
    solvent_col: str = "solvent_smiles"  # Use SMILES for molecular descriptors
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
    moe_threshold_method: str = "solubility_cutoff"  # "median", "percentile", "kmeans", "solubility_cutoff", "gradient_based"
    moe_threshold_value: float = 0.5  # For percentile method (ignored for solubility_cutoff)
    moe_n_experts: int = 3  # Number of experts for advanced methods
    moe_gate_type: str = "soft"  # "soft" or "hard" gating
    moe_regularization: float = 0.01  # Regularization for gate network
    
    # OpenCOSMO and solvent features
    use_open_cosmo: bool = True
    test_without_open_cosmo: bool = True  # Test both with and without OpenCOSMO
    
    # Model configurations
    models: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "random_forest", "neural_network", "logistic_regression"
    ])
    
    # Feature importance and model interpretation
    calculate_feature_importance: bool = True  # Calculate feature importances
    use_shap: bool = False  # Use SHAP for model interpretation (slower)
    calculate_model_metrics: bool = True  # Calculate BIC/AIC-like metrics
    importance_threshold: float = 0.001  # Threshold for feature importance
    
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
    

    output_main_dir: str = "output"
    if os.path.exists("output")==False:
        os.mkdir(output_main_dir)
    output_dir: str ="solubility_analysis"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(os.path.join(output_main_dir,f"{output_dir}_{timestamp}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_models: bool = True
    create_report: bool = True
    
    # Performance
    use_gpu: bool = USE_CUML
    n_jobs: int = 32
    verbose: int = 1
    
    # Regularization Control
    regularization_strength: str = "medium"  # "light", "medium", "strong", "very_strong"
    enable_regularized_models: bool = True  # Enable Ridge, Lasso, ElasticNet, SVR, BayesianRidge
    
    # Feature Engineering Control
    add_interaction_features: bool = True  # Add interaction features between descriptors
    add_ase_interactions: bool = True  # Add ASE-based solvent-solute interactions
    max_features_before_selection: int = 150000  # Maximum features before applying selection
    use_sparse_matrices: bool = True  # Convert to sparse matrices when beneficial
    
    # Compression options for descriptors
    compress_descriptors: bool = False  # Compress descriptor matrices
    compression_method: str = "pca"  # "pca", "svd", "autoencoder"
    soap_compression_components: int = 500  # Components for SOAP compression
    sine_compression_components: int = 30  # Components for Sine matrix compression
    descriptor_specific_compression: bool = True  # Apply compression only to SOAP and Sine
    
    # Feature Selection for Regularization
    use_feature_selection: bool = True  # Enable feature selection
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        "variance", "lasso_selection"  # "variance", "univariate", "lasso_selection", "rfe"
    ])
    variance_threshold: float = 0.01  # Threshold for variance-based feature selection
    
    SPECIES = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se", "As"]  # Extended with common heteroatoms
    N_ATOMS_MAX = 200
    SINE_EIGS = 50
    SOAP_DIM = None
    _soap_, _sine_, _mbtr_, _coulomb_ = None, None, None, None
    
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
    @property
    def ALL_COL(self):
        return self.OPEN_COSMO + self.PRECOMPUTED_SOLVENT_FEATURES + self.PRECOMPUTED_SOLUTE_FEATURES
    

    def config_json(self):
        return json.dumps(self.__dict__, 
                            indent=4, 
                            sort_keys=True, 
                            default=lambda o: o.__dict__ if hasattr(o, '__dict__') else o)
        
    def from_json(self, json_str: str):
        data = json.loads(json_str)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config does not have attribute '{key}'")
        return self



        
        



