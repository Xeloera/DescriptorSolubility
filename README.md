# Solubility Prediction Framework

A comprehensive machine learning framework for predicting molecular solubility using advanced molecular descriptors, ensemble methods, and GPU acceleration.

## Overview

This framework provides state-of-the-art solubility prediction capabilities through:

- **Advanced Molecular Descriptors**: DScribe SOAP, Sine Matrix, MBTR descriptors combined with traditional fingerprints
- **OpenCOSMO Integration**: Quantum mechanical solvent descriptors for enhanced accuracy
- **Mixture of Experts**: Specialized models for different solubility regimes
- **GPU Acceleration**: RAPIDS cuML integration for high-performance computing
- **Comprehensive Analysis**: Advanced cross-validation, feature importance, and visualization

## Project Structure

```
DescriptorSolubility/
├── main.py                          # Main execution script
├── data.tsv                         # Dataset (TSV format)
├── modules/
│   ├── config.py                    # Configuration management
│   ├── analysis.py                  # Main analysis engine
│   ├── dataset.py                   # Data preprocessing and feature generation
│   ├── model_picker.py              # Model selection and hyperparameter tuning
│   ├── functions/
│   │   └── metrics.py               # Performance metrics
│   ├── models/
│   │   ├── mixtureofexpertsregressor.py  # Mixture of Experts implementation
│   │   └── pca.py                   # PCA utilities
│   └── reporting/
│       ├── results.py               # Results analysis and visualization
│       └── testing.py               # Performance reporting
└── output/                          # Generated results and reports
```

## Features

### Molecular Descriptors

#### DScribe Descriptors
- **SOAP (Smooth Overlap of Atomic Positions)**: Multi-scale local environment descriptors
- **Sine Matrix**: Global molecular structure representation
- **MBTR (Many-Body Tensor Representation)**: Multi-body atomic interactions

#### Traditional Fingerprints
- **Morgan Fingerprints**: Circular fingerprints with customizable radius
- **Combined Fingerprints**: Morgan + Avalon + Torsion fingerprints
- **RDKit Descriptors**: Comprehensive molecular property descriptors

#### OpenCOSMO Features
- Quantum mechanical solvent descriptors
- Thermodynamic property predictions
- Enhanced solvent-specific modeling

### Machine Learning Models

#### Core Models
- **XGBoost**: Gradient boosting with GPU acceleration
- **LightGBM**: Fast gradient boosting framework
- **Random Forest**: Ensemble decision trees
- **Neural Networks**: Deep learning models
- **Ridge/Lasso/ElasticNet**: Regularized linear models

#### Advanced Techniques
- **Mixture of Experts**: Specialized models for different solubility ranges
- **Automated Hyperparameter Tuning**: Randomized search with cross-validation
- **Feature Selection**: Variance-based and L1-regularized selection
- **Dimensionality Reduction**: PCA with multiple component configurations

### Cross-Validation Strategies

- **Standard K-Fold**: Traditional cross-validation
- **Drip-Feeding CV**: Novel approach for solvent/solute generalization
- **Nested CV**: Robust model evaluation with hyperparameter optimization

## Installation

### Prerequisites

```bash
# Core dependencies
pip install pandas numpy scikit-learn xgboost lightgbm
pip install rdkit matplotlib seaborn joblib

# DScribe for molecular descriptors
pip install dscribe

# Optional: GPU acceleration (RAPIDS)
conda install -c rapidsai -c nvidia -c conda-forge cuml cupy
```

### GPU Setup (Optional)

For GPU acceleration, ensure CUDA is installed and configure:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2"
export XGB_DEVICE="GPU"
```

## Usage

### Basic Usage

```python
from modules.config import Config
from modules.analysis import SolubilityAnalyzer

# Configure the analysis
config = Config(
    tsv_path="data.tsv",
    use_dscribe=True,
    models=["xgboost", "random_forest"],
    use_gpu=True
)

# Run analysis
analyzer = SolubilityAnalyzer(config)
results = analyzer.analysis()
```

### Command Line Execution

```bash
python main.py
```

### Configuration Options

#### Data Configuration
```python
config = Config(
    tsv_path="data.tsv",                    # Input dataset
    target_col="solubility_g_100g_log",     # Target variable
    solute_col="solute_smiles",             # Solute SMILES
    solvent_col="solvent_smiles",           # Solvent SMILES
    solvent_name_col="solvent_name"         # Readable solvent names
)
```

#### Descriptor Configuration
```python
# DScribe descriptors
dscribe_descriptors=["soap", "sine_matrix", "mbtr"]

# SOAP configurations for multi-scale analysis
soap_configs=[
    {"r_cut": 6.0, "n_max": 8, "l_max": 6},   # Standard
    {"r_cut": 8.0, "n_max": 10, "l_max": 8},  # Long-range
    {"r_cut": 4.0, "n_max": 6, "l_max": 4}    # Local features
]

# Fingerprint configurations
fp_configs=[
    {"type": "morgan", "n_bits": 4096, "radius": 3},
    {"type": "combined", "n_bits": 4096, "radius": 3},
    {"type": "rdkit_descriptors"}
]
```

#### Model Configuration
```python
# Model selection
models=["xgboost", "lightgbm", "random_forest", "neural_network"]

# Cross-validation settings
outer_folds=5                    # Outer CV folds
inner_folds=3                    # Inner CV for hyperparameter tuning
search_iterations=50             # Hyperparameter search iterations

# Performance optimization
use_gpu=True                     # Enable GPU acceleration
n_jobs=32                        # Parallel processing cores
```

## Input Data Format

The framework expects a TSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `solute_smiles` | SMILES string of solute molecule | `CCO` |
| `solvent_smiles` | SMILES string of solvent molecule | `O` |
| `solvent_name` | Human-readable solvent name | `water` |
| `solubility_g_100g_log` | Log solubility (target variable) | `-1.23` |

Optional OpenCOSMO columns:
- `opencosmo_solubility_mole_frac`
- `opencosmo_solubility_mass_fraction`
- Additional thermodynamic properties

## Output

### Generated Files

```
output/solubility_analysis_YYYYMMDD_HHMMSS/
├── enhanced_detailed_results.csv        # Complete results
├── enhanced_summary_statistics.json     # Summary metrics
├── enhanced_report.md                   # Comprehensive report
├── solvent_analysis.csv                 # Per-solvent performance
├── enhanced_visualizations/
│   ├── enhanced_model_comparison.png
│   ├── opencosmo_impact.png
│   ├── performance_distributions.png
│   └── enhanced_predictions_analysis.png
└── best_model_fold_*.pkl                # Trained models
```

### Performance Metrics

- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Explained Variance**: Proportion of variance explained

### Analysis Reports

- **Model Comparison**: Performance across different algorithms
- **OpenCOSMO Impact**: Effect of quantum mechanical descriptors
- **Cross-Validation Strategy**: Comparison of CV approaches
- **Solvent-Specific Analysis**: Performance by individual solvents
- **Feature Importance**: Most influential molecular descriptors

## Advanced Features

### Mixture of Experts

Automatically segments the prediction space based on solubility thresholds:

```python
use_mixture_of_experts=True
moe_threshold_method="solubility_cutoff"  # Natural logS = 0 threshold
```

### Drip-Feeding Cross-Validation

Novel CV strategy for assessing generalization:

```python
use_drip_cv=True
drip_strategies=["standard_cv", "drip_solvent", "drip_solute"]
```

### GPU Acceleration

Leverage RAPIDS cuML for high-performance computing:

```python
use_gpu=True                    # Enable GPU models
use_gpu_viz=True               # GPU-accelerated visualizations
```

### Feature Selection

Automated feature selection for high-dimensional data:

```python
use_feature_selection=True
feature_selection_methods=["variance", "lasso_selection"]
variance_threshold=0.01
```

## Performance Optimization

### Memory Management
- Efficient sparse matrix handling
- Batch processing for large datasets
- Memory-mapped file I/O

### Computational Efficiency
- GPU acceleration with cuML
- Parallel processing with joblib
- Optimized hyperparameter search

### Scalability
- Configurable cross-validation folds
- Adjustable search iterations
- Memory-conscious feature generation


## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For questions, issues, or feature requests:

- Open an issue on GitHub
- Check the documentation
- Review example configurations

## Changelog

### Version 1.0.0
- Initial release
- DScribe descriptor integration
- OpenCOSMO feature support
- Mixture of Experts implementation
- GPU acceleration support
- Comprehensive visualization suite
