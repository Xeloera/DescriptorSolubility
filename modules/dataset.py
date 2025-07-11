import warnings
from dscribe.descriptors import SOAP, MBTR, SineMatrix, CoulombMatrix, ACSF
from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
from pandas import DataFrame
import pandas as pd
from typing import Union, List, DefaultDict
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import (Descriptors, AllChem, DataStructs,
                        rdFingerprintGenerator, ChemicalFeatures)

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=FutureWarning, module="dscribe")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")  # Suppress sklearn pipeline warnings
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from dataclasses import dataclass
from modules.config import Config

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.decomposition import TruncatedSVD, PCA
import gc

# Constants from Config
SEED = 42
SPECIES = Config.SPECIES
SINE_EIGS = Config.SINE_EIGS

# Helper functions for DScribe descriptors
def _get_soap(r_cut=6.0, n_max=8, l_max=6, **kwargs):
    """Create SOAP descriptor with given parameters"""
    return SOAP(
        species=SPECIES,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=False,
        average="inner",  # Automatically average across atomic environments
        **kwargs
    )

def _get_mbtr():
    """Create MBTR descriptor"""
    return MBTR(
        species=SPECIES,
        geometry={"function": "inverse_distance"},
        grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
        weighting={"function": "exp", "scale": 0.5, "cutoff": 1e-3}
    )

def _get_sine(permutation="sorted_l2"):
    """Create Sine Matrix descriptor"""
    return SineMatrix(
        n_atoms_max=Config.N_ATOMS_MAX,
        permutation=permutation,
        sparse=False
    )

def _get_coulomb():
    """Create Coulomb Matrix descriptor"""
    return CoulombMatrix(
        n_atoms_max=Config.N_ATOMS_MAX,
        permutation="sorted_l2",
        sparse=False
    )

@dataclass
class Dataset:
    def __init__(self, path: str, config: Config = None):
        self.path = path
        self.data = None
        self.config = config if config else Config()
        self.species = SPECIES
        self.use_descriptor = {
            'soap': True,
            'mbtr': False,
            'sine_matrix': True,
            'coulomb_matrix': False,
            'acsf': False
        }
        self.descriptor_system = None

    def load(self):
        self.data = read(self.path, format='extxyz')

    def save(self, path: str):
        write(path, self.data, format='extxyz')

    def to_dataframe(self) -> DataFrame:
        return pd.DataFrame([atom.info for atom in self.data])

class DScribeDescriptor:
    """Optimized DScribe descriptor calculator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.descriptor_names = config.dscribe_descriptors if config.use_dscribe else []
    
    def _mol_to_atoms(self, mol):
        """Convert RDKit molecule to ASE Atoms object with explicit hydrogens"""
        if mol is None:
            if self.config.verbose > 0:
                print("Warning: Input molecule is None")
            return None
        
        try:
            # Add explicit hydrogens
            mol_with_h = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol_with_h, randomSeed=SEED) == -1:
                # If embedding fails, try without optimization
                if AllChem.EmbedMolecule(mol_with_h, randomSeed=SEED, useRandomCoords=True) == -1:
                    if self.config.verbose > 0:
                        print("Warning: Failed to embed molecule in 3D")
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
            
            if self.config.verbose > 1:
                print(f"Molecule has {len(symbols)} atoms: {set(symbols)}")
            
            # Filter to known species only
            valid_indices = [i for i, s in enumerate(symbols) if s in SPECIES]
            if not valid_indices:
                if self.config.verbose > 0:
                    print(f"Warning: No valid species found. Symbols: {set(symbols)}, Known species: {SPECIES}")
                # Try to add missing species dynamically if they're common
                unknown_species = set(symbols) - set(SPECIES)
                for species in unknown_species:
                    if species in ['B', 'Si', 'Se', 'As']:  # Common heteroatoms
                        SPECIES.append(species)
                        print(f"Added {species} to known species list")
                # Retry with updated species list
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
            
            if self.config.verbose > 1:
                print(f"Successfully created atoms object with {len(atoms)} atoms")
            
            return atoms
            
        except Exception as e:
            if self.config.verbose > 0:
                print(f"Warning: Failed to convert molecule to atoms: {e}")
            return None
    
    def calculate_single(self, mol):
        """Calculate all DScribe descriptors for a single molecule with multiple configurations"""
        atoms = self._mol_to_atoms(mol)
        if atoms is None:
            if self.config.verbose > 0:
                print(f"Warning: Failed to convert molecule to atoms object")
            return None
        
        features = {}
        
        try:
            # SOAP descriptor with multiple configurations
            if "soap" in self.descriptor_names:
                for i, soap_config in enumerate(self.config.soap_configs):
                    try:
                        soap_desc = _get_soap(**soap_config)
                        soap_vec = soap_desc.create(atoms).astype(np.float32)
                        features[f'soap_config_{i}'] = soap_vec
                        if self.config.verbose > 1:
                            print(f"Successfully calculated SOAP config {i}, shape: {soap_vec.shape}")
                    except Exception as e:
                        if self.config.verbose > 0:
                            print(f"Warning: Failed to calculate SOAP config {i}: {e}")
            
            # MBTR descriptor (if enabled)
            if "mbtr" in self.descriptor_names:
                try:
                    mbtr_desc = _get_mbtr()
                    mbtr_vec = mbtr_desc.create(atoms)
                    if mbtr_vec.ndim > 1:
                        mbtr_vec = mbtr_vec.flatten()
                    features['mbtr'] = mbtr_vec.astype(np.float32)
                    if self.config.verbose > 1:
                        print(f"Successfully calculated MBTR, shape: {mbtr_vec.shape}")
                except Exception as e:
                    if self.config.verbose > 0:
                        print(f"Warning: Failed to calculate MBTR: {e}")
            
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
                        if self.config.verbose > 1:
                            print(f"Successfully calculated Sine Matrix {perm}, shape: {eigenvals.shape}")
                    except Exception as e:
                        if self.config.verbose > 0:
                            print(f"Warning: Failed to calculate Sine Matrix with {perm}: {e}")
            
            # Coulomb Matrix (if enabled)
            if "coulomb_matrix" in self.descriptor_names:
                try:
                    coulomb_desc = _get_coulomb()
                    coulomb_vec = coulomb_desc.create(atoms)
                    if coulomb_vec.ndim > 1:
                        coulomb_vec = coulomb_vec.flatten()
                    features['coulomb_matrix'] = coulomb_vec.astype(np.float32)
                    if self.config.verbose > 1:
                        print(f"Successfully calculated Coulomb Matrix, shape: {coulomb_vec.shape}")
                except Exception as e:
                    if self.config.verbose > 0:
                        print(f"Warning: Failed to calculate Coulomb Matrix: {e}")
                
        except Exception as e:
            if self.config.verbose > 0:
                print(f"Warning: Failed to calculate descriptors: {e}")
            return None
        
        if not features:
            if self.config.verbose > 0:
                print(f"Warning: No features calculated for molecule")
        
        return features if features else None
    
    def calculate_batch(self, smiles_list):
        """Calculate DScribe descriptors for a batch of SMILES"""
        print(f"Calculating DScribe descriptors for {len(smiles_list)} molecules...")
        print(f"Enabled descriptors: {self.descriptor_names}")
        
        # Convert SMILES to molecules with error tracking
        mols = []
        failed_indices = []
        for i, smi in enumerate(smiles_list):
            try:
                # Clean SMILES string
                smi_clean = smi.strip()
                if not smi_clean:
                    mols.append(None)
                    failed_indices.append(i)
                    continue
                    
                mol = Chem.MolFromSmiles(smi_clean)
                if mol is None:
                    failed_indices.append(i)
                    if self.config.verbose > 0:
                        print(f"Failed to parse SMILES at index {i}: {smi_clean[:50]}...")
                mols.append(mol)
            except Exception as e:
                mols.append(None)
                failed_indices.append(i)
                if self.config.verbose > 0:
                    print(f"Exception parsing SMILES at index {i}: {e}")
        
        valid_mols = [mol for mol in mols if mol is not None]
        print(f"Valid molecules from SMILES: {len(valid_mols)}/{len(mols)}")
        if failed_indices and self.config.verbose > 0:
            print(f"Failed indices: {failed_indices[:10]}..." if len(failed_indices) > 10 else f"Failed indices: {failed_indices}")
        
        if len(valid_mols) == 0:
            print("ERROR: No valid molecules created from SMILES!")
            return {}, []
        
        with ProcessPoolExecutor(max_workers=min(self.config.n_jobs, 8)) as executor:
            results = list(tqdm(
                executor.map(self.calculate_single, mols),
                total=len(mols),
                desc="DScribe descriptors"
            ))
        
        # Count non-None results
        non_none_results = [r for r in results if r is not None]
        print(f"Non-None results: {len(non_none_results)}/{len(results)}")
        
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
        
        print(f"Results passing validation: {len(valid_indices)}")
        print(f"Descriptor arrays found: {list(descriptor_arrays.keys())}")
        
        # Convert to numpy arrays
        final_arrays = {}
        for desc_name, vectors in descriptor_arrays.items():
            if vectors:
                final_arrays[desc_name] = np.vstack(vectors)
                print(f"Final array {desc_name}: shape {final_arrays[desc_name].shape}")
        
        print(f"Successfully calculated DScribe descriptors for {len(valid_indices)} molecules")
        return final_arrays, valid_indices


class Fingerprint:
    """Enhanced fingerprint calculator with multiple types"""

    def __init__(self, config: Config):
        self.config = config
        self.dscribe_suite = DScribeDescriptor(config) if config.use_dscribe else None

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
                dscribe_results, _ = self.dscribe_suite.calculate_batch(solute_smiles_list)
                
                for desc_name, desc_matrix in dscribe_results.items():
                    # Apply compression if enabled for SOAP and Sine matrices
                    if self.config.compress_descriptors and self.config.descriptor_specific_compression:
                        if "soap" in desc_name and desc_matrix.shape[1] > self.config.soap_compression_components:
                            print(f"Compressing SOAP descriptor from {desc_matrix.shape[1]} to {self.config.soap_compression_components} components...")
                            if self.config.compression_method == "pca":
                                compressor = PCA(n_components=self.config.soap_compression_components, random_state=42)
                            else:  # SVD
                                compressor = TruncatedSVD(n_components=self.config.soap_compression_components, random_state=42)
                            desc_matrix = compressor.fit_transform(desc_matrix)
                            print(f"Compression ratio: {compressor.explained_variance_ratio_.sum():.2%}")
                        elif "sine" in desc_name and desc_matrix.shape[1] > self.config.sine_compression_components:
                            print(f"Compressing Sine descriptor from {desc_matrix.shape[1]} to {self.config.sine_compression_components} components...")
                            if self.config.compression_method == "pca":
                                compressor = PCA(n_components=self.config.sine_compression_components, random_state=42)
                            else:  # SVD
                                compressor = TruncatedSVD(n_components=self.config.sine_compression_components, random_state=42)
                            desc_matrix = compressor.fit_transform(desc_matrix)
                            print(f"Compression ratio: {compressor.explained_variance_ratio_.sum():.2%}")
                    
                    all_features.append(desc_matrix)
                    names = [f"solute_dscribe_{desc_name}_{i}" for i in range(desc_matrix.shape[1])]
                    feature_names.extend(names)
                    print(f"Added {desc_matrix.shape[1]} SOLUTE features from DScribe {desc_name}")
            except Exception as e:
                print(f"Warning: Failed to calculate solute DScribe descriptors: {e}")
                
            try:
                print(f"Calculating SOLVENT DScribe descriptors: {self.config.dscribe_descriptors}")
                dscribe_results, _ = self.dscribe_suite.calculate_batch(solvent_smiles_list)
                
                for desc_name, desc_matrix in dscribe_results.items():
                    # Apply compression if enabled for SOAP and Sine matrices
                    if self.config.compress_descriptors and self.config.descriptor_specific_compression:
                        if "soap" in desc_name and desc_matrix.shape[1] > self.config.soap_compression_components:
                            print(f"Compressing SOAP descriptor from {desc_matrix.shape[1]} to {self.config.soap_compression_components} components...")
                            if self.config.compression_method == "pca":
                                compressor = PCA(n_components=self.config.soap_compression_components, random_state=42)
                            else:  # SVD
                                compressor = TruncatedSVD(n_components=self.config.soap_compression_components, random_state=42)
                            desc_matrix = compressor.fit_transform(desc_matrix)
                            print(f"Compression ratio: {compressor.explained_variance_ratio_.sum():.2%}")
                        elif "sine" in desc_name and desc_matrix.shape[1] > self.config.sine_compression_components:
                            print(f"Compressing Sine descriptor from {desc_matrix.shape[1]} to {self.config.sine_compression_components} components...")
                            if self.config.compression_method == "pca":
                                compressor = PCA(n_components=self.config.sine_compression_components, random_state=42)
                            else:  # SVD
                                compressor = TruncatedSVD(n_components=self.config.sine_compression_components, random_state=42)
                            desc_matrix = compressor.fit_transform(desc_matrix)
                            print(f"Compression ratio: {compressor.explained_variance_ratio_.sum():.2%}")
                    
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
        
        # Check total feature count before combining
        total_features = sum(f.shape[1] for f in all_features)
        print(f"Total features before combination: {total_features}")
        
        # Memory optimization: combine features in chunks if too large
        if total_features > self.config.max_features_before_selection:
            print(f"WARNING: Feature count ({total_features}) exceeds threshold ({self.config.max_features_before_selection})")
            print("Applying variance-based feature selection to each feature type...")
            
            # Apply variance threshold to each feature type separately
            from sklearn.feature_selection import VarianceThreshold
            filtered_features = []
            filtered_names = []
            
            start_idx = 0
            for i, feat_array in enumerate(all_features):
                end_idx = start_idx + feat_array.shape[1]
                feat_names = feature_names[start_idx:end_idx]
                
                # Apply variance threshold
                selector = VarianceThreshold(threshold=self.config.variance_threshold)
                try:
                    feat_filtered = selector.fit_transform(feat_array)
                    selected_indices = selector.get_support(indices=True)
                    selected_names = [feat_names[idx] for idx in selected_indices]
                    
                    filtered_features.append(feat_filtered)
                    filtered_names.extend(selected_names)
                    
                    print(f"Feature type {i}: {feat_array.shape[1]} -> {feat_filtered.shape[1]} features")
                except Exception as e:
                    print(f"Warning: Failed to filter feature type {i}: {e}")
                    filtered_features.append(feat_array)
                    filtered_names.extend(feat_names)
                
                start_idx = end_idx
            
            all_features = filtered_features
            feature_names = filtered_names
        
        # Combine all features
        feature_matrix = np.hstack(all_features).astype(np.float32)
        
        # Check for sparsity and convert if beneficial
        sparsity = np.sum(feature_matrix == 0) / feature_matrix.size
        print(f"Feature matrix sparsity: {sparsity:.2%}")
        
        # Store original dense matrix for interaction features
        dense_feature_matrix = feature_matrix
        
        # Convert to sparse if beneficial (but keep dense for now)
        use_sparse = sparsity > 0.5 and self.config.use_sparse_matrices
        if use_sparse:
            print("High sparsity detected - will convert to sparse after feature engineering")
        
        # Add feature engineering: interaction features for key descriptors
        if len(all_features) >= 2 and self.config.add_interaction_features:  # Only if we have multiple feature types
            # Determine optimal number of features for interactions
            max_interaction_features = min(50, all_features[0].shape[1] // 2, all_features[1].shape[1] // 2)
            
            if max_interaction_features > 0:
                # Create interaction features between first two feature types
                feat1 = all_features[0][:, :max_interaction_features]
                feat2 = all_features[1][:, :max_interaction_features]
                
                # Element-wise multiplication (interaction)
                interaction_features = feat1 * feat2
                
                # Add L2 norm features for magnitude interactions
                norm_features = np.sqrt(feat1**2 + feat2**2)
                
                # Add difference features
                diff_features = np.abs(feat1 - feat2)
                
                # Combine interaction features
                all_interactions = np.hstack([interaction_features, norm_features, diff_features])
                dense_feature_matrix = np.hstack([dense_feature_matrix, all_interactions])
                
                # Add interaction feature names
                interaction_names = [f"interaction_mult_{i}" for i in range(interaction_features.shape[1])]
                norm_names = [f"interaction_norm_{i}" for i in range(norm_features.shape[1])]
                diff_names = [f"interaction_diff_{i}" for i in range(diff_features.shape[1])]
                feature_names.extend(interaction_names + norm_names + diff_names)
                
                print(f"Added {all_interactions.shape[1]} interaction features:")
                print(f"  - {interaction_features.shape[1]} multiplicative")
                print(f"  - {norm_features.shape[1]} norm-based")
                print(f"  - {diff_features.shape[1]} difference-based")
        
        # Add ASE-based solvent-solute interaction features if enabled
        if self.config.add_ase_interactions and self.dscribe_suite:
            try:
                print("\nCalculating ASE-based solvent-solute interactions...")
                ase_features = self._calculate_ase_interactions(solute_smiles_list, solvent_smiles_list)
                if ase_features is not None:
                    dense_feature_matrix = np.hstack([dense_feature_matrix, ase_features])
                    ase_names = [f"ase_interaction_{i}" for i in range(ase_features.shape[1])]
                    feature_names.extend(ase_names)
                    print(f"Added {ase_features.shape[1]} ASE interaction features")
            except Exception as e:
                print(f"Warning: Failed to calculate ASE interactions: {e}")
        
        # Now convert to sparse if needed
        if use_sparse:
            print("Converting final feature matrix to sparse format...")
            feature_matrix = csr_matrix(dense_feature_matrix)
            gc.collect()
        else:
            feature_matrix = dense_feature_matrix
        
        # IMPORTANT: Keep dense format for now due to model compatibility
        # Many sklearn models have issues with sparse matrices
        if use_sparse:
            print("Note: Keeping dense format for model compatibility")
            feature_matrix = dense_feature_matrix
        
        print(f"Final feature matrix shape: {feature_matrix.shape}")
        print(f"Features breakdown:")
        print(f"  - Solute molecular features: {sum(1 for name in feature_names if name.startswith('solute_'))}")
        print(f"  - Solvent molecular features: {sum(1 for name in feature_names if name.startswith('solvent_') and not name.startswith('precomputed_'))}")
        print(f"  - Precomputed solvent features: {sum(1 for name in feature_names if name.startswith('precomputed_'))}")
        
        return feature_matrix, feature_names

    def _calculate_ase_interactions(self, solute_smiles_list, solvent_smiles_list):
        """Calculate ASE-based solvent-solute interaction features"""
        try:
            from ase.atoms import Atoms
            from ase.geometry import distance
            
            interaction_features = []
            
            for solute_smi, solvent_smi in zip(solute_smiles_list, solvent_smiles_list):
                try:
                    # Convert to molecules
                    solute_mol = Chem.MolFromSmiles(solute_smi)
                    solvent_mol = Chem.MolFromSmiles(solvent_smi)
                    
                    if solute_mol is None or solvent_mol is None:
                        interaction_features.append(np.zeros(10))  # Default features
                        continue
                    
                    # Get basic properties
                    solute_mw = Descriptors.MolWt(solute_mol)
                    solvent_mw = Descriptors.MolWt(solvent_mol)
                    solute_logp = Descriptors.MolLogP(solute_mol)
                    solvent_logp = Descriptors.MolLogP(solvent_mol)
                    solute_tpsa = Descriptors.TPSA(solute_mol)
                    solvent_tpsa = Descriptors.TPSA(solvent_mol)
                    
                    # Calculate interaction features
                    features = [
                        solute_mw / (solvent_mw + 1e-6),  # MW ratio
                        abs(solute_logp - solvent_logp),  # LogP difference
                        solute_tpsa * solvent_tpsa / 10000,  # TPSA interaction
                        (solute_logp + solvent_logp) / 2,  # LogP average
                        np.exp(-abs(solute_logp - solvent_logp)),  # LogP similarity
                        min(solute_tpsa, solvent_tpsa) / (max(solute_tpsa, solvent_tpsa) + 1e-6),  # TPSA ratio
                        Descriptors.NumHDonors(solute_mol) * Descriptors.NumHAcceptors(solvent_mol),  # H-bond potential
                        Descriptors.NumHAcceptors(solute_mol) * Descriptors.NumHDonors(solvent_mol),  # Reverse H-bond
                        abs(Descriptors.NumRotatableBonds(solute_mol) - Descriptors.NumRotatableBonds(solvent_mol)),  # Flexibility diff
                        (Descriptors.NumAromaticRings(solute_mol) + Descriptors.NumAromaticRings(solvent_mol)) / 2  # Aromaticity
                    ]
                    
                    interaction_features.append(features)
                    
                except Exception:
                    interaction_features.append(np.zeros(10))  # Default features on error
            
            return np.array(interaction_features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in ASE interaction calculation: {e}")
            return None
    
    def _extract_solvent_features(self, df, use_open_cosmo=True):
        """Extract solvent features from dataframe with OpenCOSMO control"""
        # Select columns based on OpenCOSMO flag
        if use_open_cosmo:
            solvent_cols = self.config.ALL_COL   # Includes both OpenCOSMO and precomputed features
        else:
            solvent_cols = self.config.PRECOMPUTED_SOLUTE_FEATURES + self.config.PRECOMPUTED_SOLVENT_FEATURES  # Only precomputed features
        
        # Check available columns
        available_cols = [col for col in solvent_cols if col in df.columns]
        if not available_cols:
            print(f"Warning: No solvent features found (use_open_cosmo={use_open_cosmo})")
            return None
        
        print(f"Using {len(available_cols)} solvent features out of {len(solvent_cols)} requested")
        
        solvent_features = df[available_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        return solvent_features.values.astype(np.float32)


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
