import logging
import sys
from datetime import datetime
from pathlib import Path
import json


class EnhancedLogger:
    """Enhanced logging for solubility analysis with detailed tracking"""
    
    def __init__(self, output_dir, verbose=1):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.log_file = self.output_dir / "analysis_log.txt"
        self.json_log = self.output_dir / "analysis_log.json"
        self.events = []
        
        # Set up file logging
        self.logger = logging.getLogger("SolubilityAnalysis")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO if verbose > 0 else logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.start_time = datetime.now()
        self.log_event("analysis_started", {"timestamp": self.start_time.isoformat()})
    
    def log_event(self, event_type, data=None):
        """Log a structured event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data or {}
        }
        self.events.append(event)
        
        # Also log to standard logger
        if data:
            self.logger.info(f"{event_type}: {json.dumps(data, indent=2)}")
        else:
            self.logger.info(event_type)
    
    def log_configuration(self, config):
        """Log analysis configuration"""
        config_data = {
            "models": config.models,
            "use_dscribe": config.use_dscribe,
            "dscribe_descriptors": config.dscribe_descriptors if config.use_dscribe else [],
            "use_gpu": config.use_gpu,
            "n_jobs": config.n_jobs,
            "outer_folds": config.outer_folds,
            "inner_folds": config.inner_folds,
            "search_iterations": config.search_iterations,
            "regularization_strength": config.regularization_strength,
            "feature_engineering": {
                "add_interaction_features": config.add_interaction_features,
                "add_ase_interactions": config.add_ase_interactions,
                "compress_descriptors": config.compress_descriptors,
                "variance_threshold": config.variance_threshold
            }
        }
        self.log_event("configuration", config_data)
    
    def log_dataset_info(self, dataset_shape, n_features, unique_solvents, unique_solutes):
        """Log dataset information"""
        self.log_event("dataset_info", {
            "n_samples": dataset_shape[0],
            "n_features_raw": dataset_shape[1] if len(dataset_shape) > 1 else 0,
            "n_features_processed": n_features,
            "unique_solvents": unique_solvents,
            "unique_solutes": unique_solutes
        })
    
    def log_feature_engineering(self, feature_type, n_features_before, n_features_after):
        """Log feature engineering steps"""
        self.log_event("feature_engineering", {
            "type": feature_type,
            "features_before": n_features_before,
            "features_after": n_features_after,
            "reduction_ratio": 1 - (n_features_after / n_features_before) if n_features_before > 0 else 0
        })
    
    def log_model_training(self, model_name, fold, metrics, duration=None):
        """Log model training results"""
        self.log_event("model_training", {
            "model": model_name,
            "fold": fold,
            "metrics": metrics,
            "duration_seconds": duration
        })
    
    def log_best_model(self, model_info):
        """Log best model information"""
        self.log_event("best_model", model_info)
    
    def log_error(self, error_type, error_message, context=None):
        """Log errors with context"""
        self.logger.error(f"{error_type}: {error_message}")
        self.log_event("error", {
            "type": error_type,
            "message": str(error_message),
            "context": context or {}
        })
    
    def log_warning(self, warning_message, context=None):
        """Log warnings"""
        self.logger.warning(warning_message)
        self.log_event("warning", {
            "message": warning_message,
            "context": context or {}
        })
    
    def finalize(self):
        """Finalize logging and save summary"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.log_event("analysis_completed", {
            "duration_seconds": duration,
            "duration_formatted": f"{duration/60:.1f} minutes",
            "total_events": len(self.events)
        })
        
        # Save JSON log
        with open(self.json_log, 'w') as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "events": self.events
            }, f, indent=2)
        
        self.logger.info(f"Analysis completed in {duration/60:.1f} minutes")
        self.logger.info(f"Logs saved to: {self.log_file}")
        
        # Generate summary statistics
        event_counts = {}
        for event in self.events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "duration": duration,
            "total_events": len(self.events),
            "event_summary": event_counts
        }