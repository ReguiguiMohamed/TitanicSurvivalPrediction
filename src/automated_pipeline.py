"""
Automated pipeline for the Titanic ML competition
Runs complete end-to-end machine learning workflow
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Local imports
from data_loader import load_data
from feature_engineering import engineer_features
from preprocessing import preprocessing_pipeline
from baseline_models import compare_models, train_random_forest, train_logistic_regression
from model_validation import cross_validate_model, calculate_metrics, model_comparison_report
from predictions import generate_predictions, create_submission

# Configuration
CONFIG_DIR = Path("config")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
SUBMISSIONS_DIR = Path("submissions")

# Create directories
for dir_path in [CONFIG_DIR, RESULTS_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load pipeline configuration from file or return default config."""
    config_file = CONFIG_DIR / "pipeline_config.json"
    
    default_config = {
        "model_params": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 42
            },
            "logistic_regression": {
                "max_iter": 1000,
                "random_state": 42
            }
        },
        "preprocessing_params": {
            "test_size": 0.2,
            "random_state": 42,
            "n_features": 12,
            "scale_features": True
        },
        "validation_params": {
            "cv_folds": 5,
            "scoring": "accuracy",
            "random_state": 42
        },
        "pipeline_params": {
            "save_models": True,
            "generate_submission": True,
            "run_validation": True
        }
    }
    
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            # Merge with default config
            for key in default_config:
                if key not in loaded_config:
                    loaded_config[key] = default_config[key]
            return loaded_config
        else:
            # Save default config for future use
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        return default_config


def run_data_pipeline() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run data loading and preprocessing pipeline."""
    logger.info("Starting data pipeline...")
    
    # Load raw data
    train_df, test_df = load_data()
    logger.info(f"Loaded {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Run preprocessing pipeline
    X_train, X_val, X_test, y_train, y_val, features, scaler = preprocessing_pipeline(train_df, test_df)
    
    logger.info(f"Preprocessing complete. Features: {len(features)}")
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    return X_train, X_val, X_test, y_train, y_val


def run_model_training(X_train, y_train, X_val, y_val, config: Dict) -> Dict[str, Any]:
    """Train and evaluate models."""
    logger.info("Starting model training...")
    
    results = {}
    
    # Train Random Forest
    rf_params = config["model_params"]["random_forest"]
    rf_model, rf_score = train_random_forest(
        np.vstack([X_train, X_val]), 
        np.concatenate([y_train, y_val])
    )
    results["random_forest"] = {
        "model": rf_model,
        "validation_score": rf_score
    }
    
    # Train Logistic Regression  
    lr_params = config["model_params"]["logistic_regression"]
    lr_model, lr_score = train_logistic_regression(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val])
    )
    results["logistic_regression"] = {
        "model": lr_model,
        "validation_score": lr_score
    }
    
    logger.info(f"Model training complete. RF: {rf_score:.4f}, LR: {lr_score:.4f}")
    
    return results


def run_model_validation(models: Dict, X, y, config: Dict) -> Dict[str, Any]:
    """Run comprehensive model validation."""
    logger.info("Starting model validation...")
    
    validation_results = {}
    
    for name, model_info in models.items():
        model = model_info["model"]
        
        # Cross validation
        cv_scores = cross_validate_model(model, X, y, cv=config["validation_params"]["cv_folds"])
        
        validation_results[name] = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
            "validation_score": model_info["validation_score"]
        }
        
        logger.info(f"{name}: CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return validation_results


def generate_final_submission(best_model, X_test, test_df, timestamp: str) -> Path:
    """Generate final submission file."""
    logger.info("Generating final submission...")
    
    # Generate predictions
    predictions = generate_predictions(best_model, X_test)
    
    # Create submission
    submission_path = SUBMISSIONS_DIR / f"automated_submission_{timestamp}.csv"
    submission_df = create_submission(test_df["PassengerId"], predictions, submission_path)
    
    logger.info(f"Submission saved to {submission_path}")
    logger.info(f"Predicted survival rate: {predictions.mean():.3f}")
    
    return submission_path


def run_complete_pipeline(config: Dict = None) -> Dict[str, Any]:
    """Run the complete automated ML pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting complete pipeline run at {timestamp}")
    
    if config is None:
        config = load_config()
    
    try:
        # Step 1: Data pipeline
        X_train, X_val, X_test, y_train, y_val = run_data_pipeline()
        
        # Step 2: Model training
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        
        models = run_model_training(X_train, y_train, X_val, y_val, config)
        
        # Step 3: Model validation (if enabled)
        validation_results = {}
        if config["pipeline_params"]["run_validation"]:
            validation_results = run_model_validation(models, X_combined, y_combined, config)
        
        # Step 4: Select best model
        best_model_name = max(models.keys(), key=lambda x: models[x]["validation_score"])
        best_model = models[best_model_name]["model"]
        
        logger.info(f"Best model: {best_model_name}")
        
        # Step 5: Generate submission (if enabled)
        submission_path = None
        if config["pipeline_params"]["generate_submission"]:
            # Need to load test data for PassengerId
            _, test_df = load_data()
            submission_path = generate_final_submission(best_model, X_test, test_df, timestamp)
        
        # Step 6: Save results
        pipeline_results = {
            "timestamp": timestamp,
            "best_model": best_model_name,
            "model_scores": {name: info["validation_score"] for name, info in models.items()},
            "validation_results": validation_results,
            "config": config,
            "submission_path": str(submission_path) if submission_path else None
        }
        
        results_path = RESULTS_DIR / f"pipeline_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        logger.info(f"Pipeline complete! Results saved to {results_path}")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def main():
    """Main entry point for the automated pipeline."""
    try:
        results = run_complete_pipeline()
        
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Best Model: {results['best_model']}")
        print(f"Best Score: {max(results['model_scores'].values()):.4f}")
        
        if results['submission_path']:
            print(f"Submission: {results['submission_path']}")
        
        print(f"Full Results: {RESULTS_DIR}/pipeline_results_{results['timestamp']}.json")
        print("="*60)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())