import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
from tqdm import tqdm

from DataLoader import DataLoader
from FeatureEngineering import FeatureEngineer
from models import MODELS
from models.EnsembleModel import EnsembleModel
from Model import MarchMadnessModel

def train_models(
    gender: str = 'M',
    start_season: int = 2003,
    end_season: int = 2023,
    val_seasons: Optional[List[int]] = None,
    test_seasons: Optional[List[int]] = None,
    model_types: List[str] = None,
    use_ensemble: bool = True,
    output_dir: str = './trained_models',
    random_state: int = 42
):
    """
    Train models for March Madness prediction.
    
    Args:
        gender: 'M' for men's data, 'W' for women's data
        start_season: First season to include in training
        end_season: Last season to include in training/testing
        val_seasons: List of seasons to use for validation 
        test_seasons: List of seasons to use for testing
        model_types: List of model types to train
        use_ensemble: Whether to create an ensemble of all models
        output_dir: Directory to save trained models
        random_state: Random seed for reproducibility
    """
    print(f"Training models for {gender}'s tournament from {start_season} to {end_season}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default models if not specified
    if model_types is None:
        model_types = list(MODELS.keys())
    
    # Validate model types
    for model_type in model_types:
        if model_type not in MODELS:
            raise ValueError(f"Model type '{model_type}' not available. Available models: {list(MODELS.keys())}")
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    
    # Create feature engineer
    print("Creating features...")
    engineer = FeatureEngineer(loader)
    
    # Prepare historical data
    X, y = engineer.prepare_historical_training_data(gender, start_season, end_season)
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    # Split data into train, validation and test sets
    if val_seasons is None:
        val_seasons = [end_season - 1]  # Use second-to-last season for validation
    if test_seasons is None:
        test_seasons = [end_season]  # Use last season for testing
        
    print(f"Validation seasons: {val_seasons}")
    print(f"Test seasons: {test_seasons}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.split_train_validation_test(
        X, y, val_seasons=val_seasons, test_seasons=test_seasons, random_split=False
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize models
    models = {}
    model_results = {}
    
    # Train each model
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        
        # Initialize the model
        model_class = MODELS[model_type]
        model = model_class(name=model_type, random_state=random_state)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics = model.evaluate(X_val, y_val)
        print(f"Validation metrics: {val_metrics}")
        
        # Save the model
        models[model_type] = model
        model_results[model_type] = {
            'validation': val_metrics
        }
        
        # Save model to disk
        with open(os.path.join(output_dir, f"{gender}_{model_type}_model.pkl"), 'wb') as f:
            pickle.dump(model, f)
        
        # Display feature importance
        if model_type == "NeuralNetwork":
            # Neural Network needs data to calculate feature importance
            feature_importance = model.get_feature_importance(X=X_val, y=y_val)
        else:
            feature_importance = model.get_feature_importance()
            
        if feature_importance is not None:
            print("\nTop 10 important features:")
            print(feature_importance.head(10))
            
            # Save feature importance
            feature_importance.to_csv(os.path.join(output_dir, f"{gender}_{model_type}_feature_importance.csv"), index=False)
    
    # Create ensemble model if requested
    if use_ensemble and len(models) > 1:
        print("\nTraining ensemble model...")
        
        # Create and train the ensemble
        ensemble = EnsembleModel(name="Ensemble", base_models=list(models.values()))
        ensemble.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics = ensemble.evaluate(X_val, y_val)
        print(f"Ensemble validation metrics: {val_metrics}")
        
        # Save the ensemble model
        models["Ensemble"] = ensemble
        model_results["Ensemble"] = {
            'validation': val_metrics
        }
        
        # Save ensemble to disk
        with open(os.path.join(output_dir, f"{gender}_Ensemble_model.pkl"), 'wb') as f:
            pickle.dump(ensemble, f)
    
    # Evaluate all models on test set
    print("\nEvaluating models on test set...")
    all_test_metrics = {}
    
    for model_name, model in models.items():
        test_metrics = model.evaluate(X_test, y_test)
        model_results[model_name]['test'] = test_metrics
        all_test_metrics[model_name] = test_metrics
        
        print(f"{model_name} test metrics: {test_metrics}")
    
    # Compare model performance
    print("\nModel comparison on test set:")
    metrics_df = pd.DataFrame(all_test_metrics).T
    print(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, f"{gender}_model_comparison.csv"))
    
    # Plot comparison of log loss (primary metric for March Madness prediction)
    plt.figure(figsize=(10, 6))
    metrics_df['log_loss'].sort_values().plot(kind='bar')
    plt.title(f"Model Log Loss Comparison ({gender}'s Tournament)")
    plt.ylabel('Log Loss (lower is better)')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{gender}_model_logloss_comparison.png"))
    
    # Plot comparison of accuracy
    plt.figure(figsize=(10, 6))
    metrics_df['accuracy'].sort_values(ascending=False).plot(kind='bar')
    plt.title(f"Model Accuracy Comparison ({gender}'s Tournament)")
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{gender}_model_accuracy_comparison.png"))
    
    # Save full results
    with open(os.path.join(output_dir, f"{gender}_model_results.pkl"), 'wb') as f:
        pickle.dump(model_results, f)
    
    print(f"\nTraining complete. Models and results saved to {output_dir}")
    return models, model_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train March Madness prediction models")
    parser.add_argument("--gender", type=str, default='M', choices=['M', 'W'], help="Gender of tournament (M or W)")
    parser.add_argument("--start_season", type=int, default=2003, help="First season to include in training")
    parser.add_argument("--end_season", type=int, default=2023, help="Last season to include in training/testing")
    parser.add_argument("--val_seasons", type=int, nargs="*", help="Seasons to use for validation")
    parser.add_argument("--test_seasons", type=int, nargs="*", help="Seasons to use for testing")
    parser.add_argument("--model_types", type=str, nargs="*", help="Model types to train")
    parser.add_argument("--no_ensemble", action="store_true", help="Don't create an ensemble model")
    parser.add_argument("--output_dir", type=str, default="./trained_models", help="Directory to save trained models")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    train_models(
        gender=args.gender,
        start_season=args.start_season,
        end_season=args.end_season,
        val_seasons=args.val_seasons,
        test_seasons=args.test_seasons,
        model_types=args.model_types,
        use_ensemble=not args.no_ensemble,
        output_dir=args.output_dir,
        random_state=args.random_state
    ) 