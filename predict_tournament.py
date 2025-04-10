import os
import argparse
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from DataLoader import DataLoader
from FeatureEngineering import FeatureEngineer
from Model import MarchMadnessModel

def load_model(model_path: str) -> MarchMadnessModel:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the pickled model file
        
    Returns:
        Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_tournament(
    model_path: str,
    gender: str = 'M',
    season: int = 2024,
    stage: int = 2,
    output_dir: str = './predictions',
    visualize: bool = True
):
    """
    Generate predictions for a tournament.
    
    Args:
        model_path: Path to the trained model file
        gender: 'M' for men's data, 'W' for women's data
        season: Season year to predict
        stage: 1 for all possible matchups, 2 for specific bracket matchups
        output_dir: Directory to save predictions
        visualize: Whether to create visualizations
    """
    print(f"Generating predictions for {gender}'s {season} tournament using model: {model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(model_path)
    print(f"Loaded model: {model.name}")
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    
    # Create feature engineer
    print("Creating features...")
    engineer = FeatureEngineer(loader)
    
    # Generate tournament features
    matchup_features = engineer.generate_tournament_features(gender, season, stage)
    print(f"Generated features for {matchup_features.shape[0]} potential matchups")
    
    # Save original matchups for later reference
    matchups = matchup_features[['Season', 'Team1ID', 'Team2ID']].copy()
    
    # Get team mapping for readability
    mens_data, womens_data, _ = loader.split_by_gender()
    data = mens_data if gender == 'M' else womens_data
    teams = data['Teams']
    team_name_map = dict(zip(teams['TeamID'], teams['TeamName']))
    
    # Add team names to matchups
    matchups['Team1Name'] = matchups['Team1ID'].map(team_name_map)
    matchups['Team2Name'] = matchups['Team2ID'].map(team_name_map)
    
    # Drop non-feature columns before prediction
    feature_columns = [col for col in matchup_features.columns 
                      if col not in ['Season', 'Team1ID', 'Team2ID']]
    X = matchup_features[feature_columns]
    
    # Ensure features match what the model expects
    if hasattr(model, 'features') and model.features:
        # Use only the features the model was trained on, in the correct order
        missing_features = set(model.features) - set(X.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with default value of 0
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns to match what the model expects
        X = X[model.features]
    
    # Generate predictions
    print("Generating predictions...")
    probabilities = model.predict_proba(X)
    
    # Create prediction DataFrame
    predictions = matchups.copy()
    predictions['Team1WinProbability'] = probabilities
    predictions['Team2WinProbability'] = 1 - probabilities
    predictions['PredictedWinner'] = predictions['Team1ID'].where(
        predictions['Team1WinProbability'] >= 0.5, 
        predictions['Team2ID']
    )
    predictions['PredictedWinnerName'] = predictions['Team1Name'].where(
        predictions['Team1WinProbability'] >= 0.5, 
        predictions['Team2Name']
    )
    predictions['UpsetProbability'] = np.where(
        predictions['Team1WinProbability'] < 0.5,
        predictions['Team2WinProbability'],
        predictions['Team1WinProbability']
    )
    
    # Save predictions
    output_file = os.path.join(output_dir, f"{gender}_{season}_stage{stage}_predictions.csv")
    predictions.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")
    
    # Check if seed information is available for upset analysis
    if 'Team1_Seed' in matchup_features.columns and 'Team2_Seed' in matchup_features.columns:
        predictions['Team1Seed'] = matchup_features['Team1_Seed']
        predictions['Team2Seed'] = matchup_features['Team2_Seed']
        
        # Define upsets (lower seed beating higher seed)
        predictions['IsSeedUpset'] = (
            (predictions['Team1WinProbability'] >= 0.5) & (predictions['Team1Seed'] > predictions['Team2Seed']) |
            (predictions['Team1WinProbability'] < 0.5) & (predictions['Team1Seed'] < predictions['Team2Seed'])
        )
        
        # Identify high-confidence upsets
        predictions['IsHighConfidenceUpset'] = predictions['IsSeedUpset'] & (predictions['UpsetProbability'] >= 0.7)
        
        # Output upset analysis
        upsets = predictions[predictions['IsSeedUpset']].sort_values('UpsetProbability', ascending=False)
        upsets_file = os.path.join(output_dir, f"{gender}_{season}_potential_upsets.csv")
        upsets.to_csv(upsets_file, index=False)
        print(f"Saved potential upsets to {upsets_file}")
        
        if not upsets.empty:
            print("\nTop 5 most likely upsets:")
            for _, row in upsets.head(5).iterrows():
                team1 = f"{row['Team1Name']} (Seed {int(row['Team1Seed'])})"
                team2 = f"{row['Team2Name']} (Seed {int(row['Team2Seed'])})"
                winner = team1 if row['Team1WinProbability'] >= 0.5 else team2
                prob = max(row['Team1WinProbability'], row['Team2WinProbability'])
                print(f"{team1} vs {team2}: {winner} wins with {prob:.1%} probability")
    
    # Create visualizations if requested
    if visualize:
        print("Creating visualizations...")
        
        # Set up the visualization directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot win probability distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions['Team1WinProbability'], bins=20, kde=True)
        plt.axvline(0.5, color='red', linestyle='--')
        plt.title(f"{gender}'s {season} Tournament Win Probability Distribution")
        plt.xlabel('Win Probability')
        plt.ylabel('Count')
        plt.savefig(os.path.join(viz_dir, f"{gender}_{season}_win_probability_dist.png"))
        
        # Plot confidence vs matchup if stage 1 (all possible matchups)
        if stage == 1 and len(predictions) > 20:
            # Select top 20 most confident predictions
            top_confident = predictions.sort_values('UpsetProbability', ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_confident)), top_confident['UpsetProbability'], color='skyblue')
            plt.yticks(range(len(top_confident)), 
                     [f"{row['Team1Name']} vs {row['Team2Name']}" for _, row in top_confident.iterrows()])
            plt.title(f"Top 20 Most Confident Predictions ({gender}'s {season} Tournament)")
            plt.xlabel('Win Probability')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{gender}_{season}_top_confident.png"))
        
        # If seed information is available, plot upset probabilities
        if 'IsSeedUpset' in predictions.columns and predictions['IsSeedUpset'].any():
            upset_probs = upsets['UpsetProbability'].head(10)
            upset_labels = [f"{row['Team1Name']} vs {row['Team2Name']}" for _, row in upsets.head(10).iterrows()]
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(upset_probs)), upset_probs, color='salmon')
            plt.yticks(range(len(upset_probs)), upset_labels)
            plt.title(f"Top 10 Most Likely Upsets ({gender}'s {season} Tournament)")
            plt.xlabel('Upset Probability')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{gender}_{season}_upset_probabilities.png"))
    
    print(f"Prediction complete. Results saved to {output_dir}")
    return predictions

def simulate_bracket(
    model_path: str,
    gender: str = 'M',
    season: int = 2024,
    num_simulations: int = 1000,
    output_dir: str = './predictions'
):
    """
    Simulate the entire tournament bracket multiple times to generate probabilities.
    
    Args:
        model_path: Path to the trained model file
        gender: 'M' for men's data, 'W' for women's data
        season: Season year to predict
        num_simulations: Number of bracket simulations to run
        output_dir: Directory to save predictions
    """
    print(f"Simulating {gender}'s {season} tournament bracket {num_simulations} times...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(model_path)
    print(f"Loaded model: {model.name}")
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    
    # Create feature engineer
    print("Creating features...")
    engineer = FeatureEngineer(loader)
    
    # NOTE: This is a placeholder for bracket simulation functionality
    # In a complete implementation, this would:
    # 1. Set up the initial bracket structure
    # 2. Simulate each round, advancing winners based on model predictions
    # 3. Track results across multiple simulations
    # 4. Calculate advancement probabilities for each team
    
    print("NOTE: Full bracket simulation is not yet implemented")
    print("This would require additional tournament structure data and round-by-round advancement logic")
    
    # Get a sample prediction to demonstrate the concept
    predictions = predict_tournament(
        model_path, gender, season, stage=1, output_dir=output_dir, visualize=False
    )
    
    print("To implement bracket simulation, we would need to:")
    print("1. Set up the initial tournament bracket structure")
    print("2. For each simulation, predict game outcomes and advance winners through the rounds")
    print("3. Track how often each team reaches various rounds across simulations")
    print("4. Calculate final advancement probabilities for each team to each round")
    
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate March Madness tournament predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--gender", type=str, default='M', choices=['M', 'W'], help="Gender of tournament (M or W)")
    parser.add_argument("--season", type=int, default=2024, help="Season year to predict")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Prediction stage (1 for all matchups, 2 for bracket)")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Directory to save predictions")
    parser.add_argument("--no_viz", action="store_true", help="Don't create visualizations")
    parser.add_argument("--simulate", action="store_true", help="Simulate full bracket instead of individual games")
    parser.add_argument("--num_simulations", type=int, default=1000, help="Number of simulations for bracket")
    
    args = parser.parse_args()
    
    if args.simulate:
        simulate_bracket(
            model_path=args.model_path,
            gender=args.gender,
            season=args.season,
            num_simulations=args.num_simulations,
            output_dir=args.output_dir
        )
    else:
        predict_tournament(
            model_path=args.model_path,
            gender=args.gender,
            season=args.season,
            stage=args.stage,
            output_dir=args.output_dir,
            visualize=not args.no_viz
        ) 