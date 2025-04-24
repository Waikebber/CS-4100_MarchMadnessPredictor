import pandas as pd
import numpy as np
from FeatureEngineering import FeatureEngineer
from DataLoader import DataLoader
from predict_tournament import load_model
import os

def load_2025_seeds():
    """Load the 2025 tournament seeds."""
    seeds = pd.read_csv('data/MNCAATourneySeeds_2025.csv')
    # Clean and convert seeds to integers
    seeds['Seed'] = pd.to_numeric(seeds['Seed'], errors='coerce').fillna(0).astype(int)
    return seeds

def create_first_round_matchups(seeds):
    """Create first round matchups based on seed numbers."""
    matchups = []
    regions = seeds['Region'].unique()
    
    for region in regions:
        region_seeds = seeds[seeds['Region'] == region].sort_values('Seed')
        num_teams = len(region_seeds)
        
        # Create matchups (1v16, 2v15, etc.)
        for i in range(num_teams // 2):
            team1 = region_seeds.iloc[i]
            team2 = region_seeds.iloc[num_teams - 1 - i]
            matchups.append({
                'Season': 2025,
                'Team1ID': team1['TeamID'],
                'Team1Name': team1['TeamName'],
                'Team1Seed': team1['Seed'],
                'Team2ID': team2['TeamID'],
                'Team2Name': team2['TeamName'],
                'Team2Seed': team2['Seed'],
                'Region': region
            })
    
    return pd.DataFrame(matchups)

def generate_bracket_predictions():
    """Generate predictions for the 2025 tournament bracket."""
    # Initialize data loader and feature engineer
    data_loader = DataLoader('data')
    feature_engineer = FeatureEngineer(data_loader)
    
    # Load models
    ensemble_model = load_model('trained_models/M_Ensemble_model.pkl')
    xgboost_model = load_model('trained_models/M_XGBoost_model.pkl')
    
    # Create matchups
    seeds = load_2025_seeds()
    matchups = create_first_round_matchups(seeds)
    
    # Generate features
    features = feature_engineer.generate_tournament_features('M', 2025)
    
    # Filter features to only include our first round matchups
    features = features[
        features.apply(lambda row: any(
            (row['Team1ID'] == m['Team1ID'] and row['Team2ID'] == m['Team2ID']) or
            (row['Team1ID'] == m['Team2ID'] and row['Team2ID'] == m['Team1ID'])
            for _, m in matchups.iterrows()
        ), axis=1)
    ]
    
    # Store metadata columns
    metadata = features[['Season', 'Team1ID', 'Team2ID']].copy()
    
    # Remove metadata columns for prediction
    feature_cols = [col for col in features.columns if col not in ['Season', 'Team1ID', 'Team2ID']]
    features_for_prediction = features[feature_cols]
    
    # Make predictions
    ensemble_preds = ensemble_model.predict_proba(features_for_prediction)
    xgboost_preds = xgboost_model.predict_proba(features_for_prediction)
    
    # Add predictions to matchups
    for i, row in matchups.iterrows():
        feature_row = metadata[
            ((metadata['Team1ID'] == row['Team1ID']) & (metadata['Team2ID'] == row['Team2ID'])) |
            ((metadata['Team1ID'] == row['Team2ID']) & (metadata['Team2ID'] == row['Team1ID']))
        ].iloc[0]
        
        if feature_row['Team1ID'] == row['Team1ID']:
            matchups.at[i, 'EnsembleWinProb'] = ensemble_preds[i]
            matchups.at[i, 'XGBoostWinProb'] = xgboost_preds[i]
        else:
            matchups.at[i, 'EnsembleWinProb'] = 1 - ensemble_preds[i]
            matchups.at[i, 'XGBoostWinProb'] = 1 - xgboost_preds[i]
    
    # Save predictions
    os.makedirs('predictions', exist_ok=True)
    matchups.to_csv('predictions/2025_tournament_predictions.csv', index=False)
    
    return matchups

def print_predictions(matchups):
    """Print predictions in a readable format."""
    print("\n2025 NCAA Tournament First Round Predictions\n")
    print("Format: [Seed] Team1 vs [Seed] Team2 (Ensemble Win% | XGBoost Win%)")
    print("-" * 100)
    
    # Sort by region and seed
    matchups = matchups.sort_values(['Region', 'Team1Seed'])
    
    current_region = None
    for _, matchup in matchups.iterrows():
        if current_region != matchup['Region']:
            current_region = matchup['Region']
            print(f"\n{current_region} Region:")
            print("-" * 100)
        
        print(f"[{matchup['Team1Seed']:2d}] {matchup['Team1Name']:<20} vs [{matchup['Team2Seed']:2d}] {matchup['Team2Name']:<20}")
        print(f"Ensemble: {matchup['Team1Name']:<20} {matchup['EnsembleWinProb']:.1%} | {matchup['Team2Name']:<20} {(1 - matchup['EnsembleWinProb']):.1%}")
        print(f"XGBoost: {matchup['Team1Name']:<20} {matchup['XGBoostWinProb']:.1%} | {matchup['Team2Name']:<20} {(1 - matchup['XGBoostWinProb']):.1%}")
        print("-" * 100)

if __name__ == "__main__":
    matchups = generate_bracket_predictions()
    print_predictions(matchups) 