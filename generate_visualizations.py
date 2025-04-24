import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from predict_tournament import predict_2025_tournament

def generate_comparison_visualizations():
    # Create output directories
    os.makedirs('./visualizations/ensemble', exist_ok=True)
    os.makedirs('./visualizations/xgboost', exist_ok=True)
    os.makedirs('./visualizations/comparison', exist_ok=True)
    
    # Load models
    with open('./trained_models/M_Ensemble_model.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)
    with open('./trained_models/M_XGBoost_model.pkl', 'rb') as f:
        xgboost_model = pickle.load(f)
    
    # Generate predictions for both models
    print("Generating predictions for M's 2025 tournament using model: ./trained_models/M_Ensemble_model.pkl")
    ensemble_predictions = predict_2025_tournament(
        model_path='./trained_models/M_Ensemble_model.pkl',
        gender='M',
        output_dir='./visualizations/ensemble'
    )
    
    print("Generating predictions for M's 2025 tournament using model: ./trained_models/M_XGBoost_model.pkl")
    xgboost_predictions = predict_2025_tournament(
        model_path='./trained_models/M_XGBoost_model.pkl',
        gender='M',
        output_dir='./visualizations/xgboost'
    )
    
    # Load predictions
    ensemble_preds = pd.read_csv('./visualizations/ensemble/M_2025_first_round_predictions.csv')
    xgboost_preds = pd.read_csv('./visualizations/xgboost/M_2025_first_round_predictions.csv')
    
    # Create comparison visualizations
    plt.figure(figsize=(15, 10))
    
    # Win probability distribution comparison
    plt.subplot(2, 2, 1)
    sns.histplot(ensemble_preds['Team1WinProbability'], label='Ensemble', alpha=0.5)
    sns.histplot(xgboost_preds['Team1WinProbability'], label='XGBoost', alpha=0.5)
    plt.title('Win Probability Distribution Comparison')
    plt.xlabel('Win Probability')
    plt.ylabel('Count')
    plt.legend()
    
    # Upset probability comparison
    plt.subplot(2, 2, 2)
    ensemble_upsets = ensemble_preds[ensemble_preds['IsSeedUpset']]['UpsetProbability']
    xgboost_upsets = xgboost_preds[xgboost_preds['IsSeedUpset']]['UpsetProbability']
    sns.histplot(ensemble_upsets, label='Ensemble', alpha=0.5)
    sns.histplot(xgboost_upsets, label='XGBoost', alpha=0.5)
    plt.title('Upset Probability Distribution Comparison')
    plt.xlabel('Upset Probability')
    plt.ylabel('Count')
    plt.legend()
    
    # Region-wise upset comparison
    plt.subplot(2, 2, 3)
    region_upsets = pd.DataFrame({
        'Region': ensemble_preds['Region'],
        'Ensemble': ensemble_preds['IsSeedUpset'],
        'XGBoost': xgboost_preds['IsSeedUpset']
    })
    region_upsets = region_upsets.groupby('Region').mean()
    region_upsets.plot(kind='bar')
    plt.title('Region-wise Upset Probability Comparison')
    plt.xlabel('Region')
    plt.ylabel('Probability of Upset')
    plt.legend()
    
    # Seed difference vs upset probability
    plt.subplot(2, 2, 4)
    seed_diff = abs(ensemble_preds['Team1Seed'] - ensemble_preds['Team2Seed'])
    plt.scatter(seed_diff, ensemble_preds['UpsetProbability'], label='Ensemble', alpha=0.5)
    plt.scatter(seed_diff, xgboost_preds['UpsetProbability'], label='XGBoost', alpha=0.5)
    plt.title('Seed Difference vs Upset Probability')
    plt.xlabel('Seed Difference')
    plt.ylabel('Upset Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./visualizations/comparison/model_comparison.png')
    
    # Create individual model visualizations
    for model_name, predictions in [('Ensemble', ensemble_preds), ('XGBoost', xgboost_preds)]:
        # Win probability distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions['Team1WinProbability'], bins=20, kde=True)
        plt.axvline(0.5, color='red', linestyle='--')
        plt.title(f"{model_name} Model Win Probability Distribution")
        plt.xlabel('Win Probability')
        plt.ylabel('Count')
        plt.savefig(f'./visualizations/{model_name.lower()}/win_probability_dist.png')
        
        # Upset probabilities
        upsets = predictions[predictions['IsSeedUpset']].sort_values('UpsetProbability', ascending=False)
        plt.figure(figsize=(12, 8))
        upset_probs = upsets['UpsetProbability'].head(10)
        upset_labels = [f"{row['Team1Name']} vs {row['Team2Name']}" for _, row in upsets.head(10).iterrows()]
        bars = plt.barh(range(len(upset_probs)), upset_probs, color='salmon')
        plt.yticks(range(len(upset_probs)), upset_labels)
        plt.title(f"{model_name} Model Top 10 Most Likely Upsets")
        plt.xlabel('Upset Probability')
        plt.tight_layout()
        plt.savefig(f'./visualizations/{model_name.lower()}/upset_probabilities.png')
    
    print("Comparison visualizations generated and saved to ./visualizations/comparison/")

if __name__ == "__main__":
    generate_comparison_visualizations() 