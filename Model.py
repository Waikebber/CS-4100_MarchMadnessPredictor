import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, brier_score_loss
from typing import Dict, Tuple, List, Any, Optional

class MarchMadnessModel(ABC):
    """
    Abstract base class for all March Madness prediction models.
    This defines the interface that all models should implement.
    """
    
    def __init__(self, name: str):
        """
        Initialize the model with a name for identification.
        
        Args:
            name: A string identifier for the model
        """
        self.name = name
        self.features = []
        self.feature_importance = None
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Train the model on the given data.
        
        Args:
            X: Feature matrix
            y: Target values (1 for team1 win, 0 for team2 win)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict win probabilities for games.
        
        Args:
            X: Feature matrix for games to predict
            
        Returns:
            Array of win probabilities for team1
        """
        pass
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict winners for games.
        
        Args:
            X: Feature matrix for games to predict
            threshold: Probability threshold for predicting a win
            
        Returns:
            Array of binary predictions (1 for team1 win, 0 for team2 win)
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix
            y: True outcomes
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)
        
        metrics = {
            'log_loss': log_loss(y, y_prob),
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob),
            'brier_score': brier_score_loss(y, y_prob)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available for the model.
        
        Returns:
            DataFrame with feature names and their importance scores, or None if not available
        """
        if self.feature_importance is None:
            return None
        
        return pd.DataFrame({
            'Feature': self.features,
            'Importance': self.feature_importance
        }).sort_values('Importance', ascending=False)
    
    def simulate_tournament(self, bracket: Dict, team_features: Dict[int, pd.DataFrame]) -> Dict:
        """
        Simulate a tournament based on the current model.
        
        Args:
            bracket: Dictionary representing the tournament structure
            team_features: Dictionary mapping team IDs to their features
            
        Returns:
            Dictionary with simulation results
        """
        # Implementation will depend on how we structure the bracket data
        # This is a placeholder for now
        pass 