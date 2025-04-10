import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import MarchMadnessModel

class LogisticRegressionModel(MarchMadnessModel):
    """
    Logistic Regression model for predicting March Madness game outcomes.
    This serves as the baseline model for the project.
    """
    
    def __init__(
        self, 
        name: str = "LogisticRegression", 
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the logistic regression model.
        
        Args:
            name: Name identifier for the model
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
            class_weight: Weights associated with classes
            random_state: Random seed for reproducibility
            n_jobs: Number of CPU cores to use
        """
        super().__init__(name)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit the logistic regression model on the training data.
        
        Args:
            X: Feature matrix
            y: Target values (1 for team1 win, 0 for team2 win)
        """
        self.features = X.columns.tolist()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        
        # Get feature importance (coefficients for logistic regression)
        self.feature_importance = np.abs(self.model.coef_[0])
        
        self.is_fitted = True
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict win probabilities for games.
        
        Args:
            X: Feature matrix for games to predict
            
        Returns:
            Array of win probabilities for team1
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Get probability of team1 winning
        return self.model.predict_proba(X_scaled)[:, 1] 