import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import MarchMadnessModel

class RandomForestModel(MarchMadnessModel):
    """
    Random Forest model for predicting March Madness game outcomes.
    """
    
    def __init__(
        self, 
        name: str = "RandomForest", 
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[str] = "sqrt",
        class_weight: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the Random Forest model.
        
        Args:
            name: Name identifier for the model
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            class_weight: Weights associated with classes
            random_state: Random seed for reproducibility
            n_jobs: Number of CPU cores to use
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit the Random Forest model on the training data.
        
        Args:
            X: Feature matrix
            y: Target values (1 for team1 win, 0 for team2 win)
        """
        self.features = X.columns.tolist()
        
        # Fit the model
        self.model.fit(X, y)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
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
        
        # Get probability of team1 winning
        return self.model.predict_proba(X)[:, 1] 