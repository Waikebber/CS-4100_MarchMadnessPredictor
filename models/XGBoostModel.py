import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import MarchMadnessModel

# Check if xgboost is available in the environment and import it
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class XGBoostModel(MarchMadnessModel):
    """
    XGBoost model for predicting March Madness game outcomes.
    """
    
    def __init__(
        self, 
        name: str = "XGBoost", 
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        objective: str = "binary:logistic",
        eval_metric: str = "logloss",
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the XGBoost model.
        
        Args:
            name: Name identifier for the model
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Step size shrinkage used to prevent overfitting
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            objective: Objective function to minimize
            eval_metric: Evaluation metric for validation data
            random_state: Random seed for reproducibility
            n_jobs: Number of CPU cores to use
            early_stopping_rounds: Stop training if validation performance does not improve for this many rounds
            **kwargs: Additional parameters to pass to XGBoost
        """
        super().__init__(name)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.kwargs
        )
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            eval_set: Optional[List[Tuple[pd.DataFrame, np.ndarray]]] = None) -> None:
        """
        Fit the XGBoost model on the training data.
        
        Args:
            X: Feature matrix
            y: Target values (1 for team1 win, 0 for team2 win)
            eval_set: Optional evaluation set for early stopping
        """
        self.features = X.columns.tolist()
        
        # Fit the model
        fit_params = {}
        if eval_set is not None and self.early_stopping_rounds is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            fit_params['verbose'] = False
        
        self.model.fit(X, y, **fit_params)
        
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
    
    def get_feature_importance(self, importance_type: str = 'weight') -> pd.DataFrame:
        """
        Get feature importance for the XGBoost model.
        
        Args:
            importance_type: Type of feature importance ('weight', 'gain', or 'cover')
            
        Returns:
            DataFrame with feature names and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting feature importance")
        
        # For XGBoost, we can get different types of feature importance
        if importance_type == 'weight':
            importance = self.model.feature_importances_
        else:
            # Get feature importance from booster
            importance = self.model.get_booster().get_score(importance_type=importance_type)
            importance = np.array([importance.get(f, 0) for f in self.features])
        
        return pd.DataFrame({
            'Feature': self.features,
            'Importance': importance
        }).sort_values('Importance', ascending=False) 