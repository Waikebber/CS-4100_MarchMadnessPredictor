import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any
from sklearn.linear_model import LogisticRegression

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import MarchMadnessModel

class EnsembleModel(MarchMadnessModel):
    """
    Ensemble model that combines predictions from multiple base models.
    This can improve prediction accuracy and provide more robust predictions.
    """
    
    def __init__(
        self, 
        name: str = "Ensemble",
        base_models: List[MarchMadnessModel] = None,
        weights: Optional[List[float]] = None,
        meta_model: Optional[Any] = None,
        use_meta_model: bool = False,
        meta_model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ensemble model.
        
        Args:
            name: Name identifier for the model
            base_models: List of fitted base models to include in the ensemble
            weights: Optional weights for each base model (must sum to 1.0)
            meta_model: Optional meta model to combine base model predictions
            use_meta_model: Whether to use a meta model instead of weighted average
            meta_model_params: Parameters for the meta model if not provided
        """
        super().__init__(name)
        self.base_models = base_models or []
        self.weights = weights
        self.meta_model = meta_model
        self.use_meta_model = use_meta_model
        self.meta_model_params = meta_model_params or {}
        
        # Validate weights if provided
        if self.weights is not None:
            if len(self.weights) != len(self.base_models):
                raise ValueError(f"Number of weights ({len(self.weights)}) must match number of base models ({len(self.base_models)})")
            if abs(sum(self.weights) - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights)}")
        
        # Initialize meta model if needed
        if self.use_meta_model and self.meta_model is None:
            self.meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, **self.meta_model_params)
    
    def add_model(self, model: MarchMadnessModel, weight: Optional[float] = None) -> None:
        """
        Add a new model to the ensemble.
        
        Args:
            model: The model to add
            weight: Optional weight for the model
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before adding to ensemble")
        
        self.base_models.append(model)
        
        # Update weights if using weighted average
        if not self.use_meta_model:
            if weight is not None:
                if self.weights is None:
                    # Initialize with equal weights
                    n_models = len(self.base_models)
                    self.weights = [(1.0 - weight) / (n_models - 1)] * (n_models - 1) + [weight]
                else:
                    # Add the new weight and rescale
                    total = sum(self.weights)
                    self.weights = [w * (1.0 - weight) / total for w in self.weights] + [weight]
            else:
                # Equal weights
                n_models = len(self.base_models)
                self.weights = [1.0 / n_models] * n_models
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit the ensemble model. If using meta model, train it on base model predictions.
        If base models are not fitted, this will raise an error.
        
        Args:
            X: Feature matrix
            y: Target values (1 for team1 win, 0 for team2 win)
        """
        if not self.base_models:
            raise ValueError("No base models provided for ensemble")
        
        # Check if all base models are fitted
        for model in self.base_models:
            if not model.is_fitted:
                raise ValueError(f"All base models must be fitted before ensemble fit. Model {model.name} is not fitted.")
        
        # For meta model, we need to train on base model predictions
        if self.use_meta_model:
            # Get predictions from base models
            meta_features = np.column_stack([model.predict_proba(X) for model in self.base_models])
            
            # Fit the meta model
            self.meta_model.fit(meta_features, y)
            
            # Store feature names for meta model
            self.features = [f"{model.name}_prob" for model in self.base_models]
        else:
            # For weighted average, we don't need additional training
            # Just make sure weights are set
            if self.weights is None:
                n_models = len(self.base_models)
                self.weights = [1.0 / n_models] * n_models
                
            # Collect all unique features from base models
            self.features = []
            for model in self.base_models:
                self.features.extend([f for f in model.features if f not in self.features])
        
        self.is_fitted = True
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict win probabilities for games using the ensemble.
        
        Args:
            X: Feature matrix for games to predict
            
        Returns:
            Array of win probabilities for team1
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from base models
        base_preds = np.column_stack([model.predict_proba(X) for model in self.base_models])
        
        if self.use_meta_model:
            # Use meta model to combine predictions
            return self.meta_model.predict_proba(base_preds)[:, 1]
        else:
            # Use weighted average
            return np.sum(base_preds * np.array(self.weights).reshape(1, -1), axis=1)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance for the ensemble model.
        
        Returns:
            DataFrame with feature names and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before extracting feature importance")
        
        if self.use_meta_model and hasattr(self.meta_model, 'coef_'):
            # For meta model, return coefficients if available
            importance = np.abs(self.meta_model.coef_[0])
            return pd.DataFrame({
                'Feature': self.features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
        else:
            # For weighted average, aggregate feature importance from base models
            # weighted by the ensemble weights
            importance_df = pd.DataFrame()
            
            for i, model in enumerate(self.base_models):
                model_importance = model.get_feature_importance()
                if model_importance is not None:
                    model_importance['Weight'] = self.weights[i]
                    importance_df = pd.concat([importance_df, model_importance])
            
            if importance_df.empty:
                return None
            
            # Aggregate by feature
            agg_importance = importance_df.groupby('Feature').apply(
                lambda x: np.sum(x['Importance'] * x['Weight']) / np.sum(x['Weight'])
            ).reset_index(name='Importance')
            
            return agg_importance.sort_values('Importance', ascending=False) 