import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union, Dict, Any
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import MarchMadnessModel

# Check if tensorflow is available in the environment and import it
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class NeuralNetworkModel(MarchMadnessModel):
    """
    Neural Network model for predicting March Madness game outcomes.
    """
    
    def __init__(
        self, 
        name: str = "NeuralNetwork", 
        hidden_layers: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
        verbose: int = 0
    ):
        """
        Initialize the Neural Network model.
        
        Args:
            name: Name identifier for the model
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs to train
            patience: Number of epochs with no improvement after which training will be stopped
            random_state: Random seed for reproducibility
            verbose: Verbosity mode (0, 1, or 2)
        """
        super().__init__(name)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please install it with 'pip install tensorflow'")
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def _build_model(self, input_dim: int) -> None:
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Dimensionality of the input features
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], input_dim=input_dim, activation='relu'))
        if self.batch_norm:
            model.add(BatchNormalization())
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            if self.batch_norm:
                model.add(BatchNormalization())
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        self.model = model
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            validation_data: Optional[Tuple[pd.DataFrame, np.ndarray]] = None) -> None:
        """
        Fit the Neural Network model on the training data.
        
        Args:
            X: Feature matrix
            y: Target values (1 for team1 win, 0 for team2 win)
            validation_data: Optional tuple of (X_val, y_val) for validation
        """
        self.features = X.columns.tolist()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build the model
        self._build_model(input_dim=X.shape[1])
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]
        
        # Prepare validation data if provided
        validation_split = 0.2
        val_data = None
        
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            val_data = (X_val_scaled, y_val)
            validation_split = 0.0
        
        # Train the model
        self.history = self.model.fit(
            X_scaled, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        # Neural networks don't have built-in feature importance,
        # so we'll use a simple permutation approach later if needed
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
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def get_feature_importance(self, X: Optional[pd.DataFrame] = None, 
                               y: Optional[np.ndarray] = None, 
                               method: str = 'permutation', 
                               n_repeats: int = 10) -> pd.DataFrame:
        """
        Get feature importance for the Neural Network model using permutation importance.
        Note: This requires the test data to compute.
        
        Args:
            X: Feature matrix for permutation importance calculation
            y: True target values
            method: Method to calculate feature importance ('permutation' is the only supported option)
            n_repeats: Number of times to permute each feature
            
        Returns:
            DataFrame with feature names and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting feature importance")
        
        if X is None or y is None:
            raise ValueError("X and y must be provided to calculate feature importance for neural networks")
        
        if method != 'permutation':
            raise ValueError("Only 'permutation' method is supported for neural network feature importance")
        
        from sklearn.inspection import permutation_importance
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Create a wrapper for the TensorFlow model that has a predict_proba method
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
                self._estimator_type = "classifier"  # Explicitly indicate this is a classifier
                self.classes_ = np.array([0, 1])  # Binary classification
                
            def predict_proba(self, X):
                # Reshape output to match what sklearn expects (n_samples, n_classes)
                probs = self.model.predict(X, verbose=0).flatten()
                return np.column_stack((1 - probs, probs))
                
            def fit(self, X, y):
                # This is just a placeholder to satisfy the scikit-learn API
                # The actual model is already fitted
                return self
        
        # Wrap the model
        model_wrapper = ModelWrapper(self.model)
        
        # Calculate permutation importance
        result = permutation_importance(
            model_wrapper, X_scaled, y, 
            n_repeats=n_repeats, 
            random_state=self.random_state,
            scoring='neg_log_loss'
        )
        
        # Store the feature importance
        self.feature_importance = result.importances_mean
        
        return pd.DataFrame({
            'Feature': self.features,
            'Importance': self.feature_importance
        }).sort_values('Importance', ascending=False) 