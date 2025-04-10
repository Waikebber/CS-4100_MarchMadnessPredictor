from models.LogisticRegressionModel import LogisticRegressionModel
from models.RandomForestModel import RandomForestModel
from models.XGBoostModel import XGBoostModel
from models.NeuralNetworkModel import NeuralNetworkModel

# Check if models are available
MODELS = {
    "LogisticRegression": LogisticRegressionModel,
    "RandomForest": RandomForestModel
}

# Check if XGBoost is available
try:
    import xgboost
    MODELS["XGBoost"] = XGBoostModel
except ImportError:
    pass

# Check if TensorFlow is available
try:
    import tensorflow
    MODELS["NeuralNetwork"] = NeuralNetworkModel
except ImportError:
    pass

__all__ = [
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
    "NeuralNetworkModel",
    "MODELS"
] 