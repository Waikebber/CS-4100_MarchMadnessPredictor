# CS-4100 March Madness Predictor

A machine learning system to predict NCAA March Madness basketball tournament outcomes.

## 🛠️ Environment Setup

To get started, create the conda environment using the provided `environment.yml` file. This will install all necessary dependencies for the project.

### Step 1: Create the Environment
```bash
conda env create -f environment.yml
```
- ⏳ *Note: This may take a few minutes to complete.*

### Step 2: Activate the Environment
```bash
conda activate marchmadness
```

## 📂 Data Setup

To set up the dataset for this project:

```bash
python data-setup.py <zip_file>
```

This will:
- Create a `./data` folder if it doesn't already exist
- Extract the contents of the zip file into `./data`

### Default Behavior
If no zip file is specified, the script will default to:
```python
ZIP_FILE = "./march-machine-learning-mania-2025.zip"
```

So you can simply run:
```bash
python data-setup.py
```

## 🧠 Machine Learning Models

This project implements several machine learning models for predicting March Madness game outcomes:

1. **Logistic Regression**: Baseline model for comparison
2. **Random Forest**: Ensemble of decision trees for robust predictions
3. **XGBoost**: Gradient boosting model for high accuracy
4. **Neural Network**: Deep learning model for complex pattern recognition
5. **Ensemble Model**: Combines predictions from multiple models

## 🔍 Feature Engineering

The project includes a sophisticated feature engineering pipeline that creates:

- Team-level season statistics (scoring, efficiency, etc.)
- Ranking and seed-based features
- Comparative features between matchups
- Tournament-specific features

## 🚂 Training Models

To train the prediction models, use:

```bash
python train_models.py [OPTIONS]
```

### Options
- `--gender`: Gender of tournament (M or W) - default: M
- `--start_season`: First season to include in training - default: 2003
- `--end_season`: Last season to include in training/testing - default: 2023
- `--val_seasons`: Seasons to use for validation (e.g., `--val_seasons 2022`)
- `--test_seasons`: Seasons to use for testing (e.g., `--test_seasons 2023`)
- `--model_types`: Specific models to train (e.g., `--model_types LogisticRegression RandomForest`)
- `--no_ensemble`: Don't create an ensemble model
- `--output_dir`: Directory to save trained models - default: ./trained_models
- `--random_state`: Random seed for reproducibility - default: 42

### Examples

Train all models on men's tournament data:
```bash
python train_models.py --gender M --start_season 2003 --end_season 2023
```

Train only Random Forest and XGBoost models on women's tournament data:
```bash
python train_models.py --gender W --model_types RandomForest XGBoost
```

## 🔮 Generating Predictions

To generate tournament predictions using a trained model:

```bash
python predict_tournament.py --model_path <PATH_TO_MODEL> [OPTIONS]
```

### Options
- `--model_path`: Path to the trained model file (REQUIRED)
- `--gender`: Gender of tournament (M or W) - default: M
- `--season`: Season year to predict - default: 2024
- `--stage`: Prediction stage (1 for all matchups, 2 for bracket) - default: 1
- `--output_dir`: Directory to save predictions - default: ./predictions
- `--no_viz`: Don't create visualizations
- `--simulate`: Simulate full bracket instead of individual games
- `--num_simulations`: Number of simulations for bracket - default: 1000

### Examples

Generate predictions for the 2024 Men's tournament using a trained XGBoost model:
```bash
python predict_tournament.py --model_path ./trained_models/M_XGBoost_model.pkl --season 2024
```

Generate predictions for the 2024 Women's tournament with the ensemble model:
```bash
python predict_tournament.py --model_path ./trained_models/W_Ensemble_model.pkl --gender W --season 2024
```

## 📊 Project Structure

```
.
├── data/                      # Extracted data files
├── models/                    # Machine learning model implementations
│   ├── LogisticRegressionModel.py
│   ├── RandomForestModel.py
│   ├── XGBoostModel.py
│   ├── NeuralNetworkModel.py
│   ├── EnsembleModel.py
│   └── __init__.py
├── predictions/               # Generated predictions
├── trained_models/            # Trained model files
├── data-setup.py              # Script to extract data
├── DataLoader.py              # Data loading utility
├── FeatureEngineering.py      # Feature creation pipeline
├── Model.py                   # Base model class
├── train_models.py            # Model training script
├── predict_tournament.py      # Tournament prediction script
├── environment.yml            # Conda environment file
└── README.md                  # Project documentation
```

## 👥 Team Members

- Kai Webber: Data collection and preprocessing
- Arjun Avinash: Model development and architecture
- Akash Alaparthi: Model evaluation and visualization