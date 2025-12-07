# --- Core Libraries ---
import pandas as pd
import os
import joblib
import json

# --- Sklearn/Modeling Libraries ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, recall_score
import xgboost as xgb

# --- MLOps/Tracking Libraries ---
import mlflow 
from huggingface_hub import HfApi

# --- Configuration ---
# NOTE: Replace <-----Hugging Face User ID -----> with your actual ID.
HF_USER_ID = "Sraja0310" 
DATASET_REPO_ID = f"{HF_USER_ID}/wellness-tourism-data"
MODEL_REPO_ID = f"{HF_USER_ID}/wellness-tourism-predictor" # For model registration

# File paths from the data preparation step (prep.py)
X_TRAIN_PATH = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/split_data/Xtrain_tourism.csv"
Y_TRAIN_PATH = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/split_data/ytrain_tourism.csv"
X_TEST_PATH = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/split_data/Xtest_tourism.csv"
Y_TEST_PATH = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/split_data/ytest_tourism.csv"

# --- 1. Load Data ---
try:
    Xtrain = pd.read_csv(X_TRAIN_PATH)
    ytrain = pd.read_csv(Y_TRAIN_PATH).squeeze() # Use squeeze() for a Series
    Xtest = pd.read_csv(X_TEST_PATH)
    ytest = pd.read_csv(Y_TEST_PATH).squeeze()
    print("Train and test data loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Error loading data: {e}. Ensure HF files are correctly named and uploaded.")
    raise

target = 'ProdTaken'

# --- 2. Feature Definition (Aligned with Data Dictionary) ---
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'MaritalStatus',
    'Designation', 'ProductPitched'
]

# Features that are ordinal/numeric but were incorrectly listed as categorical in the previous code:
interaction_numeric_features = [
    'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]

# Ensure all features in the data are correctly categorized
final_numeric_features = [col for col in (numeric_features + interaction_numeric_features) if col in Xtrain.columns]
final_categorical_features = [col for col in categorical_features if col in Xtrain.columns]

# --- 3. Preprocessing Pipeline Definition ---
# Set the class weight to handle class imbalance (Crucial for imbalanced targets like ProdTaken)
# Weight = Count of Majority Class (0) / Count of Minority Class (1)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Calculated positive class weight (scale_pos_weight): {class_weight:.2f}")

# Define the preprocessing steps using ColumnTransformer
preprocessor = make_column_transformer(
    (StandardScaler(), final_numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), final_categorical_features),
    remainder='passthrough' # Keep any columns not explicitly listed (shouldn't be any)
)

# Define base XGBoost model
# Setting use_label_encoder=False and eval_metric='logloss' suppresses common XGBoost warnings/errors
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)

# Model pipeline combines preprocessing and the model
model_pipeline = make_pipeline(preprocessor, xgb_model)


# --- 4. Hyperparameter Tuning and MLflow Logging ---
# Define hyperparameter grid for tuning (prefixed with the step name 'xgbclassifier__')
param_grid = {
    'xgbclassifier__n_estimators': [50, 75], 
    'xgbclassifier__max_depth': [2, 3], 
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.6], 
}

# The MLflow run is now streamlined to avoid API rate limits (solves previous error)
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"\n--- Starting MLflow Run: {run_id} ---")

    # Hyperparameter tuning using GridSearchCV (scoring on 'recall' due to class imbalance)
    print("Starting GridSearchCV (Tuning for Recall)...")
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='recall') 
    grid_search.fit(Xtrain, ytrain)
    print("GridSearchCV complete.")

    # Log Best Parameters and CV Score
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_recall_score", grid_search.best_score_)
    
    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # --- Evaluation ---
    classification_threshold = 0.45 # Using a slightly lower threshold to prioritize recall

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log Final Test Metrics
    mlflow.log_metrics({
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "classification_threshold": classification_threshold,
    })
    
    # --- 5. Model Registration (Criteria 5) ---
    MODEL_FILENAME = "best_model.joblib"
    
    # Save model locally
    joblib.dump(best_model, MODEL_FILENAME)
    print(f"Model saved locally as {MODEL_FILENAME}.")

    # Log model artifact to MLflow
    mlflow.log_artifact(MODEL_FILENAME, "model_artifact")
    print("Model artifact logged to MLflow.")

    # Register best model to Hugging Face Model Hub (CRITERIA 5)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=MODEL_FILENAME,
        path_in_repo=MODEL_FILENAME,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        token=os.getenv("HF_TOKEN")
    )
    print(f"Model registered to Hugging Face Model Hub: {MODEL_REPO_ID}")

print("\nPipeline execution complete. Model is registered and tracked.")
